package de.learnlib.ralib.oracles;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import de.learnlib.query.Query;
import de.learnlib.ralib.data.DataValue;
import de.learnlib.ralib.sul.TraceCache;
import de.learnlib.ralib.tools.config.Configuration;
import de.learnlib.ralib.words.PSymbolInstance;
import net.automatalib.word.Word;

import java.math.BigDecimal;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

public class RNNOracle implements MyOracle {

    private Model model;
    private Predictor<float[], Float> predictor;
    private NDManager manager;
    private Configuration config;
    private Map<String, Integer> patternToLabel;
    private Device currentDevice;

    // Constants matching Python implementation
    private static final int MAX_LEN = 51;
    private static final float PAD_VALUE = -999.0f; // Use -1.0f for LSTM, -999.0f for Transformer

    /**
     * Always‐CPU constructor: loads a TorchScript model onto CPU regardless of how it was originally saved.
     */
    public RNNOracle(Configuration config) {
        this.config = config;

        // 1) Force CPU immediately:
        currentDevice = Device.cpu();

        // 2) Create an NDManager (CPU by default)
        manager = NDManager.newBaseManager();

        // 3) Build the pattern‐to‐label map from the training set
//        List<String> patterns = trackPattern(trainSet);
//        patternToLabel = new HashMap<>();
//        for (int i = 0; i < trainSet.size(); i++) {
//            patternToLabel.put(patterns.get(i), labelSet.get(i));
//        }

        // 4) Load the TorchScript model from disk, forcing CPU via map_location
        String modelFileName = this.config.getProperty("modelFileName");
        Path modelPath = Paths.get(modelFileName);

        try {
            // Translator is told "Device.cpu()" so all NDArray tensors go to CPU
            RNNTranslator translator = new RNNTranslator(config, currentDevice);

            // Build Criteria with .optDevice(Device.cpu()) AND .optOption("map_location", "cpu")
            Criteria<float[], Float> criteria = Criteria.builder()
                    .setTypes(float[].class, Float.class)
                    .optModelPath(modelPath)
                    .optEngine("PyTorch")
                    .optDevice(currentDevice)                    // force CPU device
                    .optOption("map_location", "cpu")             // remap any CUDA tensors into CPU
                    .optTranslator(translator)
                    .build();

            // Load the model onto CPU
            ZooModel<float[], Float> zooModel = criteria.loadModel();
            this.model     = zooModel;
            this.predictor = zooModel.newPredictor(translator);

        } catch (Exception e) {
            throw new RuntimeException("Failed to load model on CPU (with map_location): " + e.getMessage(), e);
        }
    }

    /**
     * Extracts a comma‐separated string of raw numeric inputs from a trace.
     */
    public String info(Word<PSymbolInstance> trace) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < trace.length(); i++) {
            sb.append(trace.getSymbol(i).getParameterValues()[0].getValue());
            if (i < trace.length() - 1) {
                sb.append(',');
            }
        }
        return sb.toString();
    }

    @Override
    public void processQueries(Collection<? extends Query<PSymbolInstance, Boolean>> queries) {
        for (Query<PSymbolInstance, Boolean> q : queries) {
            Word<PSymbolInstance> trace = q.getInput();
            q.answer(accepts(trace));
        }
    }

    /**
     * Returns true if the model "accepts" this trace. The trace is normalized exactly
     * as your Python code does (z-score normalization followed by padding).
     */
    public boolean accepts(Word<PSymbolInstance> trace) {
        // 1) Check Cache
        if (trace.length() == 0) {
            // Decide: what do you want to return for empty input? False? True? 0? 1?
            TraceCache.put(trace, true); // Or true, if you want to accept the empty word.
            return true;
        }

        Boolean seen = TraceCache.get(trace);
        if (seen != null) {
            return seen;
        }

        // 2) Extract raw sequence values
        double[] rawSequence = new double[trace.length()];
        for (int i = 0; i < trace.length(); i++) {
            DataValue dv = trace.getSymbol(i).getParameterValues()[0];
            Object raw = dv.getValue();
            BigDecimal bd;
            if (raw instanceof BigDecimal) {
                bd = (BigDecimal) raw;
            } else if (raw instanceof Integer) {
                bd = new BigDecimal((Integer) raw);
            } else if (raw instanceof Double) {
                bd = BigDecimal.valueOf((Double) raw);
            } else if (raw instanceof Float) {
                bd = BigDecimal.valueOf((Float) raw);
            } else {
                throw new IllegalArgumentException("Unsupported type: " + raw.getClass().getName());
            }
            rawSequence[i] = bd.doubleValue();
        }

        // 3) Apply z-score normalization (same as Python)
        double mean = 0.0;
        for (double val : rawSequence) {
            mean += val;
        }
        mean /= rawSequence.length;

        double variance = 0.0;
        for (double val : rawSequence) {
            variance += (val - mean) * (val - mean);
        }
        double std = Math.sqrt(variance / rawSequence.length);

        float[] normalizedSequence = new float[rawSequence.length];
        if (std == 0.0) {
            // Avoid division by zero; use zeros
            Arrays.fill(normalizedSequence, 0.0f);
        } else {
            for (int i = 0; i < rawSequence.length; i++) {
                normalizedSequence[i] = (float) ((rawSequence[i] - mean) / std);
            }
        }

        // 4) Pad or truncate to MAX_LEN (same as Python)
        float[] paddedSequence = new float[MAX_LEN];
        if (normalizedSequence.length > MAX_LEN) {
            // Truncate
            System.arraycopy(normalizedSequence, 0, paddedSequence, 0, MAX_LEN);
        } else {
            // Copy normalized sequence
            System.arraycopy(normalizedSequence, 0, paddedSequence, 0, normalizedSequence.length);
            // Fill remaining with padding value
            Arrays.fill(paddedSequence, normalizedSequence.length, MAX_LEN, PAD_VALUE);
        }

        // 5) Run model prediction
        float prediction;
        try {
            prediction = predictor.predict(paddedSequence);
        } catch (Exception e) {
            throw new RuntimeException("Model inference failed (CPU): " + e.getMessage(), e);
        }

        boolean accepted = prediction > 0.5f;
//        boolean accepted = isS4Language(Arrays.stream(rawSequence).boxed().toList());
        TraceCache.put(trace, accepted);
//        System.out.println(String.format("[%s,%s,#%.8f", Arrays.toString(rawSequence), accepted ? "1]" : "0]", prediction));
        return accepted;
    }

    public static class RNNTranslator implements Translator<float[], Float> {

        private final Configuration config;
        private final Device modelDevice;

        public RNNTranslator(Configuration config, Device device) {
            this.config       = config;
            this.modelDevice  = device; // always Device.cpu()
        }

        @Override
        public NDList processInput(TranslatorContext ctx, float[] input) {
            NDManager ndManager = ctx.getNDManager();

            // For LSTM: shape should be (1, MAX_LEN, 1)
            // For Transformer: shape should be (1, MAX_LEN)
            NDArray array;

            // Check if this is for LSTM (needs 3D input) or Transformer (needs 2D input)
            // You can determine this from config or model type
            String modelType = config.getProperty("modelType", "lstm"); // default to lstm

            if ("transformer".equals(modelType.toLowerCase())) {
                // Transformer expects (batch_size, seq_len)
                array = ndManager.create(input).reshape(1, input.length);
            } else {
                // LSTM expects (batch_size, seq_len, input_size)
                array = ndManager.create(input).reshape(1, input.length);
            }

            return new NDList(array);
        }

        @Override
        public Float processOutput(TranslatorContext ctx, NDList list) {
            NDArray output;
            if (list.size() > 1) {
                output = list.get(0);
            } else {
                output = list.singletonOrThrow();
            }

            float prediction;
            long[] shape = output.getShape().getShape();
            if (shape.length == 0) {
                prediction = output.getFloat();
            } else if (shape.length == 1) {
                prediction = output.getFloat(0);
            } else if (shape.length == 2) {
                prediction = output.getFloat(0, 0);
            } else {
                throw new RuntimeException("Unexpected output shape: " + output.getShape());
            }
            return prediction;
        }

        @Override
        public Batchifier getBatchifier() {
            return null; // no batching
        }
    }

    /**
     * Clean up DJL resources.
     */
    public void close() {
        if (predictor != null) {
            predictor.close();
        }
        if (model != null) {
            model.close();
        }
        if (manager != null) {
            manager.close();
        }
    }
}
