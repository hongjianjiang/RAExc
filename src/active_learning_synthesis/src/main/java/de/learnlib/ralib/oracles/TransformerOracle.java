package de.learnlib.ralib.oracles;

import ai.djl.Device;
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

public class TransformerOracle implements MyOracle {

    private ZooModel<float[], Float> model;
    private Predictor<float[], Float> predictor;
    private NDManager manager;
    private Configuration config;
    private Device currentDevice;

    // Constants to match Python implementation
    private static final int MAX_LEN = 51;
    private static final float PAD_VALUE = -999.0f;  // Transformer padding value

    public TransformerOracle(Configuration config) {
        this.config = config;
        currentDevice = Device.cpu();
        manager = NDManager.newBaseManager();
        String modelFileName = this.config.getProperty("modelFileName");
        Path modelPath = Paths.get(modelFileName);

        try {
            Criteria<float[], Float> criteria = Criteria.builder()
                    .setTypes(float[].class, Float.class)
                    .optModelPath(modelPath)
                    .optEngine("PyTorch")
                    .optDevice(currentDevice)
                    .optOption("map_location", "cpu")
                    .optTranslator(new TorchScriptSingleSequenceTranslator())
                    .build();

            model = criteria.loadModel();
            predictor = model.newPredictor();
            System.out.println("Transformer JIT model loaded successfully from: " + modelPath);
        } catch (Exception e) {
            throw new RuntimeException("Failed to load Transformer JIT model: " + e.getMessage(), e);
        }
    }

    @Override
    public void processQueries(Collection<? extends Query<PSymbolInstance, Boolean>> queries) {
        for (Query<PSymbolInstance, Boolean> q : queries) {
            Word<PSymbolInstance> trace = q.getInput();
            q.answer(accepts(trace));
        }
    }

    public boolean accepts(Word<PSymbolInstance> trace) {
        if (trace.length() == 0) {
            TraceCache.put(trace, true);
            return true;
        }
        Boolean seen = TraceCache.get(trace);
        if (seen != null) {
            return seen;
        }

        // Convert trace to float array
        float[] seq = new float[trace.length()];
        for (int i = 0; i < trace.length(); i++) {
            DataValue dv = trace.getSymbol(i).getParameterValues()[0];
            Object raw = dv.getValue();
            BigDecimal bd;
            if (raw instanceof BigDecimal) {
                bd = (BigDecimal) raw;
            } else if (raw instanceof Integer) {
                bd = new BigDecimal((Integer) raw);
            } else {
                throw new IllegalArgumentException("Unsupported type: " + raw.getClass().getName());
            }
            seq[i] = bd.floatValue();
        }

        // Z-score normalization (matching Python exactly)
        float[] normSeq = zScoreNormalize(seq);

        // Pad to fixed length MAX_LEN (matching Python exactly)
        float[] paddedSeq = padToFixedLength(normSeq, MAX_LEN, PAD_VALUE);

        double javaProb;
        try {
            float javaLogit = predictor.predict(paddedSeq);
            javaProb = sigmoid(javaLogit);
        } catch (Exception e) {
            throw new RuntimeException("Model inference failed: " + e.getMessage(), e);
        }

        boolean accepted = javaProb > 0.5;
        TraceCache.put(trace, accepted);
        if (accepted){
            System.out.println(Arrays.toString(seq));

        }
        return accepted;
    }

    /**
     * Z-score normalization matching Python's numpy.std() with ddof=0 (population std)
     * This matches the exact behavior of numpy.std() which uses population standard deviation by default
     */
    private float[] zScoreNormalize(float[] seq) {
        if (seq.length == 0) {
            return new float[0];
        }

        // Calculate mean
        float sum = 0f;
        for (float v : seq) {
            sum += v;
        }
        float mean = sum / seq.length;

        // Calculate population standard deviation (ddof=0, matching Python's np.std() default)
        float sqSum = 0f;
        for (float v : seq) {
            sqSum += (v - mean) * (v - mean);
        }

        // Use population standard deviation (divide by n) to match Python's np.std() default
        float std;
        if (seq.length == 1) {
            std = 0f;  // Single element has no variance
        } else {
            std = (float) Math.sqrt(sqSum / seq.length);  // Population std (ddof=0)
        }

        // Normalize
        float[] normSeq = new float[seq.length];
        if (std == 0f) {
            // If std is 0, set all values to 0 (matching Python behavior)
            Arrays.fill(normSeq, 0f);
        } else {
            for (int i = 0; i < seq.length; i++) {
                normSeq[i] = (seq[i] - mean) / std;
            }
        }

        return normSeq;
    }

    /**
     * Pad sequence to fixed length, matching Python implementation exactly
     */
    private float[] padToFixedLength(float[] seq, int maxLen, float padValue) {
        float[] paddedSeq = new float[maxLen];

        if (seq.length >= maxLen) {
            // Truncate if too long
            System.arraycopy(seq, 0, paddedSeq, 0, maxLen);
        } else {
            // Copy original sequence
            System.arraycopy(seq, 0, paddedSeq, 0, seq.length);
            // Fill remaining with pad value
            Arrays.fill(paddedSeq, seq.length, maxLen, padValue);
        }

        return paddedSeq;
    }

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static class TorchScriptSingleSequenceTranslator implements Translator<float[], Float> {
        @Override
        public NDList processInput(TranslatorContext ctx, float[] input) {
            NDManager manager = ctx.getNDManager();
            // Input should be [batch_size, seq_len] = [1, MAX_LEN]
            NDArray array = manager.create(input).reshape(1, input.length);
            return new NDList(array);
        }

        @Override
        public Float processOutput(TranslatorContext ctx, NDList list) {
            NDArray output = list.singletonOrThrow();
            // Model outputs logits, return them directly for consistency with Python
            return output.getFloat(0);
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
    }

    public void close() {
        if (predictor != null) predictor.close();
        if (model != null) model.close();
        if (manager != null) manager.close();
    }
}
