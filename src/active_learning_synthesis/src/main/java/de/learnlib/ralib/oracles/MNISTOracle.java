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
import de.learnlib.ralib.data.MNISTDataValue;
import de.learnlib.ralib.words.PSymbolInstance;
import net.automatalib.word.Word;
import de.learnlib.ralib.tools.config.Configuration;

import java.math.BigDecimal;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Map;

public class MNISTOracle implements MyOracle {
    private final Predictor<float[], Float> predictor;
    private Model model;
    private NDManager manager;
    private Configuration config;
    private Map<String, Integer> patternToLabel;
    private Device currentDevice;

    public MNISTOracle(Configuration config) {
        this.config = config;

        // 1) Force CPU immediately:
        currentDevice = Device.cpu();

        // 2) Create an NDManager (CPU by default)
        manager = NDManager.newBaseManager();

        // 4) Load the TorchScript model from disk, forcing CPU via map_location
        String modelFileName = this.config.getProperty("modelFileName");
        Path modelPath = Paths.get(modelFileName);

        try {
            // Use MNISTTranslator that reshapes to (1, 28, 28) for the LSTM
            MNISTTranslator translator = new MNISTTranslator();

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
     * Decodes an integer row ID to 28 normalized pixel values.
     * Row ID is a 28-bit integer where each bit represents a pixel.
     * Returns normalized values: 0.0 (off) or 1.0 (on).
     */
    private static float[] decodeRowIdToNormalizedPixels(long rowId) {
        float[] pixels = new float[28];
        for (int j = 0; j < 28; j++) {
            // Extract bit j from the left (MSB first)
            // Bit 0 is the leftmost pixel (bit 27 of the integer)
            // Return normalized: 0.0 or 1.0 (equivalent to 0/255 or 255/255)
            pixels[j] = ((rowId >> (27 - j)) & 1) == 1 ? 1.0f : 0.0f;
        }
        return pixels;
    }

    /**
     * Gets normalized pixel values (0-1) from a DataValue.
     * If MNISTDataValue: uses original pixels (0-255) / 255.
     * If regular DataValue: decodes row ID to binary pixels.
     */
    private static float[] getNormalizedPixels(DataValue dv) {
        if (dv instanceof MNISTDataValue) {
            MNISTDataValue mdv = (MNISTDataValue) dv;
            int[] originalPixels = mdv.getPixels();
            float[] normalized = new float[28];
            for (int j = 0; j < 28; j++) {
                normalized[j] = originalPixels[j] / 255.0f;
            }
            return normalized;
        } else {
            // Fallback: decode row ID
            BigDecimal val = (BigDecimal) dv.getValue();
            return decodeRowIdToNormalizedPixels(val.longValue());
        }
    }

    /**
     * Accepts a Word of 28 Symbols (Rows).
     * Each symbol contains pixel values that are normalized to 0-1 for the model.
     */
    public boolean accepts(Word<PSymbolInstance> trace) {
        // 1. Prepare a blank 784-pixel canvas (all zeros)
        System.out.println("Query (rowIds): " + info(trace) + " | Trace length: " + trace.length());
        // Print actual pixel arrays for each row
        for (int i = 0; i < trace.length(); i++) {
            DataValue dv = trace.getSymbol(i).getParameterValues()[0];
            if (dv instanceof MNISTDataValue) {
                int[] px = ((MNISTDataValue) dv).getPixels();
                System.out.println("Row " + i + " pixels: " + java.util.Arrays.toString(px));
            } else {
                System.out.println("Row " + i + " is not MNISTDataValue, value: " + dv.getValue());
            }
        }
        float[] fullImage = new float[784]; 
        // 2. Determine how many rows we actually have
        // RALib often queries prefixes, so trace.length() might be < 28
        int rowsToProcess = Math.min(trace.length(), 28);
    
        for (int i = 0; i < rowsToProcess; i++) {
            // Extract the DataValue from the Symbol Instance
            DataValue dv = trace.getSymbol(i).getParameterValues()[0];
            
            // Get normalized pixel values (0-1)
            float[] rowPixels = getNormalizedPixels(dv);

            
            // Copy into the flattened model input
            for (int j = 0; j < 28; j++) {
                fullImage[i * 28 + j] = rowPixels[j];
            }
        }

        // 3. Optional: Visual Debugging (only for full words)
        if (trace.length() == 28) {
            visualizeFullDigit(fullImage);
        }

        // 4. Model Inference
        try {
            // The Predictor handles the reshape(1, 28, 28) inside the Translator
            float probability = predictor.predict(fullImage);
            return probability >= 0.5f;
        } catch (Exception e) {
            System.err.println("Model inference error: " + e.getMessage());
            return false;
        }
    }

    /**
     * Visualizes the abstracted word in the console to verify Layer 2/3 alignment.
     */
    private void visualizeFullDigit(float[] pixels) {
        System.out.println("--- Abstracted Word Visualization ---");
        for (int i = 0; i < 28; i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < 28; j++) {
                sb.append(pixels[i * 28 + j] > 0.5 ? "██" : "  ");
            }
            System.out.println(sb.toString());
        }
    }


    private static class MNISTTranslator implements Translator<float[], Float> {
        @Override
        public NDList processInput(TranslatorContext ctx, float[] input) {
            // Reshape 784 pixels to (1, 28, 28) for the LSTM
            NDArray array = ctx.getNDManager().create(input).reshape(1, 28, 28);
            return new NDList(array);
        }

        @Override
        public Float processOutput(TranslatorContext ctx, NDList list) {
            return list.singletonOrThrow().getFloat();
        }

        @Override
        public Batchifier getBatchifier() { return null; }
    }

    @Override
    public void processQueries(Collection<? extends Query<PSymbolInstance, Boolean>> queries) { 
        for (Query<PSymbolInstance, Boolean> q : queries) {
            q.answer(accepts(q.getInput()));
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
}