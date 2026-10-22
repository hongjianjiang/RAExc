package de.learnlib.ralib.tools;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import de.learnlib.query.DefaultQuery;
import de.learnlib.ralib.data.Constants;
import de.learnlib.ralib.data.DataType;
import de.learnlib.ralib.data.MNISTDataValue;
import de.learnlib.ralib.learning.Hypothesis;
import de.learnlib.ralib.learning.RaLearningAlgorithm;
import de.learnlib.ralib.learning.ralambda.RaDT;
import de.learnlib.ralib.learning.ralambda.RaLambda;
import de.learnlib.ralib.learning.rastar.RaStar;
import de.learnlib.ralib.oracles.DataWordOracle;
import de.learnlib.ralib.oracles.MNISTOracle;
import de.learnlib.ralib.oracles.MyOracle;
import de.learnlib.ralib.oracles.SimulatorOracle;
import de.learnlib.ralib.oracles.TreeOracleFactory;
import de.learnlib.ralib.oracles.mto.MultiTheorySDTLogicOracle;
import de.learnlib.ralib.oracles.mto.MultiTheoryTreeOracle;
import de.learnlib.ralib.smt.ConstraintSolver;
import de.learnlib.ralib.sul.TraderSUL;
import static de.learnlib.ralib.sul.TraderSUL.INTEGER_TYPE;
import de.learnlib.ralib.theory.Theory;
import de.learnlib.ralib.tools.config.Configuration;
import de.learnlib.ralib.tools.config.ConfigurationException;
import de.learnlib.ralib.tools.config.ConfigurationOption;
import de.learnlib.ralib.tools.theories.MNISTRowTheory;
import de.learnlib.ralib.words.PSymbolInstance;
import de.learnlib.ralib.words.ParameterizedSymbol;
import net.automatalib.word.Word;

public class MNISTSimulator extends AbstractToolWithRandomWalk {

    // --- Configuration Options ---
    protected static final ConfigurationOption.StringOption OPTION_CSV_PATH = new ConfigurationOption.StringOption(
            "csvPath", "Path to the MNIST test CSV file", "mnist_test.csv", false);

    protected static final ConfigurationOption.IntegerOption OPTION_TIMEOUT = new ConfigurationOption.IntegerOption(
            "timeout", "Timeout in seconds for the run() method", 120, false);

    protected static final ConfigurationOption.StringOption OPTION_LANGUAGE = new ConfigurationOption.StringOption(
            "language", "data language identifier", "MNIST_Binary", false);

    protected static final ConfigurationOption.StringOption OPTION_RNNTYPE = new ConfigurationOption.StringOption(
            "rnn", "Model type (lstm, transformer)", "lstm", false);

    protected static final ConfigurationOption.StringOption OPTION_QUERY_RNN_FILE_NAME = new ConfigurationOption.StringOption(
            "queryRNNFileName", "Filename for saving RNN query log", "queriesRNN.log", false);

    protected static final ConfigurationOption.DoubleOption OPTION_EPSILON = new ConfigurationOption.DoubleOption(
            "epsilon", "Error rate", 0.1d, false);

    protected static final ConfigurationOption.IntegerOption OPTION_MAX_SAMPLES = new ConfigurationOption.IntegerOption(
            "maxSamples", "Maximum samples to load from CSV", 2000, false);

    // private static final ConfigurationOption[] OPTIONS = new ConfigurationOption[]{
    //         OPTION_LEARNER, OPTION_CSV_PATH, OPTION_TIMEOUT, OPTION_LANGUAGE,
    //         OPTION_RNNTYPE, OPTION_QUERY_RNN_FILE_NAME, OPTION_EPSILON, OPTION_MAX_SAMPLES
    // };

    // --- Fields ---
    private RaLearningAlgorithm rastar;
    private TreeOracleFactory hypFactory;
    private MyOracle traceOracle;
    private MultiTheorySDTLogicOracle mlo;
    private ConstraintSolver solver;
    private TraderSUL traderSUL;
    private List<Boolean> groundTruthLabels = new ArrayList<>();
    private String csvPath;
    private String language;
    private int timeoutSeconds;
    private double epsilon;
    // private String queryRNNFileName;

    @Override
    public void setup(Configuration config) throws ConfigurationException {
        super.setup(config);
        this.csvPath = OPTION_CSV_PATH.parse(config);
        this.language = OPTION_LANGUAGE.parse(config);
        this.timeoutSeconds = OPTION_TIMEOUT.parse(config);
        this.epsilon = OPTION_EPSILON.parse(config);
        // this.queryRNNFileName = OPTION_QUERY_RNN_FILE_NAME.parse(config);

        final Map<DataType, Theory> teachers = new LinkedHashMap<>();
        teachers.put(INTEGER_TYPE, new MNISTRowTheory(INTEGER_TYPE));

        // Use Double type for pixels (0.0 - 255.0)
        this.traderSUL = new TraderSUL(INTEGER_TYPE);

        // Use MNISTOracle for MNIST model inference
        traceOracle = new MNISTOracle(config);

        Constants consts = new Constants();
        solver = new ConstraintSolver();
        mlo = new MultiTheorySDTLogicOracle(consts, solver);
        MultiTheoryTreeOracle mto = new MultiTheoryTreeOracle(traceOracle, teachers, consts, solver);

        hypFactory = hyp -> {
            DataWordOracle hypOracle = new SimulatorOracle(hyp);
            return new MultiTheoryTreeOracle(hypOracle, teachers, consts, solver);
        };

        ParameterizedSymbol priceInput = traderSUL.createPrice(1).getBaseSymbol();
        ParameterizedSymbol[] inputs = new ParameterizedSymbol[]{priceInput};

        if (this.learner.equals(LEARNER_SLLAMBDA)) {
            this.rastar = new RaLambda(mto, hypFactory, mlo, consts, false, inputs);
            ((RaLambda) this.rastar).setSolver(solver);
        } else {
            this.rastar = new RaStar(mto, hypFactory, mlo, consts, false, inputs);
        }
    }

    @Override
    public void run() throws RaLibToolException {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<?> future = executor.submit(() -> {
            int rounds = 0;
            int maxSamples = 1000;

            System.out.println("Loading MNIST data from: " + csvPath);
            List<int[]> samples;
            try {
                samples = loadMNISTFromCSV(csvPath, maxSamples);
            } catch (IOException e) {
                throw new RuntimeException("CSV Load Error: " + e.getMessage());
            }

            Word<PSymbolInstance>[] batch = create_batch_word(samples);
            long startTime = System.nanoTime();

            while (rounds < 100) {
                rounds++;
                rastar.learn();
                Hypothesis hyp = rastar.getHypothesis();
                System.out.println(hyp);
                // Save model snapshot
                try {
                    String xmlOutput = parseToJson(hyp.toString()).replace("p1", "pixel");
                    File dir = new File("active_models");
                    if (!dir.exists()) dir.mkdirs();
                    Files.write(Paths.get("active_models/" + language + ".xml"), xmlOutput.getBytes());
                } catch (Exception e) {
                    System.err.println("Export failed: " + e.getMessage());
                }

                // Check Counterexamples against Ground Truth
                boolean ceFound = false;
                int errors = 0;
                for (int i = 0; i < batch.length; i++) {
                    boolean hypDecision = hyp.accepts(batch[i]);
                    boolean actualLabel = traceOracle.accepts(batch[i]);
                    System.out.println("Hypothesis Decision: " + hypDecision + " | Actual Label: " + actualLabel);
                    if (hypDecision != actualLabel) {
                        errors++;
                        if (!ceFound) {
                            // Try to add counterexample - may fail if theory can't process it
                            try {
                                DefaultQuery<PSymbolInstance, Boolean> cex = new DefaultQuery<>(batch[i], actualLabel);
                                rastar.addCounterexample(cex);
                                ceFound = true;
                                System.out.println("Added counterexample from sample " + i);
                            } catch (Exception e) {
                                System.out.println("CE rejected (sample " + i + "): " + e.getMessage());
                                // Continue to try next disagreeing sample
                            }
                        }
                    }
                }

                double accuracy = 1.0 - ((double) errors / batch.length);
                System.out.println("Round " + rounds + " | Accuracy: " + String.format("%.2f%%", accuracy * 100));

                if (!ceFound || accuracy >= (1.0 - epsilon)) break;
            }

            long elapsed = System.nanoTime() - startTime;
            System.out.println("Learning complete in " + (elapsed / 1_000_000_000) + "s");
        });

        try {
            future.get(timeoutSeconds, TimeUnit.SECONDS);
        } catch (Exception e) {
            future.cancel(true);
            throw new RaLibToolException("Run interrupted or timed out: " + e.getMessage());
        }
    }

    /**
     * Reads original MNIST CSV: col 0 is label, cols 1-784 are pixels (0-255).
     * Maps Label 1 -> TRUE, others -> FALSE.
     * Returns list of pixel arrays (784 values each).
     */
    private List<int[]> loadMNISTFromCSV(String path, int max) throws IOException {
        List<int[]> samples = new ArrayList<>();
        groundTruthLabels.clear();
        try (BufferedReader br = Files.newBufferedReader(Paths.get(path))) {
            String line;
            int count = 0;
            // No header skip - process all lines
            while ((line = br.readLine()) != null && count < max) {
                String[] parts = line.split(",");
                if (parts.length < 785) continue;  // Need label + 784 pixels
                
                int label = Integer.parseInt(parts[0].trim());
                groundTruthLabels.add(label == 1); // Ground truth is 'is it a 1?'
                
                // Parse 784 pixel values
                int[] pixels = new int[784];
                for (int i = 0; i < 784; i++) {
                    pixels[i] = Integer.parseInt(parts[i + 1].trim());
                }
                samples.add(pixels);
                count++;
            }
        }
        System.out.println("Loaded " + samples.size() + " samples from CSV");
        return samples;
    }

    /**
     * Creates Words from original MNIST data.
     * Each Word has 28 symbols, each symbol represents a row of 28 pixels.
     */
    public Word<PSymbolInstance>[] create_batch_word(List<int[]> samples) {
        List<Word<PSymbolInstance>> batchList = new ArrayList<>();
        for (int[] pixels : samples) {
            PSymbolInstance[] symbols = new PSymbolInstance[28];
            for (int row = 0; row < 28; row++) {
                // Extract 28 pixels for this row
                int[] rowPixels = new int[28];
                for (int col = 0; col < 28; col++) {
                    rowPixels[col] = pixels[row * 28 + col];
                }
                // Create MNISTDataValue with original pixel values (0-255)
                symbols[row] = traderSUL.createMNISTRow(rowPixels);
            }
            batchList.add(Word.fromSymbols(symbols));
        }
        return batchList.toArray(new Word[0]);
    }

    // --- Remaining Helper Methods (parseToJson, info, etc.) as in your original snippet ---
    public static String parseToJson(String inputBlock) throws Exception {
        // Implementation remains same as your original provided code for DOT/JSON conversion
        // ... (truncated for brevity, keep your original implementation here)
        return "{}"; // Placeholder
    }

    @Override
    public String description() { return "Extracts RA from MNIST Binary Logic"; }

    @Override
    public String help() { return "Ensure mnist_test.csv is in the working directory."; }
}