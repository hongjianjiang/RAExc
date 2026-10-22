package de.learnlib.ralib.tools;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
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

import static org.unbescape.json.JsonEscape.escapeJson;

import de.learnlib.query.DefaultQuery;
import de.learnlib.ralib.data.Constants;
import de.learnlib.ralib.data.DataType;
import de.learnlib.ralib.learning.Hypothesis;
import de.learnlib.ralib.learning.RaLearningAlgorithm;
import de.learnlib.ralib.learning.ralambda.RaDT;
import de.learnlib.ralib.learning.ralambda.RaLambda;
import de.learnlib.ralib.learning.rastar.RaStar;
import de.learnlib.ralib.oracles.DataWordOracle;
import de.learnlib.ralib.oracles.LanguageOracle;
import de.learnlib.ralib.oracles.MyOracle;
import de.learnlib.ralib.oracles.RNNOracle;
import de.learnlib.ralib.oracles.SimulatorOracle;
import de.learnlib.ralib.oracles.TransformerOracle;
import de.learnlib.ralib.oracles.TreeOracleFactory;
import de.learnlib.ralib.oracles.mto.MultiTheorySDTLogicOracle;
import de.learnlib.ralib.oracles.mto.MultiTheoryTreeOracle;
import de.learnlib.ralib.smt.ConstraintSolver;
import de.learnlib.ralib.sul.DJLAdapterSUL;
import de.learnlib.ralib.sul.TraceCache;
import de.learnlib.ralib.sul.TraderSUL;
import static de.learnlib.ralib.sul.TraderSUL.DOUBLE_TYPE;
import static de.learnlib.ralib.sul.TraderSUL.INTEGER_TYPE;
import de.learnlib.ralib.theory.Theory;
import de.learnlib.ralib.tools.config.Configuration;
import de.learnlib.ralib.tools.config.ConfigurationException;
import de.learnlib.ralib.tools.config.ConfigurationOption;
import de.learnlib.ralib.tools.theories.DoubleInequalityTheory;
import de.learnlib.ralib.tools.theories.IntegerEqualityTheory;
import de.learnlib.ralib.tools.theories.IntegerEqualityTheory1;
import de.learnlib.ralib.words.PSymbolInstance;
import de.learnlib.ralib.words.ParameterizedSymbol;
import net.automatalib.word.Word;


public class RNNSimulator extends AbstractToolWithRandomWalk {
    private static final Random random = new Random();


    protected static final ConfigurationOption.DoubleOption OPTION_P = new ConfigurationOption.DoubleOption(
            "p",
            "Probability parameter p",
            0.01d,
            false
    );


    protected static final ConfigurationOption.IntegerOption OPTION_TIMEOUT = new ConfigurationOption.IntegerOption(
            "timeout",
            "Timeout in seconds for the run() method",
            120,
            false
    );



    protected static final ConfigurationOption.StringOption OPTION_LANGUAGE = new ConfigurationOption.StringOption(
            "language",
            "data language (e.g. \'L1\')",
            "L1",
            false
    );


    // randomness & trace‐length
    protected static final ConfigurationOption.IntegerOption     OPTION_SEED                = new ConfigurationOption.IntegerOption    (
            "seed",
            "PRNG seed",
            42,
            false
    );


    // price‐range parameters
    protected static final ConfigurationOption.IntegerOption     OPTION_MAX_PRICE           = new ConfigurationOption.IntegerOption    (
            "maxPrice",
            "Maximum allowed price",
            12,
            false
    );

    // max len
    protected static final ConfigurationOption.IntegerOption     OPTION_MAX_LENGTH           = new ConfigurationOption.IntegerOption    (
            "maxLen",
            "Maximum Length of input",
            50,
            false
    );


    protected static final ConfigurationOption.IntegerOption     OPTION_MIN_PRICE           = new ConfigurationOption.IntegerOption    (
            "minPrice",
            "Minimum allowed price",
            -12,
            false
    );

    // numeric transforms
    protected static final ConfigurationOption.DoubleOption   OPTION_SHIFT               = new ConfigurationOption.DoubleOption  (
            "shift",
            "Additive shift applied to prices",
            0.0d,
            false
    );

    protected static final ConfigurationOption.DoubleOption   OPTION_DELTA               = new ConfigurationOption.DoubleOption  (
            "delta",
            "Correct rate",
            0.0d,
            false
    );

    protected static final ConfigurationOption.DoubleOption   OPTION_EPSILON               = new ConfigurationOption.DoubleOption  (
            "epsilon",
            "Error rate",
            0.0d,
            false
    );

    protected static final ConfigurationOption.IntegerOption     OPTION_SCALING             = new ConfigurationOption.IntegerOption    (
            "scaling",
            "Multiplicative scaling factor for prices",
            1,
            false
    );

    protected static final ConfigurationOption.BooleanOption OPTION_FIX_TRACE_LEN       = new ConfigurationOption.BooleanOption(
            "fixTraceLen",
            "Whether to enforce a fixed trace length",
            true,
            false
    );

    // logging & output
    protected static final ConfigurationOption.BooleanOption OPTION_SAVE_QUERY_LOG      = new ConfigurationOption.BooleanOption(
            "saveQueryLog",
            "Log all queries to file",
            false,
            false
    );
    protected static final ConfigurationOption.BooleanOption OPTION_SAVE_RNN_QUERY_LOG  = new ConfigurationOption.BooleanOption(
            "saveRNNQueryLog",
            "Log RNN‐generated queries to file",
            false,
            false
    );
    protected static final ConfigurationOption.StringOption  OPTION_DATATYPE            = new ConfigurationOption.StringOption (
            "datatype",
            "Primitive data type for prices (e.g. \"double\", \"int\")",
            "double",
            false
    );
    protected static final ConfigurationOption.StringOption  OPTION_RNNTYPE           = new ConfigurationOption.StringOption (
            "rnn",
            "Primitive data type for prices (e.g. \"lstm\", \"transformer\")",
            "lstm",
            false
    );
    protected static final ConfigurationOption.StringOption  OPTION_QUERY_FILE_NAME     = new ConfigurationOption.StringOption (
            "queryFileName",
            "Filename for saving query log",
            "queries.log",
            false
    );
    protected static final ConfigurationOption.StringOption  OPTION_QUERY_RNN_FILE_NAME = new ConfigurationOption.StringOption (
            "queryRNNFileName",
            "Filename for saving RNN query log",
            "queriesRNN.log",
            false
    );
    protected static final ConfigurationOption.StringOption  OPTION_LOG_FILE_NAME       = new ConfigurationOption.StringOption (
            "logFileName",
            "General log output filename (null = no file)",
            null,
            false
    );
    protected static final ConfigurationOption.StringOption  OPTION_DOT_FILE_NAME       = new ConfigurationOption.StringOption (
            "dotFileName",
            "Output DOT (graphviz) filename (null = no file)",
            null,
            false
    );

    protected static final ConfigurationOption.StringOption  OPTION_MODEL_FILE_NAME     = new ConfigurationOption.StringOption (
            "modelFileName",
            "Serialized model filename (null = no file)",
            null,
            false
    );
    private Hypothesis hyp = null;
    private RaLearningAlgorithm rastar;
    private TreeOracleFactory hypFactory;
    private boolean fixTraceLen;
    private String rnntype;
    private MultiTheoryTreeOracle mto;
    private MyOracle traceOracle;
    private MultiTheorySDTLogicOracle mlo;
    private ConstraintSolver solver;
    private double epsilon;
    private double delta;
    private int maxLen;
    private List<Integer> labelSet;
                // ...existing code...
    private long startTime;
    private long endTime;
    private String queryRNNFileName;
    private TraderSUL traderSUL;
    private double p;
    private String language = "L1";
    private int timeoutSeconds = 120; // default
    private boolean cexFromRobust = false;
    private Word<PSymbolInstance> adv;
    private DJLAdapterSUL djlSUL;
    @Override
    public String description() {
        return "analyzes RNN models";
    }

    @Override
    public void setup(Configuration config) throws ConfigurationException {
        super.setup(config);
        p = OPTION_P.parse(config);
        fixTraceLen = OPTION_FIX_TRACE_LEN.parse(config);
        rnntype = OPTION_RNNTYPE.parse(config);
        delta = OPTION_DELTA.parse(config);
        epsilon = OPTION_EPSILON.parse(config);
        String datatype = OPTION_DATATYPE.parse(config);
        String teacher = OPTION_TEACHERS.parse(config);
        timeoutSeconds = OPTION_TIMEOUT.parse(config);
        maxLen = OPTION_MAX_LENGTH.parse(config);
        final Map<DataType, Theory> teachers = new LinkedHashMap<>();
        language = OPTION_LANGUAGE.parse(config);
        String advStr = config.getProperty("adv"); // e.g. "1,2,3,4,5"
        startTime = System.nanoTime();


        if (teacher.contains("Inequality")){
            teachers.put(DOUBLE_TYPE, new DoubleInequalityTheory(DOUBLE_TYPE));
        }
        else{
            teachers.put(INTEGER_TYPE, new IntegerEqualityTheory(INTEGER_TYPE));
        }
        // Only initialize djlSUL if traceOracle is an RNNOracle
        if (traceOracle instanceof RNNOracle) {
            if (datatype.equals("double")) {
                djlSUL.setInputType(DJLAdapterSUL.DOUBLE_TYPE);
            } else if (datatype.equals("int") || datatype.equals("integer")) {
                djlSUL.setInputType(DJLAdapterSUL.INTEGER_TYPE);
            } else {
                djlSUL.setInputType(DJLAdapterSUL.DOUBLE_TYPE);
            }
            djlSUL.setInputSymbolName("price");
        }
        if (datatype.equals("double")) {
            traderSUL = new TraderSUL(DOUBLE_TYPE);
        } else if (datatype.equals("int") || datatype.equals("integer")) {
            traderSUL = new TraderSUL(INTEGER_TYPE);
        } else {
            traderSUL = new TraderSUL(DOUBLE_TYPE);
        }
        if (advStr != null && !advStr.isEmpty()) {
            String[] tokens = advStr.split(",");
            PSymbolInstance[] symbols = new PSymbolInstance[tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                double value = Double.parseDouble(tokens[i].trim());
                symbols[i] = traderSUL.createPrice(value);
            }
            cexFromRobust = true;
            adv = Word.fromSymbols(symbols);
        }


        queryRNNFileName = OPTION_QUERY_RNN_FILE_NAME.parse(config);

//        config.setProperty("metaFilePath", "/Users/alexander/work/interpretera/dra_extraction_rnn/models/highschool_2011/meta");
        Constants consts = new Constants();
//        SymbolicDataValueGenerator.ConstantGenerator cgen = new SymbolicDataValueGenerator.ConstantGenerator();

//        int[] int_consts = {0};
//        for (int i = 0; i < int_consts.length; i++) {
//            SymbolicDataValue.Constant c = cgen.next(INTEGER_TYPE);
//            DataValue dv = new DataValue(INTEGER_TYPE, new BigDecimal(int_consts[i]));
//            consts.put(c, dv);
//        }
//        System.out.println(consts);

        if (Objects.equals(rnntype, "transformer")) {
            traceOracle = new TransformerOracle(config);
        } else if (Objects.equals(rnntype, "lstm")) {
            traceOracle = new RNNOracle(config);
        } else {
            traceOracle = new LanguageOracle(config);
        }

        solver = new ConstraintSolver();
        mlo = new MultiTheorySDTLogicOracle(consts, solver);
        mto = new MultiTheoryTreeOracle(
                traceOracle, teachers, consts, solver);

        hypFactory = hyp -> {
            DataWordOracle hypOracle = new SimulatorOracle(hyp);
            return new MultiTheoryTreeOracle(hypOracle, teachers, consts, solver);
        };

        ParameterizedSymbol priceInput = traderSUL.createPrice(1).getBaseSymbol();
        ParameterizedSymbol[] inputs = new ParameterizedSymbol[] { priceInput };

        switch (this.learner) {
            case AbstractToolWithRandomWalk.LEARNER_SLSTAR:
                this.rastar = new RaStar(mto, hypFactory, mlo, consts, false, inputs);
                break;
            case AbstractToolWithRandomWalk.LEARNER_SLLAMBDA:
                this.rastar = new RaLambda(mto, hypFactory, mlo, consts, false, inputs);
                ((RaLambda)this.rastar).setSolver(solver);
                break;
            case AbstractToolWithRandomWalk.LEARNER_RADT:
                this.rastar = new RaDT(mto, hypFactory, mlo, consts, false, inputs);
                break;
            default:
                throw new ConfigurationException("Unknown Learning algorithm: " + this.learner);
        }



    }
    @Override
    public String help() {
        return "";
    }

    @Override
    public void run() throws RaLibToolException {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<?> future = executor.submit(() -> {
            Hypothesis hyp;
            int rounds = 0;
            
            // Statistical parameters for PAC-style learning
            double numerator = Math.log(2.0 / epsilon);
            double denominator = 2 * delta * delta;
            double num = (int) Math.ceil(numerator / denominator);
            
            System.out.println("=============================== START ===============================");
            System.out.println("Target Language: " + language);
            System.out.println("Required Test Samples (PAC): " + (int)num);

            while (rounds < 100) {
                rounds++;
                System.out.println("\n--- Round " + rounds + " ---");
                
                // 1. Learning Phase
                System.out.println("Learning hypothesis...");
                rastar.learn();
                hyp = rastar.getHypothesis();
                
                // 2. Export / Logging Phase
                try {
                    String xmlOutput = parseToJson(hyp.toString())
                        .replace("p1", "curr")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("'", "")
                        .replace("true", "True");
                    
                    File dir = new File("active_models");
                    if (!dir.exists()) dir.mkdirs();

                    File outFile = new File(dir, language + ".xml");
                    try (FileWriter writer = new FileWriter(outFile)) {
                        writer.write(xmlOutput);
                    }
                    
                    String queries = TraceCache.string(fixTraceLen);
                    Files.write(Paths.get(queryRNNFileName), queries.getBytes(StandardCharsets.US_ASCII));
                } catch (Exception e) {
                    System.err.println("Warning: Could not save intermediate model: " + e.getMessage());
                }

                // 3. Equivalence Testing Phase (Statistical Batch)
                System.out.println("Generating test samples for equivalence check...");
                List<String> tmp;
                try {
                    tmp = gen_test_samples(language, (int) (num * 0.5), (int) (num * 0.5), maxLen);
                    tmp.sort(Comparator.comparingInt(String::length));
                } catch (IOException | InterruptedException e) {
                    throw new RuntimeException("Python script failed to generate samples", e);
                }

                Word<PSymbolInstance>[] batch = create_batch_word(tmp);
                List<Boolean> rnnResults = is_words_in_batch(batch);
                
                int equal_num = 0;
                List<Integer> cexIndices = new ArrayList<>();

                for (int i = 0; i < batch.length; i++) {
                    boolean hypAccepts = hyp.accepts(batch[i]);
                    if (Objects.equals(rnnResults.get(i), hypAccepts)) {
                        equal_num++;
                    } else {
                        cexIndices.add(i);
                    }
                }

                double equal_rate = (double) equal_num / batch.length;
                System.out.println("Accuracy (Hypothesis vs RNN): " + String.format("%.2f%%", equal_rate * 100));

                // 4. Refinement Loop
                boolean ceFoundInRound = false;
                if (equal_rate < (1 - epsilon) && !cexIndices.isEmpty()) {
                    System.out.println("Inconsistency detected above epsilon threshold. Refining...");
                    
                    for (int index : cexIndices) {
                        Word<PSymbolInstance> cexWord = batch[index];
                        boolean rnnLabel = rnnResults.get(index);

                        // Re-check against updated hypothesis (as the structure may change during this loop)
                        if (!Objects.equals(hyp.accepts(cexWord), rnnLabel)) {
                            System.out.println("Adding Counterexample: " + cexWord + " (RNN: " + rnnLabel + ")");
                            
                            DefaultQuery<PSymbolInstance, Boolean> cex = new DefaultQuery<>(cexWord, rnnLabel);
                            rastar.addCounterexample(cex);
                            
                            // Incrementally learn from this specific CEX
                            rastar.learn();
                            hyp = rastar.getHypothesis();
                            System.out.println("Refined Hypothesis after adding CEX: " + hyp);
                            ceFoundInRound = true;
                            break;
                        }
                    }
                }

                // If no NEW counterexamples were found in the batch (or accuracy is high enough), stop.
                if (!ceFoundInRound) {
                    System.out.println("No further counterexamples found. Convergence reached.");
                    break;
                }
            }

            endTime = System.nanoTime();
            long elapsedTime = endTime - startTime;
            hyp = rastar.getHypothesis();
            
            System.out.println("\n=============================== STOP ===============================");
            System.out.println("Total Time: " + (int) (elapsedTime / 1e9) + " seconds.");
            System.out.println("Final Statistics:");
            System.out.println("- States: " + hyp.getStates().size());
            System.out.println("- Registers: " + hyp.getRegisters().size());
            System.out.println("- Transitions: " + hyp.getTransitions().size());

            if (cexFromRobust && adv != null) {
                System.out.println("Adversarial/Robustness check (Adv Word): " + hyp.accepts(adv));
            }
            System.out.println("====================================================================");
        });

        try {
            future.get(timeoutSeconds, TimeUnit.SECONDS);
        } catch (TimeoutException e) {
            future.cancel(true);
            System.err.println("TIMEOUT: run() exceeded " + timeoutSeconds + " seconds.");
        } catch (ExecutionException e) {
            throw new RaLibToolException("Execution error: " + e.getCause());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RaLibToolException("Thread interrupted: " + e);
        }
    }

    public Hypothesis getHypothesis() {
        return hyp;
    }

    public static String parseToJson(String inputBlock) throws Exception {
        // 1) Split into lines
        String[] lines = inputBlock.split("\\r?\\n");

        // regex for a “state header,” e.g.  “l0 (+) (initial):”
        Pattern stateHeader = Pattern.compile("^([^\\s]+)\\s+\\((\\+|\\-)\\)(?:\\s+\\(initial\\))?:\\s*$");
        // regex for a “transition line,” e.g. “(l0 (+), ?price[INTEGER], true, [r1:=p1,], l1 (+))”
        Pattern transitionLine = Pattern.compile(
                "^\\s*\\(\\s*([^\\s]+)\\s+\\((\\+|\\-)\\)\\s*,\\s*\\?([^,]+)\\s*,\\s*([^,]+)\\s*,\\s*\\[([^\\]]*)\\]\\s*,\\s*([^\\s]+)\\s+\\((\\+|\\-)\\)\\s*\\)\\s*$"
        );

        // 1A) First pass: build a LinkedHashMap of stateName -> index (in encounter order)
        LinkedHashMap<String, Integer> stateIndexMap = new LinkedHashMap<>();
        // Also keep track of which states are marked accepting (“+”)
        Map<String, Boolean> isAccepting = new HashMap<>();

        int nextStateIndex = 0;
        for (String raw : lines) {
            String line = raw.trim();
            if (line.isEmpty()) continue;
            Matcher sh = stateHeader.matcher(line);
            if (sh.matches()) {
                String stName = sh.group(1);       // e.g. "l0"
                String marker = sh.group(2);       // "+" or "-"
                if (!stateIndexMap.containsKey(stName)) {
                    stateIndexMap.put(stName, nextStateIndex++);
                    isAccepting.put(stName, marker.equals("+"));
                }
            }
        }

        // How many distinct states did we see?
        int numStates = stateIndexMap.size();

        // 1B) Prepare to collect transitions, register‐vars, final‐state indices
        class Transition {
            int startState;
            int endState;
            String currVar;             // e.g. "price"
            String guard;               // e.g. "('r1' != 'p1')"
            Map<String,String> assign;  // e.g. { "r1" -> "p1" }
        }
        List<Transition> allTransitions = new ArrayList<>();
        Set<String> registerVars = new LinkedHashSet<>();
        Set<Integer> finalStates = new TreeSet<>();

        // If ANY state is “accepting,” add its index to finalStates
        for (Map.Entry<String,Integer> e : stateIndexMap.entrySet()) {
            if (isAccepting.getOrDefault(e.getKey(), false)) {
                finalStates.add(e.getValue());
            }
        }

        // We also want to know the name of the “input” variable (without “[…]”).
        // We’ll take the one from the very first transition line we see.
        String globalInputVar = null;

        // 2) Second pass: parse every transition line
        for (String raw : lines) {
            Matcher tm = transitionLine.matcher(raw);
            if (!tm.matches()) continue;

            String srcName     = tm.group(1).trim();   // e.g. "l0"
            // tm.group(2) is source marker “+”/“-” (ignored here)
            String rawInput    = tm.group(3).trim();   // e.g. "price[INTEGER]"
            String guardExpr   = tm.group(4).trim();   // e.g. "true" or "('r1' != 'p1')"
            String asgInside   = tm.group(5).trim();   // e.g. "r1:=p1,"  or ""
            String dstName     = tm.group(6).trim();   // e.g. "l1"
            // tm.group(7) is destination marker (“+”/“-”), we only use it to know if dst is accepting,
            // but finalStates was already populated from the header pass.

            // figure out indices
            if (!stateIndexMap.containsKey(srcName) || !stateIndexMap.containsKey(dstName)) {
                throw new Exception("Found transition with unknown state: " + srcName + " or " + dstName);
            }
            int srcIdx = stateIndexMap.get(srcName);
            int dstIdx = stateIndexMap.get(dstName);

            // extract “currVar” from rawInput by discarding anything from “[” onward
            // e.g. "price[INTEGER]" → "price"
            String currVar = "p1";

            // set globalInputVar if it’s still null
            if (globalInputVar == null) {
                globalInputVar = currVar;
            }

            // parse assignments string “r1:=p1,r2:=x,…” (possibly empty)
            Map<String,String> assignMap = new LinkedHashMap<>();
            if (!asgInside.isEmpty()) {
                String[] pieces = asgInside.split(",");
                for (String piece : pieces) {
                    piece = piece.trim();
                    if (piece.isEmpty()) continue;
                    int colonEq = piece.indexOf(":=");
                    if (colonEq < 0) {
                        throw new Exception("Malformed assignment: " + piece);
                    }
                    String regName = piece.substring(0, colonEq).trim();
                    String expr    = piece.substring(colonEq + 2).trim();
                    if (!regName.isEmpty() && !expr.isEmpty()) {
                        assignMap.put(regName, expr);
                        registerVars.add(regName);
                    }
                }
            }

            // store this transition
            Transition t = new Transition();
            t.startState = srcIdx;
            t.endState   = dstIdx;
            t.currVar    = currVar;
            t.guard      = guardExpr;
            t.assign     = assignMap;
            allTransitions.add(t);
        }

        java.util.function.Function<String,String> escapeJson = (String s) -> {
            if (s == null) return "";
            StringBuilder sb = new StringBuilder();
            for (char c : s.toCharArray()) {
                switch (c) {
                    case '\\':
                        sb.append("\\\\");
                        break;
                    case '\"':
                        sb.append("\\\"");
                        break;
                    case '\b':
                        sb.append("\\b");
                        break;
                    case '\f':
                        sb.append("\\f");
                        break;
                    case '\n':
                        sb.append("\\n");
                        break;
                    case '\r':
                        sb.append("\\r");
                        break;
                    case '\t':
                        sb.append("\\t");
                        break;
                    default:
                        if (c < 0x20) {
                            sb.append(String.format("\\u%04x", (int)c));
                        } else {
                            sb.append(c);
                        }
                }
            }
            return sb.toString();
        };

        StringBuilder j = new StringBuilder();
        j.append("{\n");

        // "states": <numStates>,
        j.append("  \"states\": ").append(numStates).append(",\n");

        // "transitions": [ … ],
        j.append("  \"transitions\": [\n");
        for (int i = 0; i < allTransitions.size(); i++) {
            Transition T = allTransitions.get(i);
            j.append("    {\n");
            //   "curr_var": "price",
            j.append("      \"curr_var\": \"").append(escapeJson(T.currVar)).append("\",\n");
            //   "start_state": 0,
            j.append("      \"start_state\": ").append(T.startState).append(",\n");
            //   "guard": "<escaped guard>",
            j.append("      \"guard\": \"").append(escapeJson(T.guard)).append("\",\n");
            //   "assignment": { "r1": "p1", … },
            j.append("      \"assignment\": {");
            // if no assignments, leave it empty; otherwise separate with commas
            if (!T.assign.isEmpty()) {
                j.append("\n");
                int cnt = 0;
                for (Map.Entry<String,String> e : T.assign.entrySet()) {
                    String rvar = e.getKey();
                    String expr = e.getValue();
                    j.append("        \"")
                            .append(escapeJson(rvar))
                            .append("\": \"")
                            .append(escapeJson(expr))
                            .append("\"");
                    cnt++;
                    if (cnt < T.assign.size()) {
                        j.append(",");
                    }
                    j.append("\n");
                }
                j.append("      ");
            }
            j.append("},\n");
            //   "end_state": 1
            j.append("      \"end_state\": ").append(T.endState).append("\n");
            j.append("    }");
            if (i < allTransitions.size() - 1) j.append(",");
            j.append("\n");
        }
        j.append("  ],\n");

        // "register_vars": [ "r1", "r2", … ],
        j.append("  \"register_vars\": [");
        if (!registerVars.isEmpty()) {
            j.append("\n");
            int cntRV = 0;
            for (String rv : registerVars) {
                j.append("    \"").append(escapeJson(rv)).append("\"");
                cntRV++;
                if (cntRV < registerVars.size()) j.append(",");
                j.append("\n");
            }
            j.append("  ");
        }
        j.append("],\n");

        // "final_states": [ 0, 2, … ]
        j.append("  \"final_states\": [");
        if (!finalStates.isEmpty()) {
            j.append("\n");
            int cntFS = 0;
            for (Integer idx : finalStates) {
                j.append("    ").append(idx);
                cntFS++;
                if (cntFS < finalStates.size()) j.append(",");
                j.append("\n");
            }
            j.append("  ");
        }
        j.append("]\n");

        j.append("}");
        return j.toString();
    }

    public ArrayList<Boolean> is_words_in_batch(Word<PSymbolInstance>[] batch) {
        ArrayList<Boolean> results = new ArrayList<>();
        for (Word<PSymbolInstance> word : batch) {
            boolean result = traceOracle.accepts(word);
            results.add(result);
        }
        return results;
    }

    public Word<PSymbolInstance>[] create_batch_word(List<String> lines) {
        List<Word<PSymbolInstance>> batchList = new ArrayList<>();
        for (String line : lines) {
            String[] tokens = line.split(",");
            PSymbolInstance[] symbols = new PSymbolInstance[tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                double value = Double.parseDouble(tokens[i].trim());
                symbols[i] = traderSUL.createPrice(value);
            }
            batchList.add(Word.fromSymbols(symbols));
        }
        batchList.sort(Comparator.comparingInt(Word::length));
        @SuppressWarnings("unchecked")
        Word<PSymbolInstance>[] batch = batchList.toArray(new Word[0]);
        return batch;
    }






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

    // Collects lines from the Python script and returns them as a list
    public static List<String> gen_test_samples(String language, int numPos, int numNeg, int maxLen) throws IOException, InterruptedException {
        ProcessBuilder pb = new ProcessBuilder(
                "python3", "src/benchmark_gen.py", language, "--num-pos", String.valueOf(numPos), "--num-neg", String.valueOf(numNeg), "--max-len", String.valueOf(maxLen)
        );
        Process process = pb.start();
        List<String> lines = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                lines.add(line);
            }
        }
        process.waitFor();
        return lines;
    }
}


