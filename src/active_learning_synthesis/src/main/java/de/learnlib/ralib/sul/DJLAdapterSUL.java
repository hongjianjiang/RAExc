package de.learnlib.ralib.sul;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import de.learnlib.ralib.data.DataType;
import de.learnlib.ralib.data.DataValue;
import de.learnlib.ralib.oracles.LanguageOracle;
import de.learnlib.ralib.tools.classanalyzer.SpecialSymbols;
import de.learnlib.ralib.words.InputSymbol;
import de.learnlib.ralib.words.PSymbolInstance;
import de.learnlib.ralib.words.ParameterizedSymbol;
import net.automatalib.word.WordBuilder;

/**
 * An adapter that wraps a native DJL-based RNNOracle so it can be 
 * stepped through symbol-by-symbol by RALib's learning algorithms 
 * and Random Walk equivalence testers.
 */
public class DJLAdapterSUL extends DataWordSUL {
    private final LanguageOracle oracle;
    private final ParameterizedSymbol outSym;
    private WordBuilder<PSymbolInstance> currentPrefix;
    public static final DataType DOUBLE_TYPE = new DataType("double");
    public static final DataType INTEGER_TYPE = new DataType("int");
    private static final Random random = new Random();
    private DataType inputType = DOUBLE_TYPE;
    private String inputSymbolName = "price";

    /**
     * Set the input data type for random walk ("double" or "int").
     */
    public void setInputType(DataType type) {
        this.inputType = type;
    }

    /**
     * Set the input symbol name for random walk.
     */
    public void setInputSymbolName(String name) {
        this.inputSymbolName = name;
    }

    /**
     * Generate a random walk trace of up to maxLen steps.
     * Each step is a random double or int in a reasonable range.
     * Returns the sequence as a List<Double> (or Integer, but always as Double for compatibility).
     */
    public List<Double> randomWalk(int maxLen) {
        pre();
        List<Double> trace = new ArrayList<>();
        int len = 1 + random.nextInt(maxLen); // random length between 1 and maxLen
        for (int i = 0; i < len; i++) {
            double value;
            if (inputType.equals(DOUBLE_TYPE)) {
                value = -100 + 200 * random.nextDouble(); // random double in [-100, 100]
            } else {
                value = random.nextInt(201) - 100; // random int in [-100, 100]
            }
            // Create a PSymbolInstance for the input
            PSymbolInstance input = new PSymbolInstance(
                new InputSymbol(inputSymbolName, inputType),
                new DataValue(inputType, BigDecimal.valueOf(value))
            );
            step(input);
            trace.add(value);
        }
        post();
        return trace;
    }

    public DJLAdapterSUL(LanguageOracle oracle) {
        this.oracle = oracle;
        // The output is a simple boolean (Accept/Reject)
        // Ensure SpecialSymbols.BOOLEAN_TYPE is accessible, or use DataType.BOOLEAN
        this.outSym = new InputSymbol("out", SpecialSymbols.BOOLEAN_TYPE);
    }

    @Override
    public void pre() {
        countResets(1);
        // Reset the sequence buffer for a new trace
        currentPrefix = new WordBuilder<>(); 
    }

    @Override
    public PSymbolInstance step(PSymbolInstance in) {
        countInputs(1);
        
        // 1. Add the new input symbol to our running sequence
        currentPrefix.add(in);
        
        // 2. Ask the native DJL Oracle if it accepts this sequence so far
        boolean isAccepted = oracle.accepts(currentPrefix.toWord());
        
        // 3. Wrap the boolean answer into a RALib output symbol
        DataValue outData = new DataValue(SpecialSymbols.BOOLEAN_TYPE, isAccepted ? BigDecimal.ONE : BigDecimal.ZERO);
        return new PSymbolInstance(outSym, outData);
    }

    @Override
    public void post() {
        // Nothing needs to be explicitly cleaned up between traces,
        // but this is where you would clear native memory if needed.
        currentPrefix = null; 
    }
}