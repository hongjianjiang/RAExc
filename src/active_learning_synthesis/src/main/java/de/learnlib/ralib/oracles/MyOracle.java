package de.learnlib.ralib.oracles;

import de.learnlib.ralib.words.PSymbolInstance;
import net.automatalib.word.Word;

public interface MyOracle extends DataWordOracle {
    boolean accepts(Word<PSymbolInstance> trace);
}
