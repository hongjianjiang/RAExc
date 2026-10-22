package de.learnlib.ralib.sul;

import de.learnlib.ralib.data.DataValue;
import de.learnlib.ralib.words.PSymbolInstance;
import net.automatalib.word.Word;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;

public class TraceCache {
    public static HashMap<Word<PSymbolInstance>, Boolean> cache = new HashMap<>();
    static int maxTraceLen = 0;
    static double minPrice = Double.MAX_VALUE;
    static double maxPrice = Double.MIN_VALUE;

    public static Boolean get(Word<PSymbolInstance> prices) {
        return cache.get(prices);
    }

    public static void put(Word<PSymbolInstance> prices, Boolean accepted) {
        int prefixLen = prices.length();
        maxTraceLen = Math.max(maxTraceLen, prefixLen);
        cache.put(prices, accepted);
    }

    public static Map<Word<PSymbolInstance>, Boolean> getTracesOfLength(int n) {
        Map<Word<PSymbolInstance>, Boolean> result = new HashMap<>();
        for (Map.Entry<Word<PSymbolInstance>, Boolean> entry : cache.entrySet()) {
            if (entry.getKey().length() == n) {
                result.put(entry.getKey(), entry.getValue());
            }
        }
        return result;
    }


    public static String string(boolean fixTraceLen) {
        StringBuilder sb = new StringBuilder();
        StringBuilder sb1 = new StringBuilder();
        boolean isFirstLine = true;
        for (Map.Entry<Word<PSymbolInstance>, Boolean> entry : TraceCache.cache.entrySet()) {
            Word<PSymbolInstance> prices = entry.getKey();
            int numPrices = prices.length();
            if (numPrices == 0) {
                continue; // don't write epsilon
            }
            if (isFirstLine) {
                isFirstLine = false;
            } else {
                sb.append("\n");
                sb1.append("\n");
            }
            PSymbolInstance symbol = prices.getSymbol(0);
            BigDecimal price = TraderSUL.getPrice(symbol); //+ diff;
            sb.append(price).append(',');
            for (int i = 1; i < numPrices; i++) {
                symbol = prices.getSymbol(i);
                PSymbolInstance lastSymbol = prices.getSymbol(i - 1);
                price = TraderSUL.getPrice(symbol); //+ diff;
                BigDecimal lastPrice = TraderSUL.getPrice(lastSymbol); //+ diff;
                sb.append(price).append(',');
                if (price.compareTo(lastPrice) > 0) {
                    sb1.append(">,");
                } else if (price.compareTo(lastPrice) < 0) {
                    sb1.append("<,");
                } else {
                    sb1.append("=,");
                }
            }
            boolean accepted = entry.getValue();
            sb.append(accepted ? 1 : 0);
        }
//        System.out.println(sb.toString());
//        System.out.println(sb1.toString());
        return sb.toString();
    }

    static double getPrice(PSymbolInstance input) {
        DataValue dv = input.getParameterValues()[0];
        Object price = dv.getValue();
        return ((BigDecimal) price).doubleValue();
    }
}
