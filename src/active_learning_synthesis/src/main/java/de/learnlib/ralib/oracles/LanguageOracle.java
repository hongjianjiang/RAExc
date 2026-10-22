package de.learnlib.ralib.oracles;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDManager;
import de.learnlib.query.Query;
import de.learnlib.ralib.data.DataValue;
import de.learnlib.ralib.sul.TraceCache;
import de.learnlib.ralib.tools.config.Configuration;
import de.learnlib.ralib.words.PSymbolInstance;
import net.automatalib.word.Word;

public class LanguageOracle implements MyOracle {

    private Model model;
    private Predictor<float[], Float> predictor;
    private NDManager manager;
    private Configuration config;
    private String language;

    public LanguageOracle(Configuration config) {
        this.config = config;
        this.language = this.config.getProperty("language");

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

        List<Double> sequence = Arrays.stream(rawSequence).boxed().toList();
        boolean accepted;

        switch (this.language) {
            case "L1":
                accepted = isL1Language(sequence);
                break;
            case "L2":
                accepted = isL2Language(sequence);
                break;
            case "L3":
                accepted = isL3Language(sequence);
                break;
            case "L4":
                accepted = isL4Language(sequence);
                break;
            case "L5":
                accepted = isL5Language(sequence);
                break;
            case "L6":
                accepted = isL6Language(sequence);
                break;
            case "L7":
                accepted = isL7Language(sequence);
                break;
            case "S1":
                accepted = isS1Language(sequence);
                break;
            case "S2":
                accepted = isS2Language(sequence);
                break;
            case "S3":
                accepted = isS3Language(sequence);
                break;
            case "S4":
                accepted = isS4Language(sequence);
                break;
            case "S5":
                accepted = is_S5Language(sequence);
                break;
            case "S6":
                accepted = is_S6Language(sequence);
                break;
            case "S7":
                accepted = upDownUp(sequence);
                break;
            case "S8":
                accepted = upDownUpDownPrefix(sequence);
                break;
            case "S9":
                accepted = isLStockLanguage(sequence);
                break;
            case "S10":
                accepted = isLStock1Language(sequence);
                break;
            case "S11":
                accepted = isLStock2Language(sequence);
                break;
            case "L2A":
                accepted = twoA(sequence);
                break;
            case "L3A":
                accepted = threeA(sequence);
                break;
            case "L4A":
                accepted = fourA(sequence);
                break;
            case "L5A":
                accepted = fiveA(sequence);
                break;
            case "L6A":
                accepted = sixA(sequence);
                break;
            case "L7A":
                accepted = sevenA(sequence);
                break;
            case "L8A":
                accepted = eightA(sequence);
                break;
            case "L9A":
                accepted = nineA(sequence);
                break;
            case "L10A":
                accepted = tenA(sequence);
                break;

            case "L20A":
                accepted = twentyA(sequence);
                break;
            case "L11A":
                accepted = elevenA(sequence);
                break;
            case "L2AB":
                accepted = twoAB(sequence);
                break;
            case "L3AB":
                accepted = threeAB(sequence);
//                accepted = false;
                break;
            case "L4AB":
                accepted = fourAB(sequence);
                break;
            case "L5AB":
                accepted = fiveAB(sequence);
                break;
            case "DistinctK":
                accepted = DistinctK(sequence);
                break;

//            case "LStock3":
//                accepted = isLStock3Language(sequence);
//                break;
            default:
                accepted = true; // or false, depending on your default policy
        }

        TraceCache.put(trace, accepted);
        return accepted;
    }

    public boolean isL1Language(List<Double> lst) {
        Set<Double> unique = new HashSet<>(lst);
        if (unique.size() == 1)
            return true;
        else
            return false;
    }


    public boolean isL2Language(List<Double> lst) {
        Set<Double> unique = new HashSet<>(lst);
        if (unique.size() == 2) {
            if (!lst.get(0).equals(lst.get(1))) {
                List<Double> pattern = Arrays.asList(lst.get(0), lst.get(1));
                List<Double> expected = new ArrayList<>();
                for (int i = 0; i < lst.size() / 2; i++) {
                    expected.addAll(pattern);
                }
                return lst.equals(expected);
            } else {
                return false;
            }
        } else if (lst.isEmpty()) {
            return true;
        } else {
            return false;
        }
    }

    public boolean DistinctK(List<Double> lst) {
        Set<Double> unique = new HashSet<>(lst);
        return unique.size() == 8 && lst.size() == 8;
    }


    public boolean isL3Language(List<Double> lst) {
        if (lst.size() == 0) {
            return false;
        }
        Set<Double> uniqueSet = new HashSet<>(lst);
        List<Double> unique = new ArrayList<>(uniqueSet);
        if (unique.size() == 1) {
            return lst.size() % 2 == 1;
        } else if (unique.size() != 2) {
            return false;
        }
        Double a = lst.get(0);
        Double b = unique.get(0).equals(a) ? unique.get(1) : unique.get(0);
        int countA = Collections.frequency(lst, a);
        int countB = Collections.frequency(lst, b);
        // One symbol occurs an odd and the other an even number of times
        // (matches L3.membership in src/benchmark_gen.py)
        return (countA % 2 == 1 && countB % 2 == 0)
                || (countB % 2 == 1 && countA % 2 == 0);
    }


    public boolean isL4Language(List<Double> sequence) {
        for (int i = 0; i < sequence.size() - 2; i++) {
            if (sequence.get(i).equals(sequence.get(i + 1)) && sequence.get(i + 1).equals(sequence.get(i + 2))) {
                return false;
            }
        }
        return true;
    }

    public boolean isL5Language(List<Double> sequence) {
        Set<Double> unique = new HashSet<>(sequence);
        if (unique.size() == 1) {
            return sequence.size() % 2 == 0;
        } else if (unique.size() > 2 || unique.size() == 0) {
            return false;
        }
        Iterator<Double> it = unique.iterator();
        Double a = it.next();
        Double b = it.next();
        int countA = Collections.frequency(sequence, a);
        int countB = Collections.frequency(sequence, b);
        return countA % 2 == 0 && countB % 2 == 0;
    }

    public boolean isL6Language(List<Double> seq) {
        if (seq.isEmpty()) {
            return true; // empty sequence: 0-0 = 0
        }
        Set<Double> unique = new HashSet<>(seq);
        if (unique.size() == 1) {
            return seq.size() % 3 == 0; // all same symbol
        } else if (unique.size() == 2) {
            Iterator<Double> it = unique.iterator();
            Double a = it.next();
            Double b = it.next();
            int countA = Collections.frequency(seq, a);
            int countB = Collections.frequency(seq, b);
            return Math.abs(countA - countB) % 3 == 0;
        } else {
            return false;
        }
    }

    public boolean isL7Language(List<Double> lst) {
        if (lst.isEmpty()) {
            return true; // empty sequence is valid
        }

        Double a = lst.get(0);
        Double b = null;
        int state = 0; // 0: first a*, 1: first b*, 2: second a*, 3: second b*

        for (Double x : lst) {
            if (x.equals(a)) {
                if (state == 0) {
                    // still in first a*
                } else if (state == 1) {
                    state = 2; // switch b* -> a*
                } else if (state == 2) {
                    // still in second a*
                } else {
                    // cannot go back to a after final b*
                    return false;
                }
            } else {
                if (b == null) {
                    b = x; // first time we see a different symbol
                }
                if (!x.equals(b)) {
                    return false; // third distinct value
                }
                if (state == 0) {
                    state = 1; // a* -> b*
                } else if (state == 1) {
                    // still in first b*
                } else if (state == 2) {
                    state = 3; // a* -> final b*
                } else {
                    // still in final b*
                }
            }
        }
        return true;
    }


    public boolean isS1Language(List<Double> arr) {
        for (int i = 1; i < arr.size(); i++) {
            if (arr.get(i) <= arr.get(i - 1)) {
                return false;
            }
        }
        return true;
    }

    public boolean isS2Language(List<Double> arr) {
        for (int i = 1; i < arr.size(); i++) {
            if (arr.get(i) >= arr.get(i - 1)) {
                return false;
            }
        }
        return true;
    }

    public boolean isS3Language(List<Double> arr) {
        if (arr.size() == 1) {
            return true;
        } else if (arr.size() > 1) {
            for (int i = 1; i < arr.size(); i++) {
                if (arr.get(i) < arr.get(i - 1)) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    public boolean isS4Language(List<Double> arr) {
        if (arr.size() == 1) {
            return true;
        } else if (arr.size() > 1) {
            for (int i = 1; i < arr.size(); i++) {
                if (arr.get(i) > arr.get(i - 1)) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }



    public boolean is_S5Language(List<Double> lst) {
        if (lst.size() < 3) {
            return false;
        }

        // Find the index of the maximum value
        int maxIdx = 0;
        double maxVal = lst.get(0);
        for (int i = 1; i < lst.size(); i++) {
            if (lst.get(i) > maxVal) {
                maxVal = lst.get(i);
                maxIdx = i;
            }
        }

        // Maximum must not be at the edges
        if (maxIdx == 0 || maxIdx == lst.size() - 1) {
            return false;
        }

        // Check strictly increasing before maximum
        for (int i = 1; i < maxIdx; i++) {
            if (lst.get(i) <= lst.get(i - 1)) {
                return false;
            }
        }

        // Check strictly decreasing after maximum
        for (int i = maxIdx + 1; i < lst.size(); i++) {
            if (lst.get(i) >= lst.get(i - 1)) {
                return false;
            }
        }

        return true;
    }

    public boolean is_S6Language(List<Double> lst) {
        if (lst.size() < 3) {
            return false;
        }

        // Find the index of the minimum value
        int minIdx = 0;
        double minVal = lst.get(0);
        for (int i = 1; i < lst.size(); i++) {
            if (lst.get(i) < minVal) {
                minVal = lst.get(i);
                minIdx = i;
            }
        }

        // Minimum must not be at the edges
        if (minIdx == 0 || minIdx == lst.size() - 1) {
            return false;
        }

        // Check strictly decreasing before minimum
        for (int i = 1; i < minIdx; i++) {
            if (lst.get(i) >= lst.get(i - 1)) {
                return false;
            }
        }

        // Check strictly increasing after minimum
        for (int i = minIdx + 1; i < lst.size(); i++) {
            if (lst.get(i) <= lst.get(i - 1)) {
                return false;
            }
        }

        return true;
    }


    public boolean isLStockLanguage(List<Double> lst) {
        if (lst.size() == 0) {
            return false;
        } else if (lst.size() == 1) {
            return lst.get(0) > 0;
        } else if (lst.size() == 2) {
            return lst.get(0) < lst.get(1);
        }
        List<Double> peaks = new ArrayList<>();
        List<Double> troughs = new ArrayList<>();

        for (int i = 1; i < lst.size() - 1; i++) {
            if (lst.get(i) > lst.get(i - 1) && lst.get(i) > lst.get(i + 1)) {
                peaks.add(lst.get(i));
            } else if (lst.get(i) < lst.get(i - 1) && lst.get(i) < lst.get(i + 1)) {
                troughs.add(lst.get(i));
            }
        }
        if (lst.get(lst.size() - 1) < lst.get(lst.size() - 2)) {
            if (!troughs.contains(lst.get(lst.size() - 1))) {
                troughs.add(lst.get(lst.size() - 1));
            }
        } else if (lst.get(lst.size() - 1) > lst.get(lst.size() - 2)) {
            if (!peaks.contains(lst.get(lst.size() - 1))) {
                peaks.add(lst.get(lst.size() - 1));
            }
        }
        // Check if peaks are in increasing order
        for (int i = 1; i < peaks.size(); i++) {
            if (peaks.get(i) <= peaks.get(i - 1)) {
                return false;
            }
        }
        // Check if troughs are in increasing order
        for (int i = 1; i < troughs.size(); i++) {
            if (troughs.get(i) <= troughs.get(i - 1)) {
                return false;
            }
        }
        return true;
    }

    public boolean isLStock1Language(List<Double> lst) {
        if (lst.size() == 0) {
            return false;
        }
        if (lst.size() == 1) {
            return lst.get(0) > 0;
        }
        if (lst.size() == 2) {
            return lst.get(0) < lst.get(1);
        }

        List<Double> peaks = new ArrayList<>();
        List<Double> troughs = new ArrayList<>();
        for (int i = 1; i < lst.size() - 1; i++) {
            if (lst.get(i) > lst.get(i - 1) && lst.get(i) > lst.get(i + 1)) {
                peaks.add(lst.get(i));
            } else if (lst.get(i) < lst.get(i - 1) && lst.get(i) < lst.get(i + 1)) {
                troughs.add(lst.get(i));
            }
        }
        // classify last endpoint
        if (lst.get(lst.size() - 1) < lst.get(lst.size() - 2)) {
            troughs.add(lst.get(lst.size() - 1));
        } else if (lst.get(lst.size() - 1) > lst.get(lst.size() - 2)) {
            peaks.add(lst.get(lst.size() - 1));
        }

        // peaks strictly increasing
        for (int i = 1; i < peaks.size(); i++) {
            if (peaks.get(i) <= peaks.get(i - 1)) {
                return false;
            }
        }
        // troughs strictly decreasing
        for (int i = 1; i < troughs.size(); i++) {
            if (troughs.get(i) >= troughs.get(i - 1)) {
                return false;
            }
        }
        return true;
    }

    public boolean isLStock2Language(List<Double> lst) {
        if (lst.size() == 0) {
            return false;
        }
        if (lst.size() == 1) {
            return true;
        }
        if (lst.size() == 2) {
            return lst.get(0) < lst.get(1);
        }

        List<Double> peaks = new ArrayList<>();
        List<Double> troughs = new ArrayList<>();
        for (int i = 1; i < lst.size() - 1; i++) {
            if (lst.get(i) > lst.get(i - 1) && lst.get(i) > lst.get(i + 1)) {
                peaks.add(lst.get(i));
            } else if (lst.get(i) < lst.get(i - 1) && lst.get(i) < lst.get(i + 1)) {
                troughs.add(lst.get(i));
            }
        }
        if (lst.get(lst.size() - 1) < lst.get(lst.size() - 2)) {
            troughs.add(lst.get(lst.size() - 1));
        } else if (lst.get(lst.size() - 1) > lst.get(lst.size() - 2)) {
            peaks.add(lst.get(lst.size() - 1));
        }

        // peaks strictly decreasing
        for (int i = 1; i < peaks.size(); i++) {
            if (peaks.get(i) >= peaks.get(i - 1)) {
                return false;
            }
        }
        // troughs strictly decreasing
        for (int i = 1; i < troughs.size(); i++) {
            if (troughs.get(i) >= troughs.get(i - 1)) {
                return false;
            }
        }
        return true;
    }

    public boolean isLStock3Language(List<Double> lst) {
        if (lst.size() == 0) {
            return false;
        }
        if (lst.size() == 1) {
            return true;
        }
        if (lst.size() == 2) {
            return lst.get(0) < lst.get(1);
        }

        List<Double> peaks = new ArrayList<>();
        List<Double> troughs = new ArrayList<>();
        for (int i = 1; i < lst.size() - 1; i++) {
            if (lst.get(i) > lst.get(i - 1) && lst.get(i) > lst.get(i + 1)) {
                peaks.add(lst.get(i));
            } else if (lst.get(i) < lst.get(i - 1) && lst.get(i) < lst.get(i + 1)) {
                troughs.add(lst.get(i));
            }
        }
        if (lst.get(lst.size() - 1) < lst.get(lst.size() - 2)) {
            troughs.add(lst.get(lst.size() - 1));
        } else if (lst.get(lst.size() - 1) > lst.get(lst.size() - 2)) {
            peaks.add(lst.get(lst.size() - 1));
        }

        // peaks strictly decreasing
        for (int i = 1; i < peaks.size(); i++) {
            if (peaks.get(i) >= peaks.get(i - 1)) {
                return false;
            }
        }
        // troughs strictly decreasing
        for (int i = 1; i < troughs.size(); i++) {
            if (troughs.get(i) >= troughs.get(i - 1)) {
                return false;
            }
        }
        return true;
    }

    public boolean twoA(List<Double> lst) {
        // Must contain exactly one unique value
        Set<Double> unique = new HashSet<>(lst);
        if (unique.size() != 1) {
            return false;
        } else {
            // Length must be even
            return lst.size() % 2 == 0;
        }
    }

    public boolean threeA(List<Double> lst) {
        // Must contain exactly one unique value
        Set<Double> unique = new HashSet<>(lst);
        if (unique.size() != 1) {
            return false;
        } else {
            // Length must be even
            return lst.size() % 3 == 0;
        }
    }

    public boolean fourA(List<Double> lst) {
        // Must contain exactly one unique value
        Set<Double> unique = new HashSet<>(lst);
        if (unique.size() != 1) {
            return false;
        } else {
            // Length must be even
            return lst.size() % 4 == 0;
        }
    }

    public boolean fiveA(List<Double> lst) {
        // Must contain exactly one unique value
        Set<Double> unique = new HashSet<>(lst);
        if (unique.size() != 1) {
            return false;
        } else {
            // Length must be even
            return lst.size() % 5 == 0;
        }
    }

    public boolean sixA(List<Double> lst) {
        // Must contain exactly one unique value
        Set<Double> unique = new HashSet<>(lst);
        if (unique.size() != 1) {
            return false;
        } else {
            // Length must be even
            return lst.size() % 6 == 0;
        }
    }

    public boolean sevenA(List<Double> lst) {
        // Must contain exactly one unique value
        Set<Double> unique = new HashSet<>(lst);
        if (unique.size() != 1) {
            return false;
        } else {
            // Length must be even
            return lst.size() % 7 == 0;
        }
    }

    public boolean eightA(List<Double> lst) {
        // Must contain exactly one unique value
        Set<Double> unique = new HashSet<>(lst);
        if (unique.size() != 1) {
            return false;
        } else {
            // Length must be even
            return lst.size() % 8 == 0;
        }
    }

    public boolean nineA(List<Double> lst) {
        // Must contain exactly one unique value
        Set<Double> unique = new HashSet<>(lst);
        if (unique.size() != 1) {
            return false;
        } else {
            // Length must be even
            return lst.size() % 9 == 0;
        }
    }

    public boolean tenA(List<Double> lst) {
        // Must contain exactly one unique value
        Set<Double> unique = new HashSet<>(lst);
        if (unique.size() != 1) {
            return false;
        } else {
            return lst.size() % 10 == 0;
        }
    }

    public boolean twentyA(List<Double> lst) {
        // Must contain exactly one unique value
        Set<Double> unique = new HashSet<>(lst);
        if (unique.size() != 1) {
            return false;
        } else {
            // Length must be even
            return lst.size() % 20 == 0;
        }
    }


    public boolean elevenA(List<Double> lst) {
        // Must contain exactly one unique value
        Set<Double> unique = new HashSet<>(lst);
        if (unique.size() != 1) {
            return false;
        } else {
            // Length must be even
            return lst.size() % 11 == 0;
        }
    }



    public boolean twoAB(List<Double> lst) {
        // Must have exactly two distinct symbols (a and b)
        Set<Double> unique = new HashSet<>(lst);
        if (unique.size() != 2) {
            return false;
        }

        // Must have even length (since it's (ab)*)
        if (lst.size() % 2 != 0) {
            return false;
        }

        // Extract a and b from the first two elements
        double a = lst.get(0);
        double b = lst.get(1);

        // Must start with a and alternate a,b,a,b,...
        for (int i = 0; i < lst.size(); i++) {
            double expected = (i % 2 == 0) ? a : b;
            if (Double.compare(lst.get(i), expected) != 0) {
                return false;
            }
        }

        return true;
    }

    public boolean threeAB(List<Double> lst) {
        // (ababab)* requires exactly two distinct symbols
        Set<Double> unique = new HashSet<>(lst);
        if (unique.size() != 2) {
            return false;
        }

        // Length must be a multiple of 6
        if (lst.size() % 6 != 0) {
            return false;
        }

        double a = lst.get(0);
        double b = lst.get(1);

        // Define one block pattern: [a, b, a, b, a, b]
        double[] pattern = {a, b, a, b, a, b};

        // Check each 6-element block
        for (int i = 0; i < lst.size(); i += 6) {
            for (int j = 0; j < 6; j++) {
                if (Double.compare(lst.get(i + j), pattern[j]) != 0) {
                    return false;
                }
            }
        }

        return true;
    }

    public boolean fourAB(List<Double> lst) {
        // (ababab)* requires exactly two distinct symbols
        Set<Double> unique = new HashSet<>(lst);
        if (unique.size() != 2) {
            return false;
        }

        // Length must be a multiple of 6
        if (lst.size() % 8 != 0) {
            return false;
        }

        double a = lst.get(0);
        double b = lst.get(1);

        // Define one block pattern: [a, b, a, b, a, b]
        double[] pattern = {a, b, a, b, a, b, a, b};

        // Check each 6-element block
        for (int i = 0; i < lst.size(); i += 8) {
            for (int j = 0; j < 8; j++) {
                if (Double.compare(lst.get(i + j), pattern[j]) != 0) {
                    return false;
                }
            }
        }

        return true;
    }

    public boolean fiveAB(List<Double> lst) {
        // (ababab)* requires exactly two distinct symbols
        Set<Double> unique = new HashSet<>(lst);
        if (unique.size() != 2) {
            return false;
        }

        // Length must be a multiple of 6
        if (lst.size() % 10 != 0) {
            return false;
        }

        double a = lst.get(0);
        double b = lst.get(1);

        // Define one block pattern: [a, b, a, b, a, b]
        double[] pattern = {a, b, a, b, a, b, a, b, a ,b};

        // Check each 6-element block
        for (int i = 0; i < lst.size(); i += 10) {
            for (int j = 0; j < 10; j++) {
                if (Double.compare(lst.get(i + j), pattern[j]) != 0) {
                    return false;
                }
            }
        }

        return true;
    }

    public boolean upDownUp(List<Double> lst) {
        // Need at least 4 points to realize: up, down, up (3 transitions)
        if (lst == null || lst.size() < 4) {
            return false;
        }

        int n = lst.size();
        double[] diffs = new double[n - 1];
        for (int i = 1; i < n; i++) {
            diffs[i - 1] = lst.get(i) - lst.get(i - 1);
        }

        // No flat differences
        for (double d : diffs) {
            if (Double.compare(d, 0.0) == 0) {
                return false;
            }
        }

        // The pattern must start up, then down, then up
        // True = up, False = down
        boolean[] pattern = {true, false, true};

        // We only care about the first 3 transitions
        for (int i = 0; i < 3; i++) {
            double d = diffs[i];
            boolean expectedUp = pattern[i];
            if (expectedUp) {
                if (Double.compare(d, 0.0) <= 0) {
                    return false;
                }
            } else {
                if (Double.compare(d, 0.0) >= 0) {
                    return false;
                }
            }
        }

        // Remaining transitions (if any) must continue the "up" trend
        for (int i = 3; i < diffs.length; i++) {
            if (Double.compare(diffs[i], 0.0) <= 0) {
                return false;
            }
        }

        return true;
    }

    public boolean upDownUpDownPrefix(List<Double> lst) {
        // Begins with: up, down, up, down (first 4 transitions). Requires at least 5 points.
        if (lst == null || lst.size() < 5) {
            return false;
        }

        int n = lst.size();
        double[] diffs = new double[n - 1];
        for (int i = 1; i < n; i++) {
            diffs[i - 1] = lst.get(i) - lst.get(i - 1);
        }

        // Ensure first four diffs are non-zero
        for (int i = 0; i < 4; i++) {
            if (Double.compare(diffs[i], 0.0) == 0) {
                return false;
            }
        }

        // Pattern: up, down, up, down
        boolean[] pattern = {true, false, true, false};
        for (int i = 0; i < 4; i++) {
            double d = diffs[i];
            boolean expectUp = pattern[i];
            if (expectUp) {
                if (Double.compare(d, 0.0) <= 0) {
                    return false;
                }
            } else {
                if (Double.compare(d, 0.0) >= 0) {
                    return false;
                }
            }
        }

        return true;
    }




}
