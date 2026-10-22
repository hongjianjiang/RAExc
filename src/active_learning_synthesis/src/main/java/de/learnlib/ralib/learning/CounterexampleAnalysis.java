/*
 * Copyright (C) 2014-2025 The LearnLib Contributors
 * This file is part of LearnLib, http://www.learnlib.de/.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package de.learnlib.ralib.learning;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.learnlib.ralib.data.Constants;
import de.learnlib.ralib.data.DataValue;
import de.learnlib.ralib.data.Mapping;
import de.learnlib.ralib.learning.rastar.CEAnalysisResult;
import de.learnlib.ralib.oracles.Branching;
import de.learnlib.ralib.oracles.SDTLogicOracle;
import de.learnlib.ralib.oracles.TreeOracle;
import de.learnlib.ralib.oracles.mto.SymbolicSuffixRestrictionBuilder;
import de.learnlib.ralib.theory.SDT;
import de.learnlib.ralib.words.PSymbolInstance;
import de.learnlib.ralib.words.ParameterizedSymbol;
import gov.nasa.jpf.constraints.api.Expression;
import net.automatalib.word.Word;

/**
 * Analyzes counterexamples in a binary search
 * as described in SEFM 2014.
 *
 * @author falk
 */
public class CounterexampleAnalysis {

    private static final long SLOW_STEP_NS = 1_000_000_000L; // 1s
    private static volatile long treeQueryTimeoutMs = 5_000L;
    private static volatile long hasCounterexampleTimeoutMs = 5_000L;

    private static final ExecutorService CE_TIMEOUT_EXECUTOR = Executors.newCachedThreadPool(new ThreadFactory() {
        @Override
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r, "ce-analysis-timeout-worker");
            t.setDaemon(true);
            return t;
        }
    });

    private final TreeOracle sulOracle;

    private final TreeOracle hypOracle;

    private final Hypothesis hypothesis;

    private final SDTLogicOracle sdtOracle;

    private final SymbolicSuffixRestrictionBuilder restrictionBuilder;

    private final Map<Word<PSymbolInstance>, LocationComponent> components;

    private final Constants consts;

    private enum IndexResult {HAS_CE_AND_REFINES, HAS_CE_NO_REFINE, NO_CE}

    private static final Logger LOGGER = LoggerFactory.getLogger(CounterexampleAnalysis.class);

    public CounterexampleAnalysis(TreeOracle sulOracle, TreeOracle hypOracle,
            Hypothesis hypothesis, SDTLogicOracle sdtOracle,
            Map<Word<PSymbolInstance>, LocationComponent> components, Constants consts) {

        this.sulOracle = sulOracle;
        this.hypOracle = hypOracle;
        this.hypothesis = hypothesis;
        this.sdtOracle = sdtOracle;
        this.components = components;
        this.consts = consts;
        this.restrictionBuilder = sulOracle.getRestrictionBuilder();
    }

    public static void configureTimeouts(long treeQueryTimeoutMs, long hasCounterexampleTimeoutMs) {
        if (treeQueryTimeoutMs > 0) {
            CounterexampleAnalysis.treeQueryTimeoutMs = treeQueryTimeoutMs;
        }
        if (hasCounterexampleTimeoutMs > 0) {
            CounterexampleAnalysis.hasCounterexampleTimeoutMs = hasCounterexampleTimeoutMs;
        }
    }

    public CEAnalysisResult analyzeCounterexample(Word<PSymbolInstance> ce) {
        long start = System.nanoTime();

        int idx = binarySearch(ce);
        //int idx = linearBackWardsSearch(ce);

        Word<PSymbolInstance> prefix = ce.prefix(idx);
        Word<PSymbolInstance> suffix = ce.suffix(ce.length() -idx);
        SymbolicSuffix symSuffix = new SymbolicSuffix(prefix, suffix, restrictionBuilder);

        long end = System.nanoTime();
        if ((end - start) > SLOW_STEP_NS) {
            System.out.println("TIMING[CEAnalysis]: analyzeCounterexample total=" + formatSeconds(end - start)
                    + " (ceLen=" + ce.length() + ", splitIdx=" + idx + ")");
        }

        return new CEAnalysisResult(prefix, symSuffix);
    }

    private IndexResult computeIndex(Word<PSymbolInstance> ce, int idx) {
        long totalStart = System.nanoTime();

        long prepStart = System.nanoTime();
        Word<PSymbolInstance> prefix = ce.prefix(idx);
        //System.out.println(idx + "  " + prefix);
        Word<PSymbolInstance> location = hypothesis.transformAccessSequence(prefix);
        Word<PSymbolInstance> transition = hypothesis.transformTransitionSequence(
            ce.prefix(idx+1));

        Word<PSymbolInstance> suffix = ce.suffix(ce.length() -idx);
        SymbolicSuffix symSuffix = new SymbolicSuffix(prefix, suffix, restrictionBuilder);
        long prepEnd = System.nanoTime();

        long treeStart = System.nanoTime();
        SDT resHyp = runWithTimeout(
            () -> hypOracle.treeQuery(location, symSuffix),
            treeQueryTimeoutMs,
            "hypOracle.treeQuery",
            idx
        );
        SDT resSul = runWithTimeout(
            () -> sulOracle.treeQuery(location, symSuffix),
            treeQueryTimeoutMs,
            "sulOracle.treeQuery",
            idx
        );
        long treeEnd = System.nanoTime();

        LOGGER.trace("------------------------------------------------------");
        LOGGER.trace("Computing index: {}", idx);
        LOGGER.trace("Prefix: {}", prefix);
        LOGGER.trace("SymSuffix: {}", symSuffix);
        LOGGER.trace("Location: {}", location);
        LOGGER.trace("Transition: {}", transition);
        LOGGER.trace("SDT HYP: {}", resHyp);
        LOGGER.trace("SDT SYS: {}", resSul);
        LOGGER.trace("------------------------------------------------------");

        LocationComponent c = components.get(location);
        ParameterizedSymbol act = transition.lastSymbol().getBaseSymbol();
        //System.out.println(c.getBranching(act).getBranches());
        Expression<Boolean> g = c.getBranching(act).getBranches().get(transition);

        long hasCeStart = System.nanoTime();
        boolean hasCE = runWithTimeout(
            () -> sdtOracle.hasCounterexample(location, resHyp, resSul, g, transition),
            hasCounterexampleTimeoutMs,
            "sdtOracle.hasCounterexample",
            idx
        );
        long hasCeEnd = System.nanoTime();

        if (!hasCE) {
            long totalEnd = System.nanoTime();
            
            return IndexResult.NO_CE;
        }

        Set<DataValue> pSul = resSul.getDataValues();
        Set<DataValue> pHyp = c.getPrimePrefix().getAssignment().keySet();

        boolean sulHasMoreRegs = !pHyp.containsAll(pSul);

        long refineStart = System.nanoTime();
        boolean hypRefinesTransition = hypRefinesTransitions(location, act, resSul);
        long refineEnd = System.nanoTime();

//        System.out.println("sulHasMoreRegs: " + sulHasMoreRegs);
//        System.out.println("hypRefinesTransition: " + hypRefinesTransition);

        IndexResult result = (sulHasMoreRegs || !hypRefinesTransition) ?
                IndexResult.HAS_CE_AND_REFINES : IndexResult.HAS_CE_NO_REFINE;

        long totalEnd = System.nanoTime();
        if ((totalEnd - totalStart) > SLOW_STEP_NS) {
            System.out.println("TIMING[CEAnalysis]: computeIndex(" + idx + ")=" + formatSeconds(totalEnd - totalStart)
                + " [prep=" + formatSeconds(prepEnd - prepStart)
                + ", treeQuery=" + formatSeconds(treeEnd - treeStart)
                + ", hasCE=" + formatSeconds(hasCeEnd - hasCeStart)
                + ", refine=" + formatSeconds(refineEnd - refineStart)
                + ", result=" + result + "]");
        }

        return result;
    }

    private boolean hypRefinesTransitions(Word<PSymbolInstance> prefix,
                                          ParameterizedSymbol action, SDT sdtSUL) {

        Branching branchSul = sulOracle.getInitialBranching(prefix, action, sdtSUL);
        LocationComponent c = components.get(prefix);
        Branching branchHyp = c.getBranching(action);

        //System.out.println("Branching Hyp:");
        for (Map.Entry<Word<PSymbolInstance>, Expression<Boolean>> e : branchHyp.getBranches().entrySet()) {
            //System.out.println(e.getKey() + " -> " + e.getValue());
        }
        //System.out.println("Branching Sys:");
        for (Map.Entry<Word<PSymbolInstance>, Expression<Boolean>> e : branchSul.getBranches().entrySet()) {
            //System.out.println(e.getKey() + " -> " + e.getValue());
        }

        for (Expression<Boolean> guardHyp : branchHyp.getBranches().values()) {
            boolean refines = false;
            for (Expression<Boolean> guardSul : branchSul.getBranches().values()) {
                if (sdtOracle.doesRefine(guardHyp, guardSul, new Mapping<>())) {
                    refines = true;
                    break;
                }
            }

            if (!refines) {
                return false;
            }
        }

        return true;
    }

//    private int linearBackWardsSearch(Word<PSymbolInstance> ce) {
//
//        assert ce.length() > 1;
//
//        IndexResult[] results = new IndexResult[ce.length()];
//        results[ce.length()-1] = IndexResult.NO_CE;
//
//        int idx = ce.length()-2;
//
//        while (idx >= 0) {
//            IndexResult res = computeIndex(ce, idx);
//            results[idx] = res;
//            if (res != IndexResult.NO_CE) {
//                break;
//            }
//            idx--;
//        }
//
//        assert (idx >= 0);
//
//        // if in the last step there was no counterexample,
//        // we have to move one step to the left
//        if (results[idx] == IndexResult.NO_CE) {
//            assert idx > 0;
//            idx--;
//        }
//
//        // if the current index has no refinement use the
//        // suffix of the next index
//        if (results[idx] == IndexResult.HAS_CE_NO_REFINE) {
//            idx++;
//        }
//
//        return idx;
//    }

    private int binarySearch(Word<PSymbolInstance> ce) {

        long start = System.nanoTime();

        assert ce.length() > 1;

        IndexResult[] results = new IndexResult[ce.length()];
        //results[0] = IndexResult.HAS_CE_NO_REFINE;
        //results[ce.length()-1] = IndexResult.NO_CE;

        int min = 0;
        int max = ce.length() - 1;
        int mid = -1;

        while (max >= min) {

            mid = (max+min+1) / 2;

            IndexResult res;
            try {
                res = computeIndex(ce, mid);
            } catch (CEAnalysisTimeoutException e) {
                // Fallback: keep refinement moving with a conservative split.
                int fallbackIdx = Math.max(0, mid - 1);
                System.out.println("WARN[CEAnalysis]: timeout at mid=" + mid +
                    ", fallback split idx=" + fallbackIdx + " (" + e.getMessage() + ")");
                return fallbackIdx;
            }
            LOGGER.trace("" + res);

            results[mid] = res;
            if (res == IndexResult.NO_CE) {
                max = mid -1;
            } else {
                min = mid +1;
            }
        }

        assert mid >= 0;

        int idx = mid;

        //System.out.println(Arrays.toString(results));
        //System.out.println(idx + " : " + results[idx]);

        // if in the last step there was no counterexample,
        // we have to move one step to the left
        if (results[idx] == IndexResult.NO_CE) {
            assert idx > 0;
            idx--;
        }

        // if the current index has no refinement use the
        // suffix of the next index
        if (results[idx] == IndexResult.HAS_CE_NO_REFINE) {
            idx++;
        }

        long end = System.nanoTime();
        if ((end - start) > SLOW_STEP_NS) {
            System.out.println("TIMING[CEAnalysis]: binarySearch=" + formatSeconds(end - start)
                    + " (ceLen=" + ce.length() + ", finalIdx=" + idx + ")");
        }

        //System.out.println("IDX: " + idx);

        return idx;
    }

    private static String formatSeconds(long nanos) {
        return String.format("%.3fs", nanos / 1_000_000_000.0);
    }

    private static <T> T runWithTimeout(Callable<T> callable, long timeoutMs,
            String stage, int idx) {
        Future<T> future = CE_TIMEOUT_EXECUTOR.submit(callable);
        try {
            return future.get(timeoutMs, TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            future.cancel(true);
            throw new CEAnalysisTimeoutException("Timeout in " + stage + " at index " + idx +
                " after " + timeoutMs + "ms", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new CEAnalysisTimeoutException("Interrupted in " + stage + " at index " + idx, e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            throw new CEAnalysisTimeoutException("Execution failed in " + stage + " at index " + idx +
                ": " + cause.getMessage(), cause);
        }
    }

    private static final class CEAnalysisTimeoutException extends RuntimeException {
        CEAnalysisTimeoutException(String message, Throwable cause) {
            super(message, cause);
        }
    }

}
