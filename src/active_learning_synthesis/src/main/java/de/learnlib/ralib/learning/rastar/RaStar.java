/*
 * Copyright (C) 2014-2015 The LearnLib Contributors
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
package de.learnlib.ralib.learning.rastar;

import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.learnlib.logging.Category;
import de.learnlib.query.DefaultQuery;
import de.learnlib.ralib.data.Constants;
import de.learnlib.ralib.learning.AutomatonBuilder;
import de.learnlib.ralib.learning.CounterexampleAnalysis;
import de.learnlib.ralib.learning.Hypothesis;
import de.learnlib.ralib.learning.IOAutomatonBuilder;
import de.learnlib.ralib.learning.LocationComponent;
import de.learnlib.ralib.learning.QueryStatistics;
import de.learnlib.ralib.learning.RaLearningAlgorithm;
import de.learnlib.ralib.learning.RaLearningAlgorithmName;
import de.learnlib.ralib.learning.SymbolicSuffix;
import de.learnlib.ralib.oracles.SDTLogicOracle;
import de.learnlib.ralib.oracles.TreeOracle;
import de.learnlib.ralib.oracles.TreeOracleFactory;
import de.learnlib.ralib.words.OutputSymbol;
import de.learnlib.ralib.words.PSymbolInstance;
import de.learnlib.ralib.words.ParameterizedSymbol;
import net.automatalib.word.Word;

/**
 * Learning algorithm for register automata
 *
 * @author falk
 */
public class RaStar implements RaLearningAlgorithm {

    public static final Word<PSymbolInstance> EMPTY_PREFIX = Word.epsilon();

    public static final SymbolicSuffix EMPTY_SUFFIX = new SymbolicSuffix(
            Word.epsilon(), Word.epsilon());

    private final ObservationTable obs;

    private final Constants consts;

    private final Deque<DefaultQuery<PSymbolInstance, Boolean>> counterexamples =
            new LinkedList<>();

    private Hypothesis hyp = null;

    private final TreeOracle sulOracle;

    private final SDTLogicOracle sdtLogicOracle;

    private final TreeOracleFactory hypOracleFactory;

    private QueryStatistics queryStats = null;

    private final boolean ioMode;

    private static final Logger LOGGER = LoggerFactory.getLogger(RaStar.class);

    public RaStar(TreeOracle oracle, TreeOracleFactory hypOracleFactory,
            SDTLogicOracle sdtLogicOracle, Constants consts, boolean ioMode,
            ParameterizedSymbol ... inputs) {

        this.ioMode = ioMode;
        this.obs = new ObservationTable(oracle, ioMode, consts, inputs);
        this.consts = consts;

        this.obs.addPrefix(EMPTY_PREFIX);
        this.obs.addSuffix(EMPTY_SUFFIX);

        //TODO: make this optional
        for (ParameterizedSymbol ps : inputs) {
            if (ps instanceof OutputSymbol) {
                this.obs.addSuffix(new SymbolicSuffix(ps));
            }
        }

        this.sulOracle = oracle;
        this.sdtLogicOracle = sdtLogicOracle;
        this.hypOracleFactory = hypOracleFactory;
    }

    public RaStar(TreeOracle oracle, TreeOracleFactory hypOracleFactory,
            SDTLogicOracle sdtLogicOracle, Constants consts,
            ParameterizedSymbol ... inputs) {

        this(oracle, hypOracleFactory, sdtLogicOracle, consts, false, inputs);
    }

    @Override
    public void learn() {
        long learnStart = System.nanoTime();
        System.out.println("TIMING[RaStar]: learn() entered");

        if (hyp != null) {
            long preCeStart = System.nanoTime();
            analyzeCounterExample();
            long preCeEnd = System.nanoTime();
            System.out.println("TIMING[RaStar]: pre-loop CE analysis=" + formatSeconds(preCeEnd - preCeStart));
        }

        do {
            LOGGER.info(Category.PHASE, "completing observation table");
            long obsStart = System.nanoTime();
            int completeIterations = 0;
            long lastHeartbeat = obsStart;
            while(! obs.complete()) {
                completeIterations++;
                long now = System.nanoTime();
                if (now - lastHeartbeat >= 5_000_000_000L) {
                    System.out.println("HEARTBEAT[RaStar]: still completing observation table, iterations="
                            + completeIterations + ", elapsed=" + formatSeconds(now - obsStart)
                            + ", " + obs.debugSnapshot());
                    lastHeartbeat = now;
                }
            }
            long obsEnd = System.nanoTime();
            LOGGER.info(Category.PHASE, "completed observation table");
            System.out.println("TIMING[RaStar]: observation-table complete in " + formatSeconds(obsEnd - obsStart)
                    + " (iterations=" + completeIterations + ")");

            //System.out.println(obs.toString());

            long buildStart = System.nanoTime();
            Map<Word<PSymbolInstance>, LocationComponent> components = new LinkedHashMap<Word<PSymbolInstance>, LocationComponent>();
            components.putAll(obs.getComponents());
            AutomatonBuilder ab = new AutomatonBuilder(components, consts);
            hyp = ab.toRegisterAutomaton();
            long buildEnd = System.nanoTime();
            System.out.println("TIMING[RaStar]: hypothesis build=" + formatSeconds(buildEnd - buildStart));

            //FIXME: the default logging appender cannot log models and data structures
            //System.out.println(hyp.toString());
            LOGGER.info(Category.MODEL, "{}", hyp);

            long ceStart = System.nanoTime();
            boolean refined = analyzeCounterExample();
            long ceEnd = System.nanoTime();
            System.out.println("TIMING[RaStar]: post-build CE analysis=" + formatSeconds(ceEnd - ceStart)
                    + " (refined=" + refined + ")");

            if (!refined) {
                break;
            }
        } while (true);

        if (queryStats != null)
        	queryStats.hypothesisConstructed();

        long learnEnd = System.nanoTime();
        System.out.println("TIMING[RaStar]: learn() total=" + formatSeconds(learnEnd - learnStart));
    }

    @Override
    public void addCounterexample(DefaultQuery<PSymbolInstance, Boolean> ce) {
        LOGGER.info(Category.EVENT, "adding counterexample: {}", ce);
        counterexamples.add(ce);
    }

    private boolean analyzeCounterExample() {
        long analysisStart = System.nanoTime();
        LOGGER.info(Category.PHASE, "Analyzing Counterexample");
        if (counterexamples.isEmpty()) {
            long analysisEnd = System.nanoTime();
            System.out.println("TIMING[RaStar]: analyzeCounterExample empty queue="
                    + formatSeconds(analysisEnd - analysisStart));
            return false;
        }

        TreeOracle hypOracle = hypOracleFactory.createTreeOracle(hyp);

        Map<Word<PSymbolInstance>, LocationComponent> components = new LinkedHashMap<Word<PSymbolInstance>, LocationComponent>();
        components.putAll(obs.getComponents());
        CounterexampleAnalysis analysis = new CounterexampleAnalysis(
                sulOracle, hypOracle, hyp, sdtLogicOracle, components, consts);

        DefaultQuery<PSymbolInstance, Boolean> ce = counterexamples.peek();

        // check if ce still is a counterexample ...
        boolean hypce = hyp.accepts(ce.getInput());
        boolean sulce = ce.getOutput();
        if (hypce == sulce) {
            LOGGER.info(Category.EVENT, "word is not a counterexample: {} - {}", ce, sulce);
            counterexamples.poll();
            long analysisEnd = System.nanoTime();
            System.out.println("TIMING[RaStar]: analyzeCounterExample stale CE="
                    + formatSeconds(analysisEnd - analysisStart));
            return false;
        }

        //System.out.println("CE ANALYSIS: " + ce + " ; S:" + sulce + " ; H:" + hypce);

        if (queryStats != null)
        	queryStats.analyzingCounterExample();

        long ceComputeStart = System.nanoTime();
        CEAnalysisResult res = analysis.analyzeCounterexample(ce.getInput());
        long ceComputeEnd = System.nanoTime();

        if (queryStats != null) {
        	queryStats.processingCounterExample();
        	queryStats.analyzeCE(ce.getInput());
        }

        long addSuffixStart = System.nanoTime();
        obs.addSuffix(res.getSuffix());
        long addSuffixEnd = System.nanoTime();

        long analysisEnd = System.nanoTime();
        System.out.println("TIMING[RaStar]: analyzeCounterExample total=" + formatSeconds(analysisEnd - analysisStart)
                + " (ceAnalysis=" + formatSeconds(ceComputeEnd - ceComputeStart)
                + ", addSuffix=" + formatSeconds(addSuffixEnd - addSuffixStart)
                + ", queueSize=" + counterexamples.size() + ")");
        return true;
    }

    private static String formatSeconds(long nanos) {
        return String.format("%.3fs", nanos / 1_000_000_000.0);
    }

    @Override
    public Hypothesis getHypothesis() {
    	Map<Word<PSymbolInstance>, LocationComponent> components = new LinkedHashMap<Word<PSymbolInstance>, LocationComponent>();
    	components.putAll(obs.getComponents());
        AutomatonBuilder ab = new AutomatonBuilder(components, consts);
        if (ioMode) {
            ab = new IOAutomatonBuilder(components, consts);
        }
        return ab.toRegisterAutomaton();
    }

    // TODO: this should not be a public method permanently!
    public Map<Word<PSymbolInstance>, LocationComponent> getComponents() {
        Map<Word<PSymbolInstance>, LocationComponent> components = new LinkedHashMap<Word<PSymbolInstance>, LocationComponent>();
        components.putAll(obs.getComponents());
        return components;
    }

    @Override
    public void setStatisticCounter(QueryStatistics queryStats) {
    	this.queryStats = queryStats;
    }

    @Override
    public QueryStatistics getQueryStatistics() {
    	return queryStats;
    }

    @Override
    public RaLearningAlgorithmName getName() {
        return RaLearningAlgorithmName.RASTAR;
    }
}
