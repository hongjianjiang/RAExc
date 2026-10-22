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
import java.util.List;
import java.util.Map;



import de.learnlib.logging.Category;
import de.learnlib.ralib.data.Constants;
import de.learnlib.ralib.learning.SymbolicSuffix;
import de.learnlib.ralib.oracles.TreeOracle;
import de.learnlib.ralib.oracles.mto.SymbolicSuffixRestrictionBuilder;
import de.learnlib.ralib.words.PSymbolInstance;
import de.learnlib.ralib.words.ParameterizedSymbol;
import net.automatalib.word.Word;

/**
 * An observation table.
 *
 * @author falk
 */
class ObservationTable {

    private static final long SLOW_PHASE_NS = 2_000_000_000L; // 2s

    private final List<SymbolicSuffix> suffixes = new LinkedList<>();

    private final Map<Word<PSymbolInstance>, Component> components
            = new LinkedHashMap<>();

    private final Deque<SymbolicSuffix> newSuffixes = new LinkedList<>();

    private final Deque<Word<PSymbolInstance>> newPrefixes = new LinkedList<>();

    private final Deque<Component> newComponents = new LinkedList<>();

    private final TreeOracle oracle;

    private final ParameterizedSymbol[] inputs;

    private final boolean ioMode;

    private final Constants consts;

    private final SymbolicSuffixRestrictionBuilder restrictionBuilder;



    public ObservationTable(TreeOracle oracle, boolean ioMode,
            Constants consts, ParameterizedSymbol ... inputs) {
        this.oracle = oracle;
        this.inputs = inputs;
        this.ioMode = ioMode;
        this.consts = consts;
        this.restrictionBuilder = oracle.getRestrictionBuilder();
    }

    void addComponent(Component c) {
        newComponents.add(c);
    }

    void addSuffix(SymbolicSuffix suffix) {
        newSuffixes.add(suffix);
    }

    void addPrefix(Word<PSymbolInstance> prefix) {
        newPrefixes.add(prefix);
    }

    boolean complete() {
        if (!newComponents.isEmpty()) {
            processNewComponent();
            return false;
        }
        if (!newPrefixes.isEmpty()) {
            processNewPrefix();
            return false;
        }
        if (!newSuffixes.isEmpty()) {
            processNewSuffix();
            checkBranchingCompleteness();
            return false;
        }
        boolean done = checkVariableConsistency();
        return done;
    }

    private boolean checkBranchingCompleteness() {
        boolean ret = true;
        for (Component c : components.values()) {
            boolean ub = c.updateBranching(oracle);
            ret = ret && ub;
        }
        return ret;
    }

    private boolean checkVariableConsistency() {
        for (Component c : components.values()) {
            if (!c.checkVariableConsistency()) {
                return false;
            }
        }
        return true;
    }

    private void processNewSuffix() {
        SymbolicSuffix suffix = newSuffixes.poll();
        suffixes.add(suffix);
        for (Component c : components.values()) {
            c.addSuffix(suffix, oracle);
        }
    }

    private void processNewPrefix() {
        Word<PSymbolInstance> prefix = newPrefixes.poll();
        Row r = Row.computeRow(oracle, prefix, suffixes, ioMode);
        for (Component c : components.values()) {
            if (c.addRow(r)) {
                return;
            }
        }
        Component c = new Component(r, this, ioMode, consts, restrictionBuilder);
        addComponent(c);
    }

    private void processNewComponent() {
        Component c = newComponents.poll();
        components.put(c.getAccessSequence(), c);
        c.start(oracle, inputs);
    }

    Map<Word<PSymbolInstance>, Component> getComponents() {
        return components;
    }

    String debugSnapshot() {
        return "components=" + components.size()
                + ", suffixes=" + suffixes.size()
                + ", queuedComponents=" + newComponents.size()
                + ", queuedPrefixes=" + newPrefixes.size()
                + ", queuedSuffixes=" + newSuffixes.size();
    }

    private static String formatSeconds(long nanos) {
        return String.format("%.3fs", nanos / 1_000_000_000.0);
    }

    private void logSlowPhase(String phase, long phaseStartNs) {
        // No-op: removed printout
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("OBS *******************************************************************\n");
        for (Component c : getComponents().values()) {
            c.toString(sb);
        }
        sb.append("***********************************************************************\n");
        return sb.toString();
    }

}
