package de.learnlib.ralib.sul;

import de.learnlib.query.Query;
import de.learnlib.ralib.oracles.MyOracle;
import de.learnlib.ralib.tools.config.Configuration;
import de.learnlib.ralib.words.PSymbolInstance;
import net.automatalib.word.Word;

import java.util.Collection;

public class TraderOracle implements MyOracle {
    private final TraderSUL sul;
    private Configuration config;

    public TraderOracle(TraderSUL sul, Configuration config) {
        this.sul = sul;
        this.config = config;
    }

    @Override
    public void processQueries(Collection<? extends Query<PSymbolInstance, Boolean>> collection) {
        for (Query<PSymbolInstance, Boolean> q : collection) {
            Word<PSymbolInstance> trace = q.getInput();
            boolean answer = accepts(trace);
            q.answer(answer);
        }
    }

    public String info(Word<PSymbolInstance> trace) {
        StringBuilder sb = new StringBuilder();
        for (PSymbolInstance symbol : trace) {
            sb.append(symbol.getParameterValues()[0].getValue()).append(',');
        }
        return sb.toString();
    }

    @Override
    public boolean accepts(Word<PSymbolInstance> trace) {
        boolean accepted = false;
        boolean fixTraceLen = Boolean.parseBoolean(this.config.getProperty("fixTraceLen"));
        int maxTraceLen = Integer.parseInt(this.config.getProperty("maxTraceLen"));
        if (trace.isEmpty()) {
            // always reject epsilon
            accepted = false;
        } else if (fixTraceLen && trace.length() != maxTraceLen) {
            // only accepts fixed-length traces
            accepted = false;
        } else {
            Boolean seen = TraceCache.get(trace);
            if (seen != null) {
                accepted = seen;
            } else {
                sul.pre();
                for (int i = 0; i < trace.length(); i++) {
                    PSymbolInstance price = trace.getSymbol(i);
                    accepted = sul.step(price);
                }
                TraceCache.put(trace, accepted);
            }
        }
        return accepted;
    }
}