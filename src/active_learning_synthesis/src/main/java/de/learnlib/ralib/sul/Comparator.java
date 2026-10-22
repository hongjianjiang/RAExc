package de.learnlib.ralib.sul;

import java.math.BigDecimal;

interface Comparator {
    Boolean check(BigDecimal curr, BigDecimal last);
}