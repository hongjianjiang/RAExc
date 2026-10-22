package de.learnlib.ralib.sul;

import de.learnlib.exception.SULException;
import de.learnlib.ralib.data.DataType;
import de.learnlib.ralib.data.DataValue;
import de.learnlib.ralib.data.MNISTDataValue;
import de.learnlib.ralib.words.InputSymbol;
import de.learnlib.ralib.words.PSymbolInstance;
import de.learnlib.ralib.words.ParameterizedSymbol;

import java.math.BigDecimal;

public class TraderSUL {
    
    private int count1, count2;

    public enum Strategy {
        StrictIncreasing, StrictDecreasing, Swing, UpTrendSimple, UpTrend, Interval, Lang1, Lang2, Lang3, Lang4,
        Lang5, Lang6, Lang7, Alternating1A, Alternating1B, Alternating2A, Alternating2B, S1, S2, S3, S4, S5, S6, S7, S8
    }

    public static final DataType DOUBLE_TYPE = new DataType("double");
    public static final DataType INTEGER_TYPE = new DataType("int");
    public static final DataType DYCK_TYPE = new DataType("DYCK");

    ParameterizedSymbol action;
    DataType priceDataType;
    BigDecimal lastPeak = null, prev1 = null, prev2 = null, prev3 = null, prev4 = null;
    BigDecimal valley1, valley2, peak1, peak2;
    int traceLen = 0;
    boolean failed;
    Strategy strategy;

    public final ParameterizedSymbol[] getActionSymbols() {
        return new ParameterizedSymbol[]{action};
    }

    public TraderSUL(DataType dataType) {
        this.priceDataType = dataType;
        if (dataType == DOUBLE_TYPE) {
            this.action = new InputSymbol("price", DOUBLE_TYPE);
        } else if (dataType == INTEGER_TYPE) {
            this.action = new InputSymbol("price", INTEGER_TYPE);
        } else{
            this.action = new InputSymbol("pop", DYCK_TYPE);
        }
    }

    boolean LT(BigDecimal v1, BigDecimal v2) {
        return v1.compareTo(v2) < 0;
    }

    boolean LE(BigDecimal v1, BigDecimal v2) {
        return v1.compareTo(v2) <= 0;
    }

    public void pre() {
        lastPeak = prev1 = prev2 = prev3 = null;
        valley1 = valley2 = peak1 = peak2 = null;
        count1 = count2 = 0;
        failed = false;
        traceLen = 0;
    }

    public void post() {
        pre();
    }

    public PSymbolInstance createWord(ParameterizedSymbol action, double price) {
        return new PSymbolInstance(action, new DataValue(priceDataType, BigDecimal.valueOf(price)));
    }


    public PSymbolInstance createPrice(double price) {
        return new PSymbolInstance(action, new DataValue(priceDataType, BigDecimal.valueOf(price)));
    }

    /**
     * Creates a PSymbolInstance with MNISTDataValue containing original pixel values.
     * The row ID is computed from the pixels (pixels > 127 are treated as "on").
     * @param pixels The original 28 pixel values (0-255)
     */
    public PSymbolInstance createMNISTRow(int[] pixels) {
        MNISTDataValue dv = MNISTDataValue.fromPixels(pixels, priceDataType);
        return new PSymbolInstance(action, dv);
    }

    /**
     * Creates a PSymbolInstance with MNISTDataValue containing original pixel values.
     * @param rowId The abstracted row ID (28-bit integer) - ignored, computed from pixels
     * @param pixels The original 28 pixel values (0-255)
     * @deprecated Use createMNISTRow(int[] pixels) instead
     */
    public PSymbolInstance createMNISTRow(long rowId, int[] pixels) {
        return createMNISTRow(pixels);
    }

    /**
     * Creates a PSymbolInstance with MNISTDataValue from a binary string.
     * @param binaryString 28-character binary string representing the row
     * @param pixels The original 28 pixel values (0-255), or null if not available
     */
    public PSymbolInstance createMNISTRowFromBinary(String binaryString, int[] pixels) {
        long rowId = Long.parseLong(binaryString, 2);
        if (pixels == null) {
            // Reconstruct pixels from binary (0 or 255)
            pixels = new int[28];
            for (int i = 0; i < 28; i++) {
                pixels[i] = binaryString.charAt(i) == '1' ? 255 : 0;
            }
        }
        return createMNISTRow(rowId, pixels);
    }


    static BigDecimal getPrice(PSymbolInstance input) {
        DataValue dv = input.getParameterValues()[0];
        Object price = dv.getValue();
        //DataValue val = new DataValue(DOUBLE_TYPE, price);
        if (price.getClass() == Integer.class) {
            return BigDecimal.valueOf(((Integer) price).doubleValue());
        } else {
            return (BigDecimal) price;
        }
    }

    /**
     * Return true if accepts the current symbol since from pre() is invoked
     **/
    public boolean step(PSymbolInstance input) throws SULException {
        switch (strategy) {
            case Interval:
                return step_interval(input);
            case StrictIncreasing:
                return step_increase(input);
            case StrictDecreasing:
                return step_decrease(input);
            case Swing:
                return step_swing(input);
            case UpTrendSimple:
                return step_up_trend_simple(input);
            case UpTrend:
                return step_up_trend(input);
            case Alternating1A:
                // (ab)* with a < b
                return step_alternating1(input, ((curr, last) -> LT(last, curr)));
            case Alternating1B:
                // (ab)* with a != b
                return step_alternating1(input, ((curr, last) -> !curr.equals(last)));
            case Alternating2A:
                // (ab)* with a < b
                return step_alternating2(input, ((curr, last) -> LT(last, curr)));
            case Alternating2B:
                // (ab)* with a != b
                return step_alternating2(input, ((curr, last) -> !curr.equals(last)));
            case Lang1:
                return step_lang1(input);
            case Lang2:
                return step_lang2(input);
            case Lang3:
                return step_lang3(input);
            case Lang4:
                return step_lang4(input);
            case Lang5:
                return step_lang5(input);
            case Lang6:
                return step_lang6(input);
        }
        throw new IllegalStateException();
    }

    public boolean step_increase(PSymbolInstance input) throws SULException {
        traceLen++;
        BigDecimal price = getPrice(input);
        prev2 = prev1;
        prev1 = price;
        failed = failed || (prev2 != null && !LT(prev1, prev2));
        return !failed;
    }

    public boolean step_decrease(PSymbolInstance input) throws SULException {
        traceLen++;
        BigDecimal price = getPrice(input);
        prev2 = prev1;
        prev1 = price;
        failed = failed || (prev2 != null && LE(prev1, prev2));
        return !failed;
    }

    //@Override
    public boolean step_swing(PSymbolInstance input) throws SULException {
        traceLen++;
        boolean accepted;
        BigDecimal price = getPrice(input);

        if (lastPeak != null && LT(lastPeak, price)) {
            accepted = true;
            lastPeak = null;
        } else {
            accepted = false;
        }
        prev3 = prev2;
        prev2 = prev1;
        prev1 = price;
        if (prev3 != null && LT(prev1, prev2) && LT(prev3, prev2)) {
            lastPeak = prev2;
        }
        return accepted;
    }

    public boolean step_up_trend(PSymbolInstance input) throws SULException {
        traceLen++;
        boolean accepted;
        BigDecimal price = getPrice(input);
        prev3 = prev2;
        prev2 = prev1;
        prev1 = price;
        if (prev3 != null) {
            if (LT(prev1, prev2) && LT(prev3, prev2)) {
                // prev2 is the latest peak
                peak2 = peak1;
                peak1 = prev2;
            }
            if (LT(prev2, prev1) && LT(prev2, prev3)) {
                // prev2 is the latest valley
                valley2 = valley1;
                valley1 = prev2;
            }
        }
        if (peak1 != null && peak2 != null
                && valley1 != null && valley2 != null
                && LT(peak2, peak1) && LT(valley2, valley1)) {
            accepted = true;
        } else {
            accepted = false;
        }
        return accepted;
    }

    public boolean step_up_trend_simple(PSymbolInstance input) throws SULException {
        traceLen++;
        boolean accepted;
        BigDecimal price = getPrice(input);
        prev3 = prev2;
        prev2 = prev1;
        prev1 = price;
        if (prev3 != null) {
            if (LT(prev1, prev2) && LT(prev3, prev2)) {
                // prev2 is the latest peak
                peak2 = peak1;
                peak1 = prev2;
            }
            if (peak1 != null && LT(prev2, prev1) && LT(prev2, prev3)) {
                // the first peak happens before the first valley
                // prev2 is the latest valley
                valley2 = valley1;
                valley1 = prev2;
            }
        }
        if (peak1 != null && peak2 != null
                && valley1 != null && valley2 != null
                && LT(peak2, peak1) && LT(valley2, valley1)) {
            accepted = true;
        } else {
            accepted = false;
        }
        return accepted;
    }

    public boolean step_interval(PSymbolInstance input) throws SULException {
        traceLen++;
        double price = getPrice(input).doubleValue();
        failed = failed || !(-1 <= price && price <= 1);
        return !failed;
    }

    public boolean step_alternating1(PSymbolInstance input, Comparator comp) throws SULException {
        traceLen++;
        if (failed)
            return false;
        BigDecimal price = getPrice(input);
        // language: (ab)*, a < b
        if (prev2 == null) {
            // prev2 is a
            prev2 = prev3 = price;
            return false;
        } else if (prev1 == null) {
            // prev1 is b
            prev1 = prev4 = price;
        } else {
            // prev4 is current
            // prev3 is last input
            prev3 = prev4;
            prev4 = price;
        }
        if (prev3 != null) {
            // need prev4 == prev1 && prev3 == prev2 && prev4 > prev3
            failed = !(prev4.equals(prev1) && prev3.equals(prev2));
            failed = failed || !comp.check(prev4, prev3);
            prev3 = prev4 = null;
            return !failed;
        }
        return false;
    }

    public boolean step_alternating2(PSymbolInstance input, Comparator comp) throws SULException {
        traceLen++;
        if (failed)
            return false;
        BigDecimal price = getPrice(input);
        prev2 = prev1;
        prev1 = price;
        if (prev2 != null) {
            // prev1 is current
            // prev2 is last input
            failed = !comp.check(prev1, prev2);
            prev1 = prev2 = null;
            return !failed;
        }
        return false;
    }

    public boolean step_lang1(PSymbolInstance input) throws SULException {
        traceLen++;
        if (failed) return false;
        BigDecimal price = getPrice(input);
        // Initialize the first price (a) if not already set
        if (prev1 == null) {
            prev1 = price;
            return true; // Accept the first input
        }

        // Check if the input matches the previously established price (a)
        if (!price.equals(prev1)) {
            failed = true; // Reject if a different symbol is encountered
            return false;
        }

        // Accept as long as the input matches prev1 (a)
        return true;
    }

    public boolean step_lang2(PSymbolInstance input) throws SULException {
        traceLen++;
        if (failed) return false;
        BigDecimal price = getPrice(input);
        if (prev1 == null) {
            prev1 = price;
            count1++;
        } else if (prev2 == null) {
            prev2 = price;
            if (price.equals(prev1)) {
                failed = true; // two same letters
                return false;
            }
            count2++;
        } else if (!price.equals(prev1) && !price.equals(prev2)) {
            failed = true; // more than 2 different letters
            return false;
        } else if (price.equals(prev1)) {
            count1++;
            if (count1 > count2 + 1) {
                failed = true; // more than one consecutive 'a'
                return false;
            }
        } else if (price.equals(prev2)) {
            count2++;
            if (count2 > count1) {
                failed = true; // more than one consecutive 'b'
                return false;
            }
        }
        return count1 == count2;
    }

    public boolean step_lang3(PSymbolInstance input) throws SULException {
        traceLen++;
        if (failed) return false;

        BigDecimal price = getPrice(input); // Get the price of the current symbol

        // Validate input symbols
        if (prev1 == null) {
            // First symbol
            prev1 = price;
        } else if (price.equals(prev1)) {
            // Continue the current sequence
            if (count2 % 2 == 1){
                count1++;
            }
            else{
                count1 = 0;
                count2 = 0;
            }
        } else if (prev2 == null) {
            // Second distinct symbol
            prev2 = price;
            count2++;
        } else if (price.equals(prev2)) {
            // Continue the second sequence
            count2++;
        } else {
            // More than two distinct symbols detected
            failed = true;
            return false;
        }

//        // Check the rule: No odd-length b's after odd-length a's
//        if (prev2 != null && prev1.equals(price)) {
//            // Shift sequence if needed
//            if ((count2 % 2 == 1) && (count1 % 2 == 1)) {
//                failed = true;
//                return false;
//            }
//        }

        // Continue with the new sequence
        return !(count2 % 2 == 1) && (count1 % 2 == 1);
    }

    public boolean step_lang4(PSymbolInstance input) throws SULException {
        traceLen++;
        if (failed) return false;

        BigDecimal price = getPrice(input); // Get the price of the current symbol

        if (prev1 == null) {
            // Initialize the first symbol
            prev1 = price;
            count1++;
        } else if (price.equals(prev1)) {
            // Increment the count for consecutive occurrences of the same symbol
            count1++;
            if (count1 == 3 && price.equals(prev1)) {
                // Fail if three consecutive 'a's are detected
                failed = true;
                return false;
            }
        } else {
            count1 = 0; // Reset the count for symbol a
        }

        return true; // No rule violations
    }





    public boolean step_lang5(PSymbolInstance input) throws SULException {
        traceLen++;
        if (failed) return false;
        BigDecimal price = getPrice(input);
        if (prev1 == null) {
            prev1 = price;
        } else if (prev2 == null) {
            prev2 = price;
        } else if (!price.equals(prev1) && !price.equals(prev2)) {
            failed = true; // more than 2 different letters
            return false;
        }
        if (price.equals(prev1)) {
            count1 = (count1 + 1) % 2;
        }
        if (price.equals(prev2) && !price.equals(prev1)) {
            count2 = (count2 + 1) % 2;
        }
        return (count1 == 0) && (count2 == 0);
    }


    public boolean step_lang6(PSymbolInstance input) throws SULException {
        traceLen++;
        if (failed) return false;
        BigDecimal price = getPrice(input);
        if (prev1 == null) {
            prev1 = price;
        } else if (prev2 == null) {
            prev2 = price;
        } else if (!price.equals(prev1) && !price.equals(prev2)) {
            failed = true; // more than 2 letters
            return false;
        }
        if (price.equals(prev1)) {
            count1 = (count1 + 1) % 3;
        }
        if (price.equals(prev2) && !price.equals(prev1)) {
            count2 = (count2 + 1) % 3;
        }
        return count1 == count2;
    }
}
