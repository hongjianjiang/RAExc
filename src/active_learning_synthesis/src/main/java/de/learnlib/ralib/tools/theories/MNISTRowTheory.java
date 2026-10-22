package de.learnlib.ralib.tools.theories;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import de.learnlib.ralib.data.DataType;
import de.learnlib.ralib.data.DataValue;
import de.learnlib.ralib.data.MNISTDataValue;
import de.learnlib.ralib.oracles.io.IOOracle;
import de.learnlib.ralib.theory.equality.EqualityTheory;
import de.learnlib.ralib.tools.classanalyzer.TypedTheory;

public class MNISTRowTheory extends EqualityTheory implements TypedTheory {
    private DataType type;

    public MNISTRowTheory(DataType type) {
        this.type = type;
    }

    public MNISTRowTheory() {
    }
    
    /**
     * Decoding: Converts a BigDecimal ID back into a 28-pixel row of integers (0-255).
     * Used for Symbol Abstraction.
     */
    public static int[] decodeRowToInts(BigDecimal val) {
        BigInteger bi = val.toBigInteger();
        int[] row = new int[28];
        for (int i = 0; i < 28; i++) {
            // Map bits back to 0 or 255
            row[i] = bi.testBit(27 - i) ? 255 : 0; 
        }
        return row;
    }

    /**
     * Encoding: Converts 28 pixels (0-255) into a single BigDecimal symbol.
     * Pixels > 127 are treated as 1, else 0.
     * Used by the Simulator when loading the CSV.
     */
    public static BigDecimal encodeRow(int[] pixelRow) {
        BigInteger bi = BigInteger.ZERO;
        for (int i = 0; i < 28; i++) {
            if (pixelRow[i] > 127) {
                bi = bi.setBit(27 - i);
            }
        }
        return new BigDecimal(bi);
    }

    /**
     * Creates an MNISTDataValue from a row of 28 original pixel values (0-255).
     */
    public MNISTDataValue createMNISTDataValue(int[] pixels) {
        return MNISTDataValue.fromPixels(pixels, type);
    }

    /**
     * Creates an MNISTDataValue from an abstracted row ID.
     */
    public MNISTDataValue createMNISTDataValue(long rowId) {
        return MNISTDataValue.fromRowId(rowId, type);
    }

    @Override
    public DataValue getFreshValue(List<DataValue> vals) {
        // Find max row ID and return next one to ensure uniqueness
        long maxRowId = -1;
        for (DataValue d : vals) {
            long rowId = d.getValue().longValue();
            if (rowId > maxRowId) {
                maxRowId = rowId;
            }
        }
        // Return next row ID (guaranteed unique)
        return MNISTDataValue.fromRowId(maxRowId + 1, type);
    }

    @Override
    public Collection<DataValue> getAllNextValues(List<DataValue> vals) {
        List<DataValue> ret = new ArrayList<>();
        for (DataValue dv : vals) {
            if (dv instanceof MNISTDataValue) {
                ret.add(dv);
            } else {
                // Convert regular DataValue to MNISTDataValue
                ret.add(MNISTDataValue.fromRowId(dv.getValue().longValue(), type));
            }
        }
        ret.add(getFreshValue(vals));
        return ret;
    }

    @Override
    public void setType(DataType type) {
        this.type = type;
    }

    @Override
    public void setUseSuffixOpt(boolean useit) {
        this.useNonFreeOptimization = useit;
    }

    @Override
    public void setCheckForFreshOutputs(boolean doit, IOOracle oracle) {
        super.setFreshValues(doit, oracle);
    }
}