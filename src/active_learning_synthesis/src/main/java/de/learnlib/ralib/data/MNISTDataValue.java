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
package de.learnlib.ralib.data;

import java.math.BigDecimal;
import java.util.Arrays;

/**
 * Extension of DataValue for MNIST row data.
 * Stores both the integer row ID (as BigDecimal) and the original 28 pixel values (0-255).
 */
public class MNISTDataValue extends DataValue {

    private final int[] pixels;  // 28 original pixel values (0-255)

    /**
     * Creates an MNISTDataValue from a row ID and original pixel values.
     */
    public MNISTDataValue(DataType type, BigDecimal rowId, int[] pixels) {
        super(type, rowId);
        this.pixels = pixels.clone();
    }

    /**
     * Creates an MNISTDataValue from original pixel values.
     * The row ID is computed by abstracting pixels (>127 → 1, else 0) and encoding as 28-bit integer.
     */
    public MNISTDataValue(DataType type, int[] pixels) {
        super(type, new BigDecimal(pixelsToRowId(pixels)));
        this.pixels = pixels.clone();
    }

    /**
     * Returns the original 28 pixel values (0-255) for this row.
     */
    public int[] getPixels() {
        return pixels.clone();
    }

    /**
     * Returns the abstracted binary pixels (0.0f or 1.0f) based on threshold > 127.
     */
    public float[] getAbstractedPixels() {
        float[] abstracted = new float[28];
        for (int j = 0; j < 28; j++) {
            abstracted[j] = pixels[j] > 127 ? 1.0f : 0.0f;
        }
        return abstracted;
    }

    /**
     * Returns the row ID as a long.
     */
    public long getRowId() {
        return getValue().longValue();
    }

    /**
     * Returns the binary string representation of the row (abstracted).
     */
    public String toBinaryString() {
        StringBuilder sb = new StringBuilder(28);
        for (int p : pixels) {
            sb.append(p > 127 ? '1' : '0');
        }
        return sb.toString();
    }

    /**
     * Converts original pixel values to an abstracted row ID (28-bit integer).
     */
    private static long pixelsToRowId(int[] pixels) {
        long rowId = 0;
        for (int j = 0; j < 28; j++) {
            if (pixels[j] > 127) {
                rowId |= (1L << (27 - j));
            }
        }
        return rowId;
    }

    /**
     * Factory method to create from original pixel values.
     */
    public static MNISTDataValue fromPixels(int[] pixels, DataType type) {
        return new MNISTDataValue(type, pixels);
    }

    /**
     * Factory method to create from a row ID (no original pixels available).
     * Pixels will be set to 0 or 255 based on the bit value.
     */
    public static MNISTDataValue fromRowId(long rowId, DataType type) {
        int[] pixels = new int[28];
        for (int j = 0; j < 28; j++) {
            pixels[j] = ((rowId >> (27 - j)) & 1) == 1 ? 255 : 0;
        }
        return new MNISTDataValue(type, new BigDecimal(rowId), pixels);
    }

    @Override
    public String toString() {
        return getValue().toString() + "[" + getDataType().getName() + "]";
    }

    @Override
    public int hashCode() {
        // Use only parent's hashCode (based on BigDecimal value) for consistency with theory
        return super.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        // Use only parent's equals (based on BigDecimal value) for consistency with RALib theory
        // This ensures fresh values and original values with same rowId are considered equal
        return super.equals(obj);
    }
}
