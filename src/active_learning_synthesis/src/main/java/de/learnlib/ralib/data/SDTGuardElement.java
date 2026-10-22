package de.learnlib.ralib.data;

import java.math.BigDecimal;

import gov.nasa.jpf.constraints.api.Expression;

public interface SDTGuardElement extends TypedValue {

    static boolean isConstant(SDTGuardElement e) {
        return e.getClass().equals(SymbolicDataValue.Constant.class);
    }

    static boolean isDataValue(SDTGuardElement e) {
        // Check if it's a DataValue but not a SymbolicDataValue (which includes Constant, SuffixValue, Register)
        return e instanceof DataValue && !(e instanceof SymbolicDataValue);
    }

    static boolean isSuffixValue(SDTGuardElement e) {
        return e.getClass().equals(SymbolicDataValue.SuffixValue.class);
    }

    static boolean isRegister(SDTGuardElement e) {
    	return e.getClass().equals(SymbolicDataValue.Register.class);
    }

    Expression<BigDecimal> asExpression();

}
