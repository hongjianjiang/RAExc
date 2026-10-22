from typing import Dict, Set, Tuple, List, Union
from dataclasses import dataclass
from enum import Enum
from z3 import Solver, Real, And, Or, Not, sat
from graphviz import Digraph
from local_serach_synthesis.utils import *
from itertools import product
import cProfile
import re
import math
from itertools import chain, combinations
import json
import re
from copy import deepcopy

class Operator(Enum):
    EQ = 1
    NEQ = 2
    TT = 3
    FF = 4
    
class DoubleOperator(Enum):
    EQ = 1
    NEQ = 2
    LT = 3
    LEQ = 4


@dataclass(frozen=True)
class ExecutionTrace:
    source_states: Tuple[int, ...]
    final_state: int | None
    completed: bool
    


# LB OP UB OP

def createMap(num_registers: int, constants: List[int], theory = "Integer"):
    """
    Create a map from integers to tuples representing conditions of the form:
    (lowerbound, op1, upperbound, op2), excluding duplicates where bounds and operators are identical.

    Args:
        num_registers (int): Number of registers (r0 to r(num_registers-1)).
        constants (Set[int]): Set of constants (c0, c1, ...).

    Returns:
        Dict[int, Tuple[str, Operator, str, Operator]]: A map from integers to condition tuples.
    """
    # Combine registers and constants into a single list of bounds
    
    if theory == "Integer":
        bounds = [f"r{i}" for i in range(num_registers)] + [f"c{i}" for i in range(len(constants))]
        operator_functions = [
            Operator.EQ,
            Operator.NEQ,
            Operator.TT,
            
        ]
        condition_map = {}
        index = 0

        # Iterate over all combinations of lower bound, upper bound, and operators
        for lower in bounds:
            for op1 in operator_functions[:-1]:
                condition_map[index] = (lower, op1)
                index += 1
        condition_map[index] = (None, Operator.TT)
        index += 1
        condition_map[index] = (None, Operator.FF)
        return condition_map

    else:
        bounds = [f"r{i}" for i in range(num_registers)] + [f"c{i}" for i in range(len(constants))]
        operators = [DoubleOperator.LT, DoubleOperator.LEQ]
        operator_functions = {
            DoubleOperator.LT: lambda x, y: x < y,
            DoubleOperator.LEQ: lambda x, y: x <= y,
        }
        constants_list = list(constants)  # Convert set to list for indexing
        condition_map = {}
        index = 0

        # Iterate over all combinations of lower bound, upper bound, and operators
        for lower in bounds + ["nop"]:
            for op1 in operator_functions.keys():
                for upper in bounds + ["nop"]:
                    for op2 in operator_functions.keys():
                        # Skip cases where the lower and upper bounds and their operators are identical
                        if lower == upper and (op1 != DoubleOperator.LEQ or op2 != DoubleOperator.LEQ) and lower != "nop":
                            continue
                        if lower.startswith("c") and upper.startswith("c"):
                            lower_idx = int(lower[1:])
                            upper_idx = int(upper[1:])
                            lower_val = constants_list[lower_idx]
                            upper_val = constants_list[upper_idx]
                            if not operator_functions[op1](lower_val, upper_val):
                                continue
                        if (lower == "nop" and op1 != DoubleOperator.LT ) or (upper == "nop" and op2 != DoubleOperator.LT):
                            continue
                        condition_map[index] = (lower, op1, upper, op2)
                        index += 1
        return condition_map

  
# def createRegisterAssignmentMap(num_registers: int, constants: List[float]) -> Dict[int, List[Tuple[str, str]]]:
#     """
#     Create a map of register assignments, systematically generating all possibilities.

#     Args:
#         num_registers (int): Number of registers.
#         constants (List[float]): List of constants.

#     Returns:
#         Dict[int, List[Tuple[str, str]]]: A mapping of state IDs to register assignments.
#     """
#     # Available values for assignment: registers, constants, and "curr"
#     assign = dict()
#     for i in range(num_registers):
#         assign[i] = f"r{i}"
#     inx = 0
#     res_dict = dict()
#     constants = []
#     for i in range(num_registers):
#         values = [f"r{i}" for i in range(num_registers)] + [f"c{i}" for i in range(len(constants))] + ["curr"]
#         tmp = assign.copy()

#         for v in values:
#             tmp_list = []
#             tmp[i] = v
#             for k,v in tmp.items():
#                 tmp_list.append((f"r{k}", v))
#             if tmp_list not in res_dict.values():
#                 res_dict[inx] = tmp_list
#                 inx += 1

#     res_dict[inx] = [(f'r{i}', 'curr') for i in range(num_registers)]
#     print(res_dict)
#     return res_dict

def createRegisterAssignmentMap(
    num_registers: int,
    constants: List[float] | None = None,
    include_constants: bool = True,
) -> Dict[int, List[Tuple[str, str]]]:
    """
    Enumerate all n-tuples of register assignments.
    For each register r_i, choose a source from:
      - any register r_j,
      - 'curr',
      - constants (c0, c1, ...) if include_constants=True.

    Returns:
        Dict[int, List[Tuple[str, str]]]:
            { assignment_id : [("r0", src0), ("r1", src1), ..., ("r{n-1}", src{n-1})] }
    """
    constants = constants or []
    sources = [f"r{i}" for i in range(num_registers)] + ["curr"]
    if include_constants and len(constants) > 0:
        sources += [f"c{i}" for i in range(len(constants))]

    res: Dict[int, List[Tuple[str, str]]] = {}
    inx = 0
    for combo in product(sources, repeat=num_registers):
        res[inx] = [(f"r{i}", combo[i]) for i in range(num_registers)]
        inx += 1
    return res
             




class DRA:
    _BOUND_NONE = 0
    _BOUND_REGISTER = 1
    _BOUND_CONSTANT = 2

    _ASSIGN_CURRENT = 0
    _ASSIGN_REGISTER = 1
    _ASSIGN_CONSTANT = 2

    def __init__(
        self,
        states: Set[int],
        num_registers: int,
        constants: List[float],
        condition_map,
        assignment_map: Dict[int, List[Tuple[str, str]]],
        transitions: Dict[int, List[Tuple[int, int]]],
        register_transitions: Dict[int, List[Tuple[int, int]]],
        initial_state: int,
        final_states: Set[int],
        theory = "Integer"
    ):
        self.states = states
        self.num_registers = num_registers
        self.constants = constants
        self.condition_map = condition_map
        self.assignment_map = assignment_map
        self.transitions = transitions
        self.register_transitions = register_transitions
        self.initial_state = initial_state
        self.final_states = final_states
        self.registers = [0.0] * num_registers  # Initialize registers to 0.0
        self.theory = theory
        self.refresh_execution_cache()

    def refresh_execution_cache(self) -> None:
        """Compile guards and assignments once so scoring avoids repeated string parsing."""
        self._compiled_conditions = {
            condition_id: self._compile_condition(condition)
            for condition_id, condition in self.condition_map.items()
        }
        self._compiled_assignments = {
            assignment_id: self._compile_assignment(assignments)
            for assignment_id, assignments in self.assignment_map.items()
        }

    def _compile_bound(self, bound):
        if bound is None or bound == "nop":
            return (self._BOUND_NONE, None)
        if bound.startswith("r"):
            return (self._BOUND_REGISTER, int(bound[1:]))
        if bound.startswith("c"):
            return (self._BOUND_CONSTANT, self.constants[int(bound[1:])])
        raise ValueError(f"Unknown bound: {bound}")

    def _compile_condition(self, condition):
        if self.theory == "Integer":
            lower, op1 = condition
            if op1 in (Operator.TT, Operator.FF):
                return (op1, self._BOUND_NONE, None)
            bound_kind, bound_value = self._compile_bound(lower)
            return (op1, bound_kind, bound_value)

        lower, op1, upper, op2 = condition
        lower_kind, lower_value = self._compile_bound(lower)
        upper_kind, upper_value = self._compile_bound(upper)
        return (lower_kind, lower_value, op1, upper_kind, upper_value, op2)

    def _compile_assignment(self, assignments):
        compiled = []
        for lhs, rhs in assignments:
            lhs_index = int(lhs[1:])
            if rhs == "curr":
                compiled.append((lhs_index, self._ASSIGN_CURRENT, None))
            elif rhs.startswith("r"):
                compiled.append((lhs_index, self._ASSIGN_REGISTER, int(rhs[1:])))
            elif rhs.startswith("c"):
                compiled.append((lhs_index, self._ASSIGN_CONSTANT, self.constants[int(rhs[1:])]))
            else:
                raise ValueError(f"Unknown RHS: {rhs}")
        return tuple(compiled)

    def _resolve_compiled_bound(self, bound_kind: int, bound_value, registers: List[float]):
        if bound_kind == self._BOUND_REGISTER:
            return registers[bound_value]
        if bound_kind == self._BOUND_CONSTANT:
            return bound_value
        return None

    def _compiled_condition_holds(self, condition_spec, curr: float, registers: List[float]) -> bool:
        if self.theory == "Integer":
            op1, bound_kind, bound_value = condition_spec
            if op1 == Operator.TT:
                return True
            if op1 == Operator.FF:
                return False
            lower_bound = self._resolve_compiled_bound(bound_kind, bound_value, registers)
            return curr == lower_bound if op1 == Operator.EQ else curr != lower_bound

        lower_kind, lower_value, op1, upper_kind, upper_value, op2 = condition_spec
        lower_satisfied = True
        if lower_kind != self._BOUND_NONE:
            lower_bound = self._resolve_compiled_bound(lower_kind, lower_value, registers)
            lower_satisfied = curr > lower_bound if op1 == DoubleOperator.LT else curr >= lower_bound

        upper_satisfied = True
        if upper_kind != self._BOUND_NONE:
            upper_bound = self._resolve_compiled_bound(upper_kind, upper_value, registers)
            upper_satisfied = curr < upper_bound if op2 == DoubleOperator.LT else curr <= upper_bound

        return lower_satisfied and upper_satisfied

    def _execute_compiled_assignment(self, assignment_ops, registers: List[float], curr: float) -> None:
        for lhs_index, source_kind, source_value in assignment_ops:
            if source_kind == self._ASSIGN_CURRENT:
                registers[lhs_index] = curr
            elif source_kind == self._ASSIGN_REGISTER:
                registers[lhs_index] = registers[source_value]
            else:
                registers[lhs_index] = source_value


    def to_json(self, filename: str = "dra.json", theory = "Integer") -> None:
        """
        Save the DRA object to a JSON file, converting numpy types to native Python types.

        Args:
            filename (str): The name of the file to save the JSON to.
        """
        import numpy as np

        def convert(obj):
            if isinstance(obj, dict):
                return {convert(k): convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, tuple):
                return tuple(convert(i) for i in obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        dra_dict = {
            "states": list(self.states),
            "num_registers": self.num_registers,
            "constants": self.constants,
            "initial_state": self.initial_state,
            "final_states": list(self.final_states),
            "transitions": {
                str(state): transitions for state, transitions in self.transitions.items()
            },
            "register_transitions": {
                str(state): assignments for state, assignments in self.register_transitions.items()
            },
            "condition_map":  {
                str(cid): [cond[0], cond[1].name]
                for cid, cond in self.condition_map.items()
            }
            if theory == 'Integer' else 
            {
                str(cid): [cond[0], cond[1].name, cond[2], cond[3].name]
                for cid, cond in self.condition_map.items()
            }, 
            "assignment_map": {
                str(aid): assignments for aid, assignments in self.assignment_map.items()
            },
        }

        dra_dict = convert(dra_dict)
        with open(filename, "w") as file:
            json.dump(dra_dict, file, indent=4)
        # print(f"DRA saved to {filename}")

    def _decode_assignment(self, assignment_id: int) -> str:
        """
        Decode a list of register assignments from the assignment_map to a human-readable string.

        Args:
            assignment_id (int): The ID of the assignment to decode.

        Returns:
            str: A human-readable representation of the register assignments.
        """
        assignments = self.assignment_map[assignment_id]
        decoded_assignments = [f"{lhs} := {rhs}" for lhs, rhs in assignments]
        return "; ".join(decoded_assignments)

    def _execute_assignment(self, assignment_id: int, curr: float):
        """
        Execute a list of register assignments identified by `assignment_id`.

        Args:
            assignment_id (int): The ID of the assignments to execute.
            curr (float): The current input value.
        """
        # Decode the list of assignments
        assignments = self.assignment_map[assignment_id]

        # Helper function to resolve the RHS value
        def resolve_rhs(rhs: str) -> float:
            if rhs == "curr":
                return curr
            elif rhs.startswith("c"):
                constant_index = int(rhs[1:])
                if 0 <= constant_index < len(self.constants):
                    return self.constants[constant_index]
                raise ValueError(f"Constant index out of range: {rhs}")
            elif rhs.startswith("r"):
                register_index = int(rhs[1:])
                if 0 <= register_index < len(self.registers):
                    return self.registers[register_index]
                raise ValueError(f"Register index out of range: {rhs}")
            else:
                raise ValueError(f"Unknown RHS: {rhs}")

        # Execute all assignments in the list
        for lhs, rhs in assignments:
            lhs_index = int(lhs[1:])  # Extract register index from 'rX'
            if 0 <= lhs_index < len(self.registers):
                self.registers[lhs_index] = resolve_rhs(rhs)
            else:
                raise ValueError(f"Register index out of range for LHS: {lhs}")

    def __str__(self) -> str:
        """
        Generate a human-readable string representation of the automaton.
        """
        result = []
        result.append("Deterministic Register Automaton:")
        result.append(f"States: {self.states}")
        result.append(f"Initial State: {self.initial_state}")
        result.append(f"Final States: {self.final_states}")
        result.append(f"Number of Registers: {self.num_registers}")
        result.append(f"Constants: {self.constants}\n")

        # Decode and display transitions
        result.append("Transitions:")
        for state, transitions in self.transitions.items():
            for condition_id, next_state in transitions:
                condition = self._decode_condition(condition_id)
                result.append(f"  {state} --[{condition}]--> {next_state}")

        # Decode and display register transitions
        result.append("\nRegister Assignments:")
        for state, assignments in self.register_transitions.items():
            for assignment_id, next_state in assignments:
                assignment = self._decode_assignment(assignment_id)
                result.append(f"  {state} --[{assignment}]--> {next_state}")

        return "\n".join(result)
    
    def translate_to_initial_zero(self):
        """
        Translate the DRA to an equivalent one where the initial state is named 0, preserving all semantics and score.
        Only remap if the initial state is not already 0.
        Returns:
            DRA: A new DRA instance with the initial state renamed to 0.
        """
        old_initial = self.initial_state
        if old_initial == 0:
            # Already canonical, return a deepcopy
            return deepcopy(self)

        # Create a bijective mapping: initial state -> 0, others shifted
        sorted_states = sorted(self.states)
        mapping = {}
        next_id = 1
        mapping[old_initial] = 0
        for s in sorted_states:
            if s == old_initial:
                continue
            mapping[s] = next_id
            next_id += 1

        def remap_transitions(transitions):
            return {
                mapping[state]: [
                    (condition_id, mapping[next_state])
                    for condition_id, next_state in trans_list
                ]
                for state, trans_list in transitions.items()
            }

        new_transitions = remap_transitions(self.transitions)
        new_register_transitions = remap_transitions(self.register_transitions)
        new_states = {mapping[state] for state in self.states}
        new_final_states = {mapping[state] for state in self.final_states}

        return DRA(
            states=new_states,
            num_registers=self.num_registers,
            constants=self.constants,
            condition_map=deepcopy(self.condition_map),
            assignment_map=deepcopy(self.assignment_map),
            transitions=new_transitions,
            register_transitions=new_register_transitions,
            initial_state=0,
            final_states=new_final_states,
            theory=self.theory
        )


    def _decode_condition(self, condition_id: int) -> str:
        """
        Decode a condition from the condition_map to a human-readable string.

        Args:
            condition_id (int): The ID of the condition to decode.

        Returns:
            str: A human-readable representation of the condition.
        """
        if self.theory != "Integer":
            lower, op1, upper, op2 = self.condition_map[condition_id]

            # Translate operator enums to mathematical symbols
            operator_map = {
                DoubleOperator.LT: "<",
                DoubleOperator.LEQ: "<=",
            }

            def resolve_bound(bound):
                """
                Resolve the value of a bound (register or constant).
                """
                if bound.startswith("r"):
                    return bound  # Register remains as "r0", "r1", etc.
                elif bound.startswith("c"):
                    constant_index = int(bound[1:])  # Extract constant index (e.g., "c0" -> 0)
                    return str(self.constants[constant_index])  # Replace with constant value
                return "curr"  # "curr" remains as is

            # Adjust "nop" to display as "True"
            if lower == "nop" and upper == "nop":
                return "True"
            elif lower == "nop":
                return f"curr {operator_map[op2]} {resolve_bound(upper)}"
            elif upper == "nop":
                return f"{resolve_bound(lower)} {operator_map[op1]} curr"
            else:
                return f"{resolve_bound(lower)} {operator_map[op1]} curr {operator_map[op2]} {resolve_bound(upper)}"

        else:


            lower, op1 = self.condition_map[condition_id]

            # Translate operator enums to mathematical symbols
            operator_map = {
                Operator.EQ: "==",
                Operator.NEQ: "!=",
                Operator.TT:"True",
                Operator.FF:"False"
            }

            def resolve_bound(bound):
                """
                Resolve the value of a bound (register or constant).
                """
                if bound.startswith("r"):
                    return bound  # Register remains as "r0", "r1", etc.
                elif bound.startswith("c"):
                    constant_index = int(bound[1:])  # Extract constant index (e.g., "c0" -> 0)
                    return str(self.constants[constant_index])  # Replace with constant value
                return "curr"  # "curr" remains as is

            # Adjust "nop" to display as "True"
            if op1 == Operator.TT:
                return "True"
            elif op1 == Operator.FF:
                return "False"
            else:
                return f"{resolve_bound(lower)} {operator_map[op1]} curr"




    def _accepts_input_slow(self, word: List[float], debug: bool = False) -> bool:
        """
        Check if the given word is accepted by the automaton.

        Args:
            word (List[float]): A sequence of floats as input.
            debug (bool): If True, prints debugging information.

            Returns:
                bool: True if the word is accepted, False otherwise.
        """
        self.registers = [0.0] * self.num_registers  # Reset registers
        current_state = self.initial_state

        if debug:
            print(f"Initial state: {current_state}")
            print(f"Initial registers: {self.registers}")
            print(f"Input word: {word}")
        
        for curr in word:
            transition_found = False

            # Process state transitions
            if current_state in self.transitions:
                state_register_transitions = self.register_transitions.get(current_state, [])
                for idx, (condition_id, next_state) in enumerate(self.transitions[current_state]):
                    if self._evaluate_condition(condition_id, curr):
                        # Execute the register assignment at the SAME position
                        # (transitions/register_transitions are positionally
                        # synchronized; matching by next_state instead is
                        # ambiguous when two transitions share a target).
                        if idx < len(state_register_transitions):
                            assignment_id, _ = state_register_transitions[idx]
                            self._execute_assignment(assignment_id, curr)
                            if debug:
                                print(f"Register Assignment: {assignment_id} - {self._decode_assignment(assignment_id)}")
                                print(f"Registers updated: {self.registers}")

                        current_state = next_state
                        transition_found = True
                        if debug:
                            print(f"Transition: {condition_id} - {self._decode_condition(condition_id)}")
                            print(f"Moved to state: {current_state}")
                        break

            # If no valid transition is found, reject the word
            if not transition_found:
                if debug:
                    print(f"No valid transition found for current: {curr}")
                return False

        # Check if the final state is accepting
        is_accepted = current_state in self.final_states
        if debug:
            print(f"Final state: {current_state}")
            print(f"Accepted: {is_accepted}")
        return is_accepted

    def evaluate_input_trace(self, word: List[float]) -> ExecutionTrace:
        registers = [0.0] * self.num_registers
        current_state = self.initial_state
        transitions = self.transitions
        register_transitions = self.register_transitions
        compiled_conditions = self._compiled_conditions
        compiled_assignments = self._compiled_assignments
        source_states = []

        for symbol in word:
            curr = float(symbol)
            source_states.append(current_state)
            state_transitions = transitions.get(current_state)
            if not state_transitions:
                self.registers = registers
                return ExecutionTrace(tuple(source_states), None, False)

            state_register_transitions = register_transitions.get(current_state, ())
            transition_found = False

            # transitions[state] and register_transitions[state] are
            # positionally synchronized lists (the i-th guard pairs with the
            # i-th register assignment) -- matching by "same next_state"
            # instead of by index is ambiguous whenever a state has two
            # transitions to the same target via different guards (a
            # structure mutate_transition legitimately produces), since it
            # would use whichever same-target assignment happens to come
            # first regardless of which guard actually fired.
            for idx, (condition_id, next_state) in enumerate(state_transitions):
                if not self._compiled_condition_holds(compiled_conditions[condition_id], curr, registers):
                    continue

                if idx < len(state_register_transitions):
                    assignment_id, _ = state_register_transitions[idx]
                    self._execute_compiled_assignment(compiled_assignments[assignment_id], registers, curr)

                current_state = next_state
                transition_found = True
                break

            if not transition_found:
                self.registers = registers
                return ExecutionTrace(tuple(source_states), None, False)

        self.registers = registers
        return ExecutionTrace(tuple(source_states), current_state, True)

    def _accepts_input_fast(self, word: List[float]) -> bool:
        trace = self.evaluate_input_trace(word)
        return trace.completed and trace.final_state in self.final_states

    def accepts_input(self, word: List[float], debug: bool = False) -> bool:
        if debug:
            return self._accepts_input_slow(word, debug=True)
        return self._accepts_input_fast(word)

    def _evaluate_condition(self, condition_id: int, curr: float, debug: bool = False) -> bool:
        """
        Evaluate if the given `curr` satisfies the condition identified by `condition_id`.

        Args:
            condition_id (int): The ID of the condition to evaluate.
            curr (float): The current input value.
            debug (bool): If True, prints debugging information.

        Returns:
            bool: True if the condition is satisfied, False otherwise.
        """

        if self.theory == 'Integer':
            lower, op1 = self.condition_map[condition_id]
            def resolve_bound(bound: str) -> float:
                if bound:
                    if bound.startswith("c"):
                        return self.constants[int(bound[1:])]
                    elif bound.startswith("r"):
                        return self.registers[int(bound[1:])]
                    else:
                        return curr
                else:
                    return curr
                
            curr = float(curr)
            lower_bound = resolve_bound(lower)
    
            # Evaluate conditions
            lower_satisfied = True if (curr == lower_bound if op1 == Operator.EQ else curr != lower_bound) else False
            if op1 == Operator.TT:
                lower_satisfied = True
                
            elif op1 == Operator.FF:
                lower_satisfied = False
            # if debug:
            # print(f"Evaluating condition {condition_id}:")
            # print(f"  Condition: {self._decode_condition(condition_id)}")
            # print(f"  Current value: {curr}")
            # print(f"  Resolved lower bound: {lower} ({lower_bound})")
            # # print(f"  Resolved upper bound: {upper} ({upper_bound})")
            # print(f"  Lower satisfied: {lower_satisfied}")
            # # print(f"  Upper satisfied: {upper_satisfied}")
            # # print(f"  Result: {lower_satisfied and upper_satisfied}")
            return lower_satisfied

        else:
            lower, op1, upper, op2 = self.condition_map[condition_id]

            def resolve_bound(bound: str) -> float:
                if bound == "nop":
                    return float("-inf") if op1 == DoubleOperator.LT else float("inf")
                elif bound.startswith("c"):
                    return self.constants[int(bound[1:])]
                elif bound.startswith("r"):
                    return self.registers[int(bound[1:])]
                else:
                    return curr
                
            curr = float(curr)
            lower_bound = resolve_bound(lower)
            upper_bound = resolve_bound(upper)

            # Evaluate conditions
            lower_satisfied = True if lower == "nop" else (curr > lower_bound if op1 == DoubleOperator.LT else curr >= lower_bound)
            upper_satisfied = True if upper == "nop" else (curr < upper_bound if op2 == DoubleOperator.LT else curr <= upper_bound)

            if debug:
                print(f"Evaluating condition {condition_id}:")
                print(f"  Condition: {self._decode_condition(condition_id)}")
                print(f"  Current value: {curr}")
                print(f"  Resolved lower bound: {lower} ({lower_bound})")
                print(f"  Resolved upper bound: {upper} ({upper_bound})")
                print(f"  Lower satisfied: {lower_satisfied}")
                print(f"  Upper satisfied: {upper_satisfied}")
                print(f"  Result: {lower_satisfied and upper_satisfied}")

            return lower_satisfied and upper_satisfied

    
    def is_deterministic(self, debug: bool = False) -> bool:
        """
        Check if the DRA is deterministic by verifying the pairwise determinism
        of all transitions within each state.

        Args:
            debug (bool): If True, print debugging information.

        Returns:
            bool: True if the automaton is deterministic, False otherwise.
        """
        # Check condition-based transitions
        for state, state_transitions in self.transitions.items():
            if len(state_transitions) > 1:  # Only check determinism if there are multiple transitions
                transition_ids = [cond_id for cond_id, _ in state_transitions]

                # Use Z3-based check for determinism
                #print("output: ", not is_deterministic_z3(transition_ids, self.condition_map, self.constants, debug=debug))
                if not is_deterministic_z3(transition_ids, self.condition_map, self.constants, debug=debug):
                    if debug:
                        print(f"State {state} is not deterministic.")
                    return False
            elif debug:
                print(f"State {state} has only one transition and is deterministic by default.")


        if debug:
            print("The automaton is deterministic.")
        return True
    
    def visualize(self, filename: str = "dra_visualization") -> None:
        """
        Visualize the Deterministic Register Automaton (DRA) using Graphviz.

        Args:
            filename (str): The filename to save the visualization (default: "dra_visualization").
        """
        dot = Digraph()

        # Add states to the graph
        for state in self.states:
            if state in self.final_states:
                dot.node(str(state), shape="doublecircle")  # Final states are double circles
            else:
                dot.node(str(state), shape="circle")  # Regular states are circles

        # Add initial state marker
        dot.node("start", shape="none", label="")
        dot.edge("start", str(self.initial_state))

        # Prepare a combined dictionary for transitions and register transitions
        combined_transitions = {}

        for state, transitions in self.transitions.items():
            for condition_id, next_state in transitions:
                condition = self._decode_condition(condition_id)
                if (state, next_state) not in combined_transitions:
                    combined_transitions[(state, next_state)] = [condition, None]
                else:
                    combined_transitions[(state, next_state)][0] = condition

        for state, register_transitions in self.register_transitions.items():
            for assignment_id, next_state in register_transitions:
                assignment = self._decode_assignment(assignment_id)
                if (state, next_state) not in combined_transitions:
                    combined_transitions[(state, next_state)] = [None, assignment]
                else:
                    combined_transitions[(state, next_state)][1] = assignment

        # Add combined edges
        for (state, next_state), (condition, assignment) in combined_transitions.items():
            label = ""
            if condition:
                label += f"{condition}"
            if assignment:
                label += f"\nassig: {assignment}"
            dot.edge(str(state), str(next_state), label=label)

        # Render and save the visualization
        dot.render(f"output/{filename}.dot", format="png", cleanup=True)
        print(f"DRA visualization saved as output/{filename}.png")


    def correctness_check(self, pos:List[List[float]], neg:List[List[float]]) -> float:
        """
        Check the correctness of a DRA by scoring positive and negative samples.

        Args:
            dra (DRA): The DRA to evaluate.
            pos (List[List[float]]): List of positive samples.
            neg (List[List[float]]): List of negative samples.

        Returns:
            float: The correctness score of the DRA.
        """
        correct_pos = sum(self.accepts_input(sample) for sample in pos)
        correct_neg = sum(not self.accepts_input(sample) for sample in neg)
        total_samples = len(pos) + len(neg)
        return (correct_pos + correct_neg) / total_samples


def load_dra_from_json(file_path: str, constants) -> DRA:
    """
    Load a DRA from a JSON file and return an initialized DRA object.

    Args:
        file_path (str): Path to the JSON file containing the DRA data.

    Returns:
        DRA: Initialized DRA object.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)  # Load JSON data from a file

    # Follow the rest of your function implementation
    # Extracting states, transitions, etc., and initializing the DRA object

    # Dummy implementation for the continuation (use your actual data structure mapping)
    states = set(range(data['states']))
    num_registers = len(data['register_vars'])
    # print(data.keys())
    initial_state = 0  # Assuming
    final_states = set(data['final_states'])
    
    # Assuming placeholder logic for generating conditions and assignments
    condition_map = createMap(num_registers, constants)
    assignment_map = createRegisterAssignmentMap(num_registers, constants)

    tmp_map = dict()
    for i, transition in enumerate(data['transitions']):
        if "And" in transition['guard']:
            matches = re.findall(r'(>=|<=|>|<)\s*([^\s,)]+)', transition['guard'])
            for match in matches:
                operator, value = match
                if operator == ">":
                    op2 = Operator.LT
                    tmp_var1 = value
                elif operator == "<":
                    op1 = Operator.LT
                    tmp_var2 = value
                elif operator == "<=":
                    op1 = Operator.LEQ
                    tmp_var2 = value
                else:
                    op2 = Operator.LEQ
                    tmp_var1 = value
            tmp_map[i] = (tmp_var1, op1, tmp_var2, op2)
        elif "<" in transition['guard']:
            if '=' in transition['guard']:
                match = re.search(r'<=\s*(\S+)', transition['guard'])
                var = match.group(1) if match else None
                op1 = Operator.LT
                op2 = Operator.LEQ
                tmp_map[i] = ('nop', op1, var, op2)
            else:
                # greater_than = transition['guard'][transition['guard'].index('<')]
                match = re.search(r'<\s*(\S+)', transition['guard'])
                var = match.group(1) if match else None
                op1 = Operator.LT
                op2 = Operator.LT
                tmp_map[i] = ('nop', op1, var, op2)
        elif ">" in transition['guard']:
            if '=' in transition['guard']:
                match = re.search(r'>=\s*(\S+)', transition['guard'])
                var = match.group(1) if match else None
                op1 = Operator.LEQ
                op2 = Operator.LT
                tmp_map[i] = (var, op1, 'nop', op2)
            else:
                match = re.search(r'>\s*(\S+)', transition['guard'])
                var = match.group(1) if match else None
                op1 = Operator.LT
                op2 = Operator.LT
                tmp_map[i] = (var, op1, 'nop', op2)
        else:
            tmp_map[i] = ('nop', Operator.LT, 'nop', Operator.LT)
        # assignment_map[i] = list(zip(data['register_vars'], transition['assignment'])) 
    transitions = {}
    register_transitions = {}
    list1 = data['register_vars']

    for i, transition in enumerate(data['transitions']):
        start_state = transition['start_state']
        end_state = transition['end_state']
        
        if start_state not in transitions:
            transitions[start_state] = []
        for k, v in condition_map.items():
            # print(f"v: {v} tmp_map: {tmp_map[i]}")
            # print(f"comparing some things {v[0] == tmp_map[i][0]} {v[1] == tmp_map[i][1]} {v[2] == tmp_map[i][2]} {v[3] == tmp_map[i][3]}")
            if v == tmp_map[i]:
                transitions[start_state].append((k, end_state))
                break       

        if start_state not in register_transitions:
            register_transitions[start_state] = []
        list2 = transition["assignment"]
        tmp = []
        for i in range(len(list2)):
            tmp.append((list1[i], list2[i]))

        for k, v in assignment_map.items():
            if v == tmp:
                register_transitions[start_state].append((k, end_state))
                break


    dra = DRA(
        states=states,
        num_registers=num_registers,
        constants=constants,
        condition_map=condition_map,
        assignment_map=assignment_map,
        transitions=transitions,
        register_transitions=register_transitions,
        initial_state=initial_state,
        final_states=final_states
    )
    return dra

 

def translate_to_z3(
    condition_id: int,
    condition_map: Dict[int, Tuple[str, Operator, str, Operator]],
    constants: List[float]
):
    """
    Translate a condition to a Z3 formula.

    Args:
        condition_id (int): Condition ID to translate.
        condition_map (Dict[int, Tuple[str, Operator, str, Operator]]): Map of condition IDs to conditions.
        constants (List[float]): List of constants.

    Returns:
        Tuple[ExprRef, ExprRef]: Lower and upper bound Z3 formulas for the condition.
    """
    lower, op1, upper, op2 = condition_map[condition_id]

    # Resolve bounds
    def resolve(bound):
        if bound == "nop":
            return None
        elif bound.startswith("c"):
            return constants[int(bound[1:])]  # Resolve constant
        else:
            return Real(bound)  # Treat register as a Z3 variable

    lower_bound = resolve(lower)
    upper_bound = resolve(upper)

    # Translate bounds to Z3 formulas
    lower_formula = None if lower_bound is None else (
        lower_bound < Real("curr") if op1 == DoubleOperator.LT else lower_bound <= Real("curr")
    )
    upper_formula = None if upper_bound is None else (
        Real("curr") < upper_bound if op2 == DoubleOperator.LT else Real("curr") <= upper_bound
    )

    return lower_formula, upper_formula

 


def is_deterministic_z3(
    transitions: List[int],
    condition_map: Dict[int, Tuple[str, Operator, str, Operator]],
    constants: List[float],
    debug: bool = False,
    solver: "Solver" = None,
) -> bool:
    """
    Use Z3 to check if a set of transitions is deterministic.

    Args:
        transitions (List[int]): List of transition condition IDs.
        condition_map (Dict[int, Tuple[str, Operator, str, Operator]]): Map of condition IDs to conditions.
        constants (List[float]): List of constants.
        debug (bool): If True, print debugging information.
        solver (Solver, optional): A pre-built Solver to reuse via push/pop.
            Constructing a fresh z3.Solver() per call dominates the runtime
            of callers that invoke this pairwise over a large condition map
            (e.g. compute_pairwise_det_map_dra), so callers doing many checks
            back-to-back should pass one in and reuse it.

    Returns:
        bool: True if the transitions are deterministic, False otherwise.
    """
    if solver is None:
        solver = Solver()

    for i, cond_id1 in enumerate(transitions):
        for cond_id2 in transitions[i + 1:]:
            # Translate both conditions to Z3
            lower1, upper1 = translate_to_z3(cond_id1, condition_map, constants)
            lower2, upper2 = translate_to_z3(cond_id2, condition_map, constants)
            # Build Z3 formula for overlap
            overlap = [f for f in (lower1, upper1, lower2, upper2) if f is not None]

            # If overlap list is empty, skip
            if not overlap:
                continue

            # Combine conditions into a single overlap formula
            overlap_formula = And(*overlap)

            solver.push()
            solver.add(overlap_formula)

            # Check for satisfiability
            sat_result = solver.check()

            if debug:
                print(f"Checking overlap between conditions {cond_id1} and {cond_id2}:")
                print(f"Formula: {overlap_formula} {sat_result}")

            if sat_result == sat:
                if debug:
                    print(f"SAT: Conditions {cond_id1} and {cond_id2} overlap. {sat_result} {solver.model()}")
                solver.pop()
                return False  # Found an overlap; transitions are not deterministic
            solver.pop()
    return True  # All pairs are deterministic

    

if __name__ == '__main__':
    dra = load_dra_from_json("L2_dra_exact.json",  []) 
    dra.visualize("123")
    # print(dra.condition_map)
    # dra.to_json("new.json")
    # print(dra)
    # dra.visualize("testing")
    print(dra.accepts_input([2.2,6]))
  
