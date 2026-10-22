from typing import List, Callable, Any, Tuple
import z3
from register_automata.ra import RegisterAutomata
from register_automata.ra import Transition
from math import inf
from z3 import Real
from graphviz import Source
import re
import heapq
from fractions import Fraction


def float_to_z3_real(val):
    """Convert a float to a z3 RatVal to avoid parser errors with precise floats."""
    if isinstance(val, float):
        frac = Fraction(val).limit_denominator(10**12)
        return z3.RatVal(frac.numerator, frac.denominator)
    elif isinstance(val, int):
        return z3.IntVal(val)
    else:
        return val

class RAA_Transition(Transition):
    """
    Represents one transition of the RAA.
    """

    def __init__(
        self,
        cv1,
        cv2,
        start_state,
        guard,
        assignment,
        end_state,
        cost_function,
        move: int,
    ):
        """
        guard(registers, accumulator, input_heads) -> bool
        register_assignment(registers) -> modifies registers in-place
        accumulator_update(registers, old_acc) -> new_acc
        move in {'1', '2', 'both'}
        """
        super().__init__(cv1, start_state, guard, assignment, end_state)
        self.cv1 = cv1
        self.cv2 = cv2
        self.cost_function = cost_function
        self.move = move

    def is_enabled(
        self, start_state, symbol1, symbol2, registers, register_vars, solver
    ):


        if self.start_state != start_state:
            return False
        solver.push()
        solver.add(self.cv1 == float_to_z3_real(symbol1))
        solver.add(self.cv2 == float_to_z3_real(symbol2))
        solver.add(self.guard)
        for i in register_vars:
            z3_val = float_to_z3_real(registers[str(i)])
            solver.add(i == z3_val)
        res = solver.check()
        # print("\n -------- result ", res, "\n--------\n")
        solver.pop()
        if res == z3.sat:
            return True
        else:
            return False
        
    def is_enabled_fixed_curr1(
        self, start_state, C, symbol1, registers, register_vars, solver
    ):
        """
        Check if transition is enabled when curr1 can be any value from set C.
        
        Parameters:
        ----------
        start_state : int
            The starting state to check
        C : set
            Set of possible values for curr1
        symbol1 : Any
            The symbol for curr1/head1
        registers : dict
            Current register values
        register_vars : list
            Z3 variables representing registers
        solver : z3.Solver
            Z3 solver instance
            
        Returns:
        -------
        bool
            True if transition is enabled for any curr1 in C, False otherwise
        """
        if self.start_state != start_state:
            return False
        
        solver.push()
        
        # Create a disjunction over all possible values for curr1 in C
        # curr1_constraints = []
        # for c in C:
            # curr1_constraints.append(self.cv1 == c)
        
        # if curr1_constraints:
            # Add the constraint that curr1 must be one of the values in C
            # solver.add(z3.Or(*curr1_constraints))
        # else:
            # If C is empty, no transition can be enabled
            # solver.pop()
            # return False
        
        # Add other constraints
        solver.add(self.cv1 == float_to_z3_real(symbol1))
        solver.add(self.guard)

        # Add register constraints - FIX: Extract register name properly
        for i in register_vars:
            # Extract the register name from the Z3 variable
            reg_name = str(i.decl().name())
            # Use the register name as the key for the registers dictionary
            if reg_name in registers:
                # Convert float to z3 Rational to avoid parser errors with precise floats
                z3_val = float_to_z3_real(registers[reg_name])
                solver.add(i == z3_val)

        # Check if constraints are satisfiable
        res = solver.check()
        value = None
        if res == z3.sat:
            model = solver.model()
            value = model[self.cv1]
        solver.pop()
        return res == z3.sat, value

    def execute(self, symbol1, symbol2, registers):
        # Create a local context with individual register values
        local_context = registers
        local_context["curr1"] = symbol1  # Adding the first symbol to the context
        local_context["curr2"] = symbol2  # Adding the second symbol to the context
        for reg_index, assign_value in self.assignment.items(): 
            if isinstance(assign_value, str):
                if assign_value == "curr1" or assign_value == "c1":
                    registers[reg_index] = symbol1
                elif assign_value == "curr2" or assign_value == "c2":
                    registers[reg_index] = symbol2
                elif assign_value.startswith("r"):
                    # Register-to-register assignment (e.g., "r0" -> registers[0])
                    try:
                        registers[reg_index] = registers[assign_value]
                    except ValueError:
                        raise Exception(f"Invalid register reference: {assign_value}")
                else:
                    # Try to evaluate as a constant or expression
                    try:
                        # First try as a numeric constant
                        registers[reg_index] = float(assign_value)
                    except ValueError:
                        # Then try as an expression using the local context
                        try:
                            registers[reg_index] = eval(assign_value, {"__builtins__": None}, local_context)
                        except Exception as e:
                            raise Exception(f"Error evaluating assignment '{assign_value}': {e}")
            elif isinstance(assign_value, (int, float)):
                # Direct assignment of a numeric constant
                registers[reg_index] = assign_value
            elif isinstance(assign_value, z3.ExprRef):
                # Z3 expression - evaluate it in the local context
                try:
                    registers[reg_index] = eval(str(assign_value), {"__builtins__": None}, local_context)
                except Exception as e:
                    raise Exception(f"Error evaluating Z3 expression '{assign_value}': {e}")
            else:
                raise Exception(f"Unsupported assignment type: {type(assign_value)}")
            
    
            

    def __str__(self):
        return f"Transition: {self.start_state} -> {self.end_state}, Guard: {self.guard}, Assignment: {self.assignment}, ACC: {self.cost_function[0]} * curr1 + {self.cost_function[1]} * curr2 + {self.cost_function[2]}, Move: {self.move if self.move != 3 else 'both'}"


class Configuration:
    """
    Represents a configuration of the RAA.
    """
    def __init__(
        self,
        current_state: int,
        h1_tape: int,
        h2_tape: int,
        registers: List[int],
        accumulator: int,
    ):
        self.current_state = current_state
        self.registers = registers
        self.accumulator = accumulator
        self.h1_tape_inx = h1_tape
        self.h2_tape_inx = h2_tape

    def __str__(self):
        return f"State: {self.current_state}, Head1: {self.h1_tape_inx}, Head2: {self.h2_tape_inx}, Registers: {self.registers}, Accumulator: {self.accumulator}"


class RAA(RegisterAutomata):
    def __init__(
        self,
        states: List[int],
        initial_state: int,
        final_states: List[int],
        transitions: List[RAA_Transition],
        register_vars: List[int],
    ):
        self.states = states
        self.initial_state = initial_state
        self.final_states = final_states
        self.transitions = transitions
        self.register_vars = register_vars
        self.solver = z3.Solver()
        self.acc = 0
        self.head_positions = [0, 0]
        self.registers = {str(r) : 0 for r in register_vars}

    def product(self, ra):
        product_states = [
            (q_a, q_b)
            for q_a in range(len(self.states))
            for q_b in range(ra.states)
        ]

        final_states = [
            product_states.index((q_a, q_b))
            for q_a in self.final_states
            for q_b in ra.final_states
        ]

        # Only curr -> curr1 substitution (Real version)
        subs = [(z3.Real("curr"), z3.Real("curr1"))]

        transitions = []
        for ta in self.transitions:
            for tb in ra.transitions:
                # Start with ta assignments as-is
                assign = dict(ta.assignment)

                # For tb assignments, substitute "curr" -> "curr1" in values
                for k, v in tb.assignment.items():
                    if isinstance(v, z3.ExprRef):
                        v_sub = z3.substitute(v, *subs)
                    elif isinstance(v, str) and v == "curr":
                        v_sub = "curr1"
                    else:
                        v_sub = v
                    assign[k] = v_sub  # No p-suffix!

                # Guards
                if isinstance(tb.guard, bool):
                    tb_guard_z3 = z3.BoolVal(tb.guard)
                else:
                    tb_guard_z3 = tb.guard
                tb_guard_sub = z3.substitute(tb_guard_z3, *subs)
                
                guard = z3.And(ta.guard, tb_guard_sub)
                t = RAA_Transition(
                    z3.Real("curr1"),
                    z3.Real("curr2"),
                    start_state=product_states.index((ta.start_state, tb.start_state)),
                    guard=guard,
                    assignment=assign,
                    end_state=product_states.index((ta.end_state, tb.end_state)),
                    cost_function=ta.cost_function,
                    move=ta.move,
                )
                transitions.append(t)

        initial_state_index = product_states.index((self.initial_state, 0))
        # for inx, state in enumerate(product_states):
            # print(inx, state)
        # for inx, i in enumerate(product_states):
            # print(inx, i)
        return RAA(
            states=product_states,
            initial_state=initial_state_index,
            final_states=final_states,
            transitions=transitions,
            register_vars=self.register_vars + ra.register_vars,
        )

    

    

    def vector_product(self, ra):
        # Cartesian product of states
        product_states = [
            (q_a, q_b)
            for q_a in range(len(self.states))
            for q_b in range(ra.states)
        ]

        # Final states: both components must be final
        final_states = [
            product_states.index((q_a, q_b))
            for q_a in self.final_states
            for q_b in ra.final_states
        ]

        # Only substitute: curr -> curr1 (as Real)
        subs = [(z3.Real("curr"), z3.Real("curr2"))]

        transitions = []
        for ta in self.transitions:
            for tb in ra.transitions:
                # Assignments: ta's as is; tb's with curr→curr1 substituted
                assign = dict(ta.assignment)
                for k, v in tb.assignment.items():
                    if isinstance(v, z3.ExprRef):
                        v_sub = z3.substitute(v, *subs)
                    elif isinstance(v, str) and v == "curr":
                        v_sub = "curr1"
                    else:
                        v_sub = v
                    assign[k] = v_sub

                # Guard: substitute curr→curr1 if needed
                if isinstance(tb.guard, z3.ExprRef):
                    tb_guard_z3 = tb.guard
                elif isinstance(tb.guard, bool):
                    tb_guard_z3 = z3.BoolVal(tb.guard)
                elif isinstance(tb.guard, str):
                    if tb.guard.lower() == "true":
                        tb_guard_z3 = z3.BoolVal(True)
                    elif tb.guard.lower() == "false":
                        tb_guard_z3 = z3.BoolVal(False)
                    else:
                        raise ValueError(f"Unsupported tb.guard string: {tb.guard}")
                else:
                    raise TypeError(f"tb.guard has unexpected type: {type(tb.guard)}")

                tb_guard_sub = z3.substitute(tb_guard_z3, *subs)
                guard = z3.And(ta.guard, tb_guard_sub)
                if tb.guard is False:
                    cost_function = ta.cost_function
                else:
                # Extracted number, cost function computation (as you had)
                    extracted_number = extract_number_from_guard(tb.guard)

                    tmp_a = ta.cost_function[0]
                    # tmp_b = 0
                    tmp_c = ta.cost_function[2] + extracted_number * ta.cost_function[1]
                    cost_function = (tmp_a, 0, tmp_c)
                    # print(cost_function)

                t = RAA_Transition(
                    z3.Real("curr1"),
                    z3.Real("curr2"),
                    start_state=product_states.index((ta.start_state, tb.start_state)),
                    guard=guard,
                    assignment=assign,
                    end_state=product_states.index((ta.end_state, tb.end_state)),
                    cost_function=cost_function,
                    move=ta.move,
                )
                transitions.append(t)

        initial_state_index = product_states.index((self.initial_state, 0))

        return RAA(
            product_states,
            initial_state=initial_state_index,
            final_states=final_states,
            transitions=transitions,
            register_vars=self.register_vars,  # or combine if needed
        )


    def run_all_paths(self, v, w):
        """
        Optimized shortest-path search over RAA configurations using Dijkstra’s algorithm.
        Returns the minimal accumulator among all accepting runs, or float('inf') if none.
        """
        # best[(state, h1, h2, regs_tuple)] = best accumulator seen
        best = {}

        # Pack the initial register values as a tuple
        start_regs = tuple(self.registers[r] for r in self.register_vars)
        start_key = (self.initial_state, 0, 0, start_regs)
        best[start_key] = 0

        # Min-heap ordered by accumulator cost
        heap = [(0, *start_key)]  # (acc, state, h1, h2, regs)

        while heap:
            acc, q, h1, h2, regs = heapq.heappop(heap)
            # If we've already found a strictly better way here, skip
            if best.get((q, h1, h2, regs), inf) < acc:
                continue

            # Check accepting condition
            if q in self.final_states and h1 == len(v) and h2 == len(w):
                return acc

            # Read current symbols (or None)
            c1 = v[h1] if 0 <= h1 < len(v) else None
            c2 = w[h2] if 0 <= h2 < len(w) else None

            # Reconstruct register dict from tuple
            regs_dict = dict(zip(self.register_vars, regs))

            # Explore outgoing transitions
            for t in self.transitions:
                if t.start_state != q:
                    continue
                if not t.is_enabled(q, c1, c2, regs_dict, self.register_vars, self.solver):
                    continue

                # Simulate transition
                new_regs = regs_dict.copy()
                t.execute(c1, c2, new_regs)
                new_regs_tuple = tuple(new_regs[r] for r in self.register_vars)

                # Compute new accumulator
                delta = 0
                if c1 is not None:
                    delta += c1 * t.cost_function[0]
                if c2 is not None:
                    delta += c2 * t.cost_function[1]
                delta += t.cost_function[2]
                new_acc = acc + delta

                # Advance heads
                nh1, nh2 = self._compute_new_heads(t.move, h1, h2)

                key = (t.end_state, nh1, nh2, new_regs_tuple)
                if new_acc < best.get(key, inf):
                    best[key] = new_acc
                    heapq.heappush(heap, (new_acc, *key))

        # No accepting run found
        return float('inf')


    def to_dot(self):
        dot = "digraph Automaton {\n"
        dot += 'init [label="", shape=point];\n'
        dot += 'init -> 0 [shape="solid"];\n'
        for transition in self.transitions:
            transition_assignments = []
            for k, v in transition.assignment.items():
                transition_assignments.append(k + ":=" + v)
            if (
                transition.cost_function[0] != 0
                and transition.cost_function[1] != 0
                and transition.cost_function[2] != 0
            ):
                dot += f"  {transition.start_state} -> {transition.end_state} [label=\"{(transition.guard)}, \\n acc+={transition.cost_function[0]}*c1+{transition.cost_function[1]}*c2+{transition.cost_function[2]},\\n{transition.move if transition.move != 3 else 'both'}\\n {'; '.join(transition_assignments)}\"];\n"
            elif (
                transition.cost_function[0] != 0
                and transition.cost_function[1] != 0
                and transition.cost_function[2] == 0
            ):
                dot += f"  {transition.start_state} -> {transition.end_state} [label=\"{(transition.guard)}, \\n acc+={transition.cost_function[0]}*c1+{transition.cost_function[1]}*c2,\\n{transition.move if transition.move != 3 else 'both'}\\n {'; '.join(transition_assignments)}\"];\n"
            elif (
                transition.cost_function[0] == 0
                and transition.cost_function[1] != 0
                and transition.cost_function[2] != 0
            ):
                dot += f"  {transition.start_state} -> {transition.end_state} [label=\"{(transition.guard)}, \\n acc+={transition.cost_function[1]}*c2+{transition.cost_function[2]},\\n{transition.move if transition.move != 3 else 'both'}\\n {'; '.join(transition_assignments)}\"];\n"
            elif (
                transition.cost_function[0] == 0
                and transition.cost_function[1] != 0
                and transition.cost_function[2] == 0
            ):
                dot += f"  {transition.start_state} -> {transition.end_state} [label=\"{(transition.guard)}, \\n acc+={transition.cost_function[1]}*c2,\\n{transition.move if transition.move != 3 else 'both'}\\n {'; '.join(transition_assignments)}\"];\n"
            elif (
                transition.cost_function[0] != 0
                and transition.cost_function[1] == 0
                and transition.cost_function[2] != 0
            ):
                dot += f"  {transition.start_state} -> {transition.end_state} [label=\"{(transition.guard)}, \\n acc+={transition.cost_function[0]}*c1+{transition.cost_function[2]},\\n{transition.move if transition.move != 3 else 'both'}\\n {'; '.join(transition_assignments)}\"];\n"
            elif (
                transition.cost_function[0] != 0
                and transition.cost_function[1] == 0
                and transition.cost_function[2] == 0
            ):
                dot += f"  {transition.start_state} -> {transition.end_state} [label=\"{(transition.guard)}, \\n acc+={transition.cost_function[0]}*c1,\\n{transition.move if transition.move != 3 else 'both'}\\n {'; '.join(transition_assignments)}\"];\n"
            else:
                dot += f"  {transition.start_state} -> {transition.end_state} [label=\"{(transition.guard)}, \\n acc+={transition.cost_function[2]},\\n{transition.move if transition.move != 3 else 'both'}\\n {'; '.join(transition_assignments)}\"];\n"

        for i in self.final_states:
            dot += "  " + str(i) + " [shape=doublecircle];\n"
        dot += "}\n"
        return dot

    def show(self, filename):
        dot = self.to_dot()
        s = Source(dot, "output/" + filename, format="png")
        s.view()

    def _compute_new_heads(self, move_command, current_h1, current_h2):
        """
        Compute new head positions based on move command and current positions.
        Returns the new positions without modifying any global state.
        """
        new_h1, new_h2 = current_h1, current_h2
        if move_command == 1:
            new_h1 += 1
        elif move_command == 2:
            new_h2 += 1
        elif move_command == 3:  # both
            new_h1 += 1
            new_h2 += 1
        else:
            raise ValueError(f"Unknown move command: {move_command}")
        return new_h1, new_h2
    
def extract_number_from_guard(guard_expr):
    """
    Extract the number from a guard expression, including rationals like 'a/b'.
    
    Parameters
    ----------
    guard_expr : str or z3.ExprRef
        The guard expression to extract the number from
        
    Returns
    -------
    float or None
        The extracted number as float, or None if not found
    """
    guard_str = str(guard_expr)
    
    # Match patterns like "== number", "== a/b", etc.
    matches = re.findall(r'==\s*([-+]?\d+(?:/\d+)?(?:\.\d+)?(?:/\d+)?(?:\.\d+)?)', guard_str)
    
    if matches:
        num_str = matches[-1].strip()
        if "/" in num_str:
            try:
                num, denom = num_str.split("/")
                return float(num) / float(denom)
            except Exception:
                pass
        else:
            try:
                return float(num_str)
            except Exception:
                pass

    return None

# if __name__ == "__main__":
    # def create_dtw_raa():
    #     from z3 import Real
    #     from register_automata.raa import RAA_Transition, RAA
    #     trans1 = RAA_Transition(
    #         cv1 = Real("curr1"),
    #         cv2 = Real("curr2"),
    #         start_state= 0,
    #         guard = Real("curr1") == Real("curr2"),  
    #         assignment = {}, 
    #         end_state = 0,
    #         cost_function = [0, 0, 0],  
    #         move = 3
    #     )
    #     trans2 = RAA_Transition(
    #         cv1 = Real("curr1"),
    #         cv2 = Real("curr2"),
    #         start_state= 0,
    #         guard = Real("curr1") == Real("curr2"),  
    #         assignment = {}, 
    #         end_state = 0,
    #         cost_function = [0, 0, 0],  
    #         move = 2
    #     )
    #     # Add more transitions as needed for DTW
    #     return RAA(states=1, transitions=[trans1, trans2], final_states=[0])

    # auto1 = create_dtw_raa()
    # print(auto1.run_all_paths([0], [13, 11.5]))
     