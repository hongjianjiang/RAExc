import z3
from itertools import permutations
from graphviz import Source
import json
from itertools import product
# Fix import for benchmark_gen
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import benchmark_gen

class Transition:
    def __init__(self, curr_var, start_state, guard, assignment, end_state):
        self.curr_var = curr_var
        self.start_state = start_state
        self.guard = guard
        self.assignment = assignment
        self.end_state = end_state

    def is_enabled(self, start_state, symbol, registers, register_vars, solver):
        if self.start_state != start_state:
            return False
        solver.push()
        #print(" add: ", self.curr_var == symbol)
        #print(" add2: ", self.guard)
        solver.add(self.curr_var == symbol)
        solver.add(self.guard)
        #print(type(registers), registers, register_vars)
        for i in register_vars:
            #print(i == registers[str(i)])
            solver.add(i == registers[str(i)])
        res = solver.check()
        #print("\n -------- result ", res, "\n--------\n")
        solver.pop()
        if res == z3.sat:
            return True
        else:
            return False

    def getGuard(self):
        return self.guard

    def execute(self, symbol, registers):
        # Create a local context with individual register values
        local_context = registers
        local_context["curr"] = symbol  # Adding the current symbol to the context
        for reg, assign in self.assignment.items():
            if isinstance(assign, str):
                if assign.startswith(
                    "c"
                ):  # Assuming 'c' indicates copying the input symbol
                    registers[reg] = symbol
                elif assign.startswith("r"):  # Register to register assignment
                    registers[reg] = local_context[
                        assign
                    ]  # Directly use local_context to fetch register value
                else:
                    raise Exception(f"Unrecognized string assignment {assign}")
            elif isinstance(assign, (float, int)):
                registers[reg] = assign  # Direct assignment of a constant to a register
            elif isinstance(assign, z3.ExprRef):
                # This is a placeholder; actual implementation may vary based on needs
                registers[reg] = eval(str(assign), {"__builtins__": None}, local_context)
            else:
                raise Exception(
                    f"Assignment type not implemented for type {type(assign)}: {assign}"
                )

    def next_state(self):
        return self.end_state

    def __str__(self):
        return f"Transition: {self.start_state} -> {self.end_state}, Guard: {self.guard}, Assignment: {self.assignment}"


class RegisterAutomata:
    def __init__(self, states, transitions: Transition, register_vars, final_states):
        self.states = states
        self.transitions = transitions
        self.register_vars = register_vars
        self.final_states = final_states
        self.solver = z3.Solver()

    def complete(self):
        sink_state = self.states
        transitions_to_sink = [Transition(z3.Real("curr"), sink_state, True, {}, sink_state)]
        
        # Create an environment for evaluating string guards
        curr_var = z3.Real("curr")
        register_vars = {f"r{i}": var for i, var in enumerate(self.register_vars)}
        local_env = {**register_vars, "curr": curr_var, "And": z3.And, "Or": z3.Or, "Not": z3.Not}
        
        for state in range(self.states):
            outgoing_guards = [t.guard for t in self.transitions if t.start_state == state]
            
            # Process guards, converting strings to Z3 expressions if needed
            processed_guards = []
            for guard in outgoing_guards:
                if isinstance(guard, str):
                    if guard == "True":
                        processed_guards.append(True)
                    else:
                        try:
                            # Convert string guard to Z3 expression
                            processed_guards.append(eval(guard, {"__builtins__": None}, local_env))
                        except Exception as e:
                            print(f"Error evaluating guard: {guard}, {e}")
                            processed_guards.append(guard)  # Keep as is if eval fails
                else:
                    processed_guards.append(guard)
            
            if len(processed_guards) > 1:
                new_guards = z3.Not(z3.Or(*processed_guards))
            elif len(processed_guards) == 1:
                if processed_guards[0] == True:
                    continue
                new_guards = z3.Not(processed_guards[0])            
            else:
                new_guards = True
                
            transitions_to_sink.append(Transition(z3.Real("curr"), state, new_guards, {}, sink_state))
        
        if len(transitions_to_sink) == 1:
            return self

        return RegisterAutomata(self.states + 1, self.transitions + transitions_to_sink, self.register_vars, self.final_states)


    def complement(self):
        states = []
        completed_self = self.complete()
        for i in range(completed_self.states):
            if i not in completed_self.final_states:
                states.append(i)
                
        new_final_states = states
        

        return RegisterAutomata(completed_self.states, completed_self.transitions, completed_self.register_vars, new_final_states)

    def run(self, input_sequence):
        current_state = self.initial_state()
        registers = {str(i): 0 for i in self.register_vars}  # Initialize registers to 0 or any other default value

        for symbol in input_sequence:
            #print("symbol", symbol, "current state", current_state)
            next_state = self.transition(current_state, symbol, registers)
            # print("next state", next_state)
            if next_state is None:
                return False
            current_state = next_state
        # print("current state", current_state, "final states", self.final_states)
        return current_state in self.final_states

    def consistency_check(self, pos, neg):
        count1 = 0
        count2 = 0
        pos_err = []
        neg_err = []
        for i in pos:
            if not self.run(i):
                pos_err.append(i)
                count1 += 1

        for i in neg:
            if self.run(i):
                neg_err.append(i)
                count2 += 1

        return ((len(pos) - count1) / len(pos) + (len(neg) - count2) / len(neg)) / 2
    
    def initial_state(self):
        return 0  # Assuming initial state is always 0, you can modify as per your requirement

    def transition(self, state, symbol, registers):
        for transition in self.transitions:
            #print(transition)

            if transition.is_enabled(
                state, symbol, registers, self.register_vars, self.solver
            ):
                transition.execute(symbol, registers)
                return transition.next_state()
        return None

    def is_deterministic(self):
        for st in range(self.states):
            trans = [i for i in self.transitions if i.start_state == st]
            if len(trans) > 1:
                for trans_pair in permutations(trans, 2):
                    self.solver.push()
                    self.solver.add(trans_pair[0].guard)
                    self.solver.add(trans_pair[1].guard)
                    res = self.solver.check()
                    self.solver.pop()
                    if res == z3.sat:
                        return False
        return True

    def __str__(self):
        transitions_str = ", ".join(map(str, self.transitions))
        final_states_str = ", ".join(map(str, self.final_states))
        return f"States: {self.states}, Transitions: [{transitions_str}], Final States: [{final_states_str}]"
     
    def to_dot(self):
        dot = "digraph Automaton {\n"
        dot += 'init [label="", shape=point];\n'
        dot += 'init -> 0 [shape="solid"];\n'
        for transition in self.transitions:
            transition_assignments = []
            assign = dict()
            # print("assign", transition.assignment)
            if transition.guard == False:
                continue
            if not isinstance(transition.assignment, dict):
                for i in range(len(transition.assignment)):
                    assign[f"r{i}"] = transition.assignment[i]
            else:
                assign = transition.assignment
            for k, v in assign.items():
                transition_assignments.append(
                    k + ":=" + v
                )
            dot += f"  {transition.start_state} -> {transition.end_state} [label=\"{(transition.guard)}\\n {'; '.join(transition_assignments)}\"];\n"

        for i in self.final_states:
            dot += "  " + str(i) + " [shape=doublecircle];\n"
        dot += "}\n"
        return dot

    def store_dot(self, filename):
        dot = self.to_dot()
        with open(f"output/{filename}.dot", "w") as file:
            file.write(dot)

    def show(self, filename):
        dot = self.to_dot()
        s = Source(dot, "output/" + filename, format="png")
        s.view()

    def save_to_json(self, filename):
        # save to json in AlphabetRegisterAutomata format
        automaton_data = {
            "states": self.states,
            "transitions": [],
            "register_vars": [
                str(var) for var in self.register_vars
            ],  # Convert Z3 variables to string
            "final_states": self.final_states,
        }
        for transition in self.transitions:
            # print(transition.assignment)
            if not isinstance(transition.assignment, dict):
                assign = dict()
                for i in range(len(transition.assignment)):
                    assign[f"r{i}"] = transition.assignment[i]
            else:
                assign = transition.assignment
            transition_data = {
                "curr_var": str(transition.curr_var),  # Convert Z3 variables to string
                "start_state": transition.start_state,
                "guard": str(transition.guard),  # Convert Z3 guard to string
                "assignment": assign,  # Convert each Z3 expression in assignment to string
                "end_state": transition.end_state,

            }
            automaton_data["transitions"].append(transition_data)

        with open(filename, "w") as f:
            json.dump(automaton_data, f, indent=4)


def load_smt_dra(filename):
    # load a register automata from RALearner/registerAut.py format
    with open(filename, "r") as f:
        data = json.load(f)

    states = data["states"]
    final_states = data["final_states"]
    register_vars = [z3.Real(var) for var in data["register_vars"]]
    transitions = []

    # Define a dictionary to hold Z3 variable objects by name, to be used with eval
    z3_vars = {var: z3.Real(var) for var in data["register_vars"]}
    for trans_data in data["transitions"]:
        curr_var_name = trans_data["curr_var"]
        if curr_var_name not in z3_vars:
            z3_vars[curr_var_name] = z3.Real(
                curr_var_name
            )  # Add the current variable to the dictionary
        curr_var = z3_vars[curr_var_name]

        start_state = trans_data["start_state"]
        # Prepare an environment for eval where all variables are defined
        local_env = {**z3_vars, "And": z3.And, "Or": z3.Or, "Not": z3.Not}
        # print(trans_data["guard"])
        
        # Transform guard expression if it contains && operators
        guard_str = transform_guard_expression(trans_data["guard"])
        guard = eval(guard_str, {"__builtins__": None}, local_env)


        end_state = trans_data["end_state"]
        transitions.append(
            Transition(curr_var, start_state, guard, trans_data["assignment"], end_state)
        )
    ra = RegisterAutomata(states, transitions, register_vars, final_states)
    return ra


# Function to load a RegisterAutomata from Local search JSON
def load_double_ls_dra(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    z3_vars = {f"r{var}": z3.Real(f"r{var}") for var in range(data["num_registers"])}

    states = len(data["states"])
    num_registers = data["num_registers"]
    final_states = data["final_states"]
    register_vars = [z3.Real(f"r{i}") for i in range(num_registers)]
    cons_dict = {}
    for i, c in enumerate(data["constants"]):
        cons_dict[f"c{i}"] = c
    
    # Create transitions from the JSON data
    transitions = []
    guard_dict = {}
    for state, state_transitions in data["transitions"].items():
        curr_var_name = "curr"
        if curr_var_name not in z3_vars:
            z3_vars[curr_var_name] = z3.Real(
                curr_var_name
            )  # Add the current variable to the dictionary
        curr_var = z3_vars[curr_var_name]

        for condition_id, end_state in state_transitions:
            condition = data["condition_map"][str(condition_id)]
            lhs, op1, rhs, op2 = condition
            if "c" in lhs:
                lhs = cons_dict[lhs]
            if "c" in rhs:
                rhs = cons_dict[rhs]
            if op1 == "LT":
                op1 = "<"
            else:
                op1 = "<="
            if op2 == "LT":
                op2 = "<"
            else:
                op2 = "<="
            if lhs == "nop":
                if rhs == "nop":
                    guard = "True"
                else:
                    guard = f"curr{op2}{rhs}"
            elif rhs == "nop":
                tmp = ">" if op1 == "<" else ">="
                guard = f"curr{tmp}{lhs}"
            else:
                tmp = ">" if op1 == "<" else ">="
                guard = f"And(curr{tmp}{lhs}, curr{op2}{rhs})"
                # Prepare an environment for eval where all variables are defined
             
            local_env = {
                **z3_vars,
                "And": z3.And,
                "Or": z3.Or,
                "Not": z3.Not,
                "True": z3.BoolVal(True),
            }
            guard = eval(guard, {"__builtins__": None}, local_env)
            guard_dict[(state, end_state)] = guard

    assign_dict = {}
    for start_state, assign_transitions in data["register_transitions"].items():
        for assigs in assign_transitions:
            assign_id = assigs[0]
            end_state = assigs[1]

            assignment_pairs = data["assignment_map"][str(assign_id)]
            assignment_dict = {reg: val for reg, val in assignment_pairs}
            assign_dict[(start_state, end_state)] = assignment_dict

    for k, v in assign_dict.items():
        transitions.append(Transition(curr_var, int(k[0]), guard_dict[k], v, k[1]))

    # Create the RegisterAutomata instance
    return RegisterAutomata(states, transitions, register_vars, final_states)



# Function to load a RegisterAutomata from Local search JSON
def load_integer_ls_dra(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    z3_vars = {f"r{var}": z3.Real(f"r{var}") for var in range(data["num_registers"])}

    states = len(data["states"])
    num_registers = data["num_registers"]
    final_states = data["final_states"]
    register_vars = [z3.Real(f"r{i}") for i in range(num_registers)]
    cons_dict = {}
    for i, c in enumerate(data["constants"]):
        cons_dict[f"c{i}"] = c
    
    # Create transitions from the JSON data
    transitions = []
    guard_dict = {}
    for state, state_transitions in data["transitions"].items():
        curr_var_name = "curr"
        if curr_var_name not in z3_vars:
            z3_vars[curr_var_name] = z3.Real(
                curr_var_name
            )  # Add the current variable to the dictionary
        curr_var = z3_vars[curr_var_name]

        for condition_id, end_state in state_transitions:
            condition = data["condition_map"][str(condition_id)]
            lhs, op1 = condition
            if lhs:
                if "c" in lhs:
                    lhs = cons_dict[lhs]
                # if "c" in rhs:
                #     rhs = cons_dict[rhs]
            if op1 == "EQ":
                op1 = "=="
            else:
                op1 = "!="
            
            # if lhs == "nop":
            #     if rhs == "nop":
            #         guard = "True"
            #     else:
            #         guard = f"curr{op2}{rhs}"
            # elif rhs == "nop":
            #     tmp = ">" if op1 == "<" else ">="
            #     guard = f"curr{tmp}{lhs}"
            # else:
            if op1 == "TT":
                guard = "True"
            elif op1 == "FF":
                guard = "False"
            else:
                tmp = "==" if op1 == "==" else "!="
                guard = f"curr{tmp}{lhs}"
                # Prepare an environment for eval where all variables are defined
             
            local_env = {
                **z3_vars,
                "And": z3.And,
                "Or": z3.Or,
                "Not": z3.Not,
                "True": z3.BoolVal(True),
            }
            
            # print(guard)
            guard = eval(guard, {"__builtins__": None}, local_env)
            guard_dict[(state, end_state)] = guard

    assign_dict = {}
    for start_state, assign_transitions in data["register_transitions"].items():
        for assigs in assign_transitions:
            assign_id = assigs[0]
            end_state = assigs[1]

            assignment_pairs = data["assignment_map"][str(assign_id)]
            assignment_dict = {reg: val for reg, val in assignment_pairs}
            assign_dict[(start_state, end_state)] = assignment_dict

    for k, v in assign_dict.items():
        transitions.append(Transition(curr_var, int(k[0]), guard_dict[k], v, k[1]))

    # Create the RegisterAutomata instance
    return RegisterAutomata(states, transitions, register_vars, final_states)

def transform_guard_expression(guard_str):
    """
    Transform guard expressions from format like 'r2 != curr && r1 != curr'
    to format like 'And(r2 != curr, r1 != curr)'
    """
    if not isinstance(guard_str, str):
        return guard_str
    
    # Handle AND conditions with &&
    if '&&' in guard_str:
        # Split by && and clean up each condition
        conditions = [cond.strip() for cond in guard_str.split('&&')]
        if len(conditions) > 1:
            return f"And({', '.join(conditions)})"
        else:
            return conditions[0] if conditions else guard_str
    
    return guard_str

def create_v_register_automata(v):
    curr_var = z3.Real("curr")
    states = len(v)+2
    final_states = [len(v)]
    register_vars = [curr_var]
    transitions = []
    for i in range(len(v)):
        trans = Transition(
            curr_var,
            start_state=i,
            guard= int(v[i]) == curr_var,
            assignment={},
            end_state=i+1
        )
        
        transitions.append(trans)

    transitions.append(
        Transition(
            curr_var,
            start_state=len(v),
            guard=False,
            assignment={},
            end_state=len(v)+1
        )
    )

    return RegisterAutomata(states, transitions, register_vars, final_states)

def create_n_register_automata(v):
    curr_var = z3.Real("curr")
    states = len(v)+2
    final_states = [len(v)]
    register_vars = [curr_var]
    transitions = []
    for i in range(len(v)):
        trans = Transition(
            curr_var,
            start_state=i,
            guard= True,
            assignment={},
            end_state=i+1
        )
        
        transitions.append(trans)

    transitions.append(
        Transition(
            curr_var,
            start_state=len(v),
            guard=False,
            assignment={},
            end_state=len(v)+1
        )
    )

    return RegisterAutomata(states, transitions, register_vars, final_states)

if __name__ == "__main__":
    # Example usage of the RegisterAutomata class
    # auto1 = load_ls_dra("L1_ls.json")
    # auto1.show("L2")
    # auto1.save_to_json("L2_new.json")
    autov = create_v_register_automata([17, 17, 17, 17, 33, 33, 51, 58, 67])
    # autov.show("v")
    # print(autov.run([0,0,0,0,0]))
    # print(autov.run([0,0,0,0,1]))

    # auto1 = load_smt_dra("model.xml")
    # print(auto1.run([2,2,3,3]))
    # print(auto1.run([0,0,0,1,1,1]))
    # comp = auto1.complement()
    # print(comp.run([0,0,1,1]))
    # print(comp.run([0,0,0,1,1,1]))
    # auto1.show("S1")
    # # comp.show("Complemented")
    curr = z3.Real("curr")

    # Define transitions as per your specification
    transitions = [
        Transition(curr, 0, curr == 0, {}, 1),
        Transition(curr, 1, z3.And(curr > 0, curr < 8), {}, 2),
        Transition(curr, 2, z3.And(curr > 0, curr < 8), {}, 2),
        Transition(curr, 1, curr == 0, {}, 1),
        Transition(curr, 2, curr == 0, {}, 3),
        Transition(curr, 3, curr == 0, {}, 3),
    ]

    # Register variable list
    register_vars = []

    # Final state
    final_states = [3]

    # Number of states
    num_states = 4

    # Create the automaton
    ra = RegisterAutomata(num_states, transitions, register_vars, final_states)
    ra.show("example_automaton")
    pos, neg = benchmark_gen.random_sample_generation("mnist", 386, 386, 5)
    print(ra.consistency_check(pos, neg))
    for i in pos:
        if not ra.run(i):
            print("pos error: ", i)
    for i in neg:
        if ra.run(i):
            print("neg error: ", i)