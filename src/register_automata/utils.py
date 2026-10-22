from register_automata.ra import *
from z3 import *
import re
import sys
import random 
import networkx as nx
import z3
from typing import Any, Tuple, List, Set
from register_automata.ra import *
from z3 import *
import random
import re


def normalize_sequence(seq):
    if not seq:
        return []
    max_val = max(seq) if max(seq) > 0 else 1
    return [x / max_val for x in seq]



def evaluate_z3_expression(expr):
    """
    Convert Z3 expressions to Python native numeric types.
    
    Parameters:
    ----------
    expr : Any
        A Z3 expression or value to convert
        
    Returns:
    -------
    float or other native type
        The concrete value of the expression
    """
    # If already a Python native type, return as is
    if isinstance(expr, (int, float, bool)):
        return expr
    
    # Handle Z3 RatNumRef (rational numbers)
    if z3.is_rational_value(expr):
        return float(expr.numerator_as_long()) / float(expr.denominator_as_long())
    
    # Handle Z3 IntNumRef (integers)
    if z3.is_int_value(expr):
        return float(expr.as_long())
    
    # Handle Z3 BoolRef (boolean values)
    if z3.is_bool_value(expr):
        return expr.bool_value()
    
    # For other expressions, try to get a model and evaluate
    s = z3.Solver()
    s.add(expr == expr)  # Add a dummy constraint to force evaluation
    if s.check() == z3.sat:
        m = s.model()
        return float(m.evaluate(expr).as_fraction())
    
    # If all else fails, convert to string and try to extract a float
    try:
        return float(str(expr))
    except ValueError:
        raise ValueError(f"Could not evaluate Z3 expression: {expr}")


def extract_numbers_from_formula_string(formula_str):
    """
    Extract numeric constants (including rationals) from a string representation of a Z3 formula.
    
    Parameters
    ----------
    formula_str : str
        String representation of a Z3 formula
        
    Returns
    -------
    set
        Set of numeric constants found in the formula (as float)
    """
    # Extended regex to capture integers, floats, and rationals
    number_pattern = r'[-+]?\d+(?:\.\d+)?(?:/\d+)?'
    comparison_ops = ['<', '>', '<=', '>=', '==']

    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|'+number_pattern+r'|<=|>=|<|>|==|[(),]|\S', formula_str)
    
    constants = set()
    
    for i in range(len(tokens)):
        if tokens[i] in comparison_ops:
            # Check token before
            if i > 0 and re.fullmatch(number_pattern, tokens[i-1]):
                constants.add(parse_number(tokens[i-1]))
            # Check token after
            if i < len(tokens) - 1 and re.fullmatch(number_pattern, tokens[i+1]):
                constants.add(parse_number(tokens[i+1]))
    
    return constants

def parse_number(s):
    """
    Convert a string to float, handling rational fractions.
    """
    if '/' in s:
        num, denom = s.split('/')
        return float(num) / float(denom)
    else:
        return float(s)
    
    
def compute_alpha_from_dra(auto, delta):
    alpha = 0
    flag = False
    for t in auto.transitions:
        a = t.cost_function[0]
        b = t.cost_function[2]
        if a == 0:
            continue
        else:
            num = abs((delta - b) / a)
            alpha = max(alpha, num)
            flag = True
    return alpha, flag

def construct_alpha_bound_RAA(auto, delta, v):
    """
    Creates a bounded RAA where the second input is bounded by -alpha and alpha.
    
    Parameters
    ----------
    auto : RAA
        The register-augmented automaton to modify
    delta : float
        The distance threshold
        
    Returns
    -------
    RAA
        The modified RAA with bounded inputs
    """
    # Create proper Z3 expressions for bounds
    curr1 = z3.Real("curr1")
    auto_v = create_v_register_automata(v)
    auto = auto.vector_product(auto_v)
    alpha, flag = compute_alpha_from_dra(auto, delta)
    if not flag:
        alpha = sys.maxsize
    lhs = curr1 >= -alpha  # Lower bound constraint
    rhs = curr1 <= alpha   # Upper bound constraint 
    for t in auto.transitions: 
        t.guard = z3.And(z3.And(t.guard, lhs), rhs)


    return auto
    
def construct_desired_RAA(dra, raa, delta, v):
    # Corresponds to Lemma 1
    # dra : A
    # raa : A_D
    # Ouput: RAA A_D' which takes the product of A_D and the complement of A
    a_d = construct_alpha_bound_RAA(raa, delta, v)
    negation = dra.complement()

    res = a_d.product(negation) 

    return res

 

def clousure_raa(dra, raa, delta, v, convert_strict=True):
    """
    Convert all strict inequalities in RAA guards to non-strict inequalities.
    This effectively computes the closure of the language accepted by the RAA.
    
    Parameters
    ----------
    raa : RAA
        The register-augmented automaton to modify
        
    Returns
    -------
    RAA
        The modified RAA with non-strict inequalities
    """
    auto = construct_desired_RAA(dra, raa, delta, v)
    if convert_strict:
        for t in auto.transitions:
            t.guard = convert_strict_to_nonstrict(t.guard)
    return auto

def convert_strict_to_nonstrict(expr):
    """
    Recursively replace all < with <=, and > with >= in any Z3 formula.
    Works for nested or flat And/Or expressions and all subterms.
    """
    if z3.is_lt(expr):
        a, b = expr.children()
        return a <= b
    elif z3.is_gt(expr):
        a, b = expr.children()
        return a >= b
    elif z3.is_and(expr):
        # Flatten and process all children
        args = []
        for c in expr.children():
            cc = convert_strict_to_nonstrict(c)
            # Flatten again if child is an And (handles deep nesting)
            if z3.is_and(cc):
                args.extend(cc.children())
            else:
                if z3.is_lt(cc):
                    a, b = expr.children()
                    cc = a <= b
                elif z3.is_gt(cc):
                    a, b = expr.children()
                    cc = a >= b
                else:
                    # If it's not a strict inequality, just append
                    pass
                args.append(cc)

        return z3.And(*args)
    elif z3.is_or(expr):
        args = []
        for c in expr.children():
            cc = convert_strict_to_nonstrict(c)
            if z3.is_or(cc):
                args.extend(cc.children())
            else:
                args.append(cc)
        return z3.Or(*args)
    elif expr.num_args() > 0:
        # Recursively process any other operators (==, >=, <=, etc)
        return expr.decl()(*[convert_strict_to_nonstrict(c) for c in expr.children()])
    else:
        # Constants, variables
        return expr


def compute_C_from_alpha_bound_v_projected(RAA, v:list, delta):
    c = set(v)
    for i in v:
        c.add(i + delta)
        c.add(i - delta)
    # for t in RAA.transitions:
    #     c.update(extract_numbers_from_formula_string(str(t.guard)))
    alpha, flag = compute_alpha_from_dra(RAA, delta)
    if not flag:
        alpha = sys.maxsize

    return c, alpha


def add_total_combination_enums(regs, uc, ic):
    val_list = regs + list(uc) + list(ic)
    Enum1 = Datatype("Enum1")
    Enum2 = Datatype("Enum2")
    Enum3 = Datatype("Enum3")
    form_list = []

    enum1s = [f"curr<{i}" for i in val_list] + [f"curr<={i}" for i in val_list] + ["NU"]
    enum2s = [f"curr>{i}" for i in val_list] + [f"curr>={i}" for i in val_list] + ["NL"]
    enum3s = []

    for i in enum1s:
        Enum1.declare(i)

    Enum1 = Enum1.create()
    for i in range(len(enum1s)):
        for j in range(i + 1, len(enum1s)):
            form_list.append(Enum1.constructor(i)() != Enum1.constructor(j)())

    enum2s = [f"curr>{i}" for i in val_list] + [f"curr>={i}" for i in val_list] + ["NL"]
    for i in enum2s:
        Enum2.declare(i)

    Enum2 = Enum2.create()
    for i in range(len(enum2s)):
        for j in range(i + 1, len(enum2s)):
            form_list.append(Enum2.constructor(i)() != Enum2.constructor(j)())

    for i in regs:
        for j in regs:
            enum3s.append(f"{i}=={j}")
        enum3s.append(f"{i}==curr")

    for i in enum3s:
        Enum3.declare(i)

    Enum3 = Enum3.create()

    for i in range(len(enum3s)):
        for j in range(i + 1, len(enum3s)):
            form_list.append(Enum3.constructor(i)() != Enum3.constructor(j)())

    return form_list


def transform_edge_to_formula(auto, t, regs, uc, ic):
    # print(t.guard, t.assignment)
    val_list = regs + list(uc) + list(ic)
    Enum1 = Datatype("Enum1")
    Enum2 = Datatype("Enum2")
    Enum3 = Datatype("Enum3")
    enum1 = {}
    enum2 = {}
    enum3 = {}
    res_dict = dict()

    enum1s = [f"curr<{i}" for i in val_list] + [f"curr<={i}" for i in val_list] + ["NU"]
    enum2s = [f"curr>{i}" for i in val_list] + [f"curr>={i}" for i in val_list] + ["NL"]
    for i in enum1s:
        Enum1.declare(i)

    Enum1 = Enum1.create()
    enum2s = [f"curr>{i}" for i in val_list] + [f"curr>={i}" for i in val_list] + ["NL"]
    for i in enum2s:
        Enum2.declare(i)

    Enum2 = Enum2.create()
    enum3s = []
    for i in regs:
        for j in regs:
            enum3s.append(f"{i}=={j}")
        enum3s.append(f"{i}==curr")
    for i in enum3s:
        Enum3.declare(i)

    Enum3 = Enum3.create()

    start_state = f"s{t.start_state}"
    end_state = f"s{t.end_state}"
    import z3
    # Use z3 API to check guard type
    def normalize(s):
        return s.replace(' ', '')

    enum1s_norm = [normalize(e) for e in enum1s]
    enum2s_norm = [normalize(e) for e in enum2s]

    if z3.is_and(t.guard):
        children = t.guard.children()
        enum2_str = str(children[0])
        enum1_str = str(children[1])
        enum1_str_norm = normalize(enum1_str)
        enum2_str_norm = normalize(enum2_str)
        # Only search for <, <= in enum1s, >, >= in enum2s
        try:
            inx_enum1 = enum1s_norm.index(enum1_str_norm)
        except ValueError:
            inx_enum1 = len(enum1s) - 1  # fallback to last (default/NU)
        try:
            inx_enum2 = enum2s_norm.index(enum2_str_norm)
        except ValueError:
            inx_enum2 = len(enum2s) - 1  # fallback to last (default/NL)
    elif z3.is_lt(t.guard) or z3.is_le(t.guard):
        enum1_str = str(t.guard)
        enum1_str_norm = normalize(enum1_str)
        try:
            inx_enum1 = enum1s_norm.index(enum1_str_norm)
        except ValueError:
            inx_enum1 = len(enum1s) - 1
        inx_enum2 = len(enum1s) - 1
    elif z3.is_gt(t.guard) or z3.is_ge(t.guard):
        enum2_str = str(t.guard)
        enum2_str_norm = normalize(enum2_str)
        inx_enum1 = len(enum1s) - 1
        try:
            inx_enum2 = enum2s_norm.index(enum2_str_norm)
        except ValueError:
            inx_enum2 = len(enum2s) - 1
    else:
        inx_enum1 = len(enum1s) - 1
        inx_enum2 = len(enum2s) - 1
    ass_res = []
    # Handle both dict and list/tuple for t.assignment
    if isinstance(t.assignment, dict):
        for reg, val in t.assignment.items():
            ass_res.append(f"{reg}=={val}")
    else:
        for i, r in enumerate(auto.register_vars):
            try:
                ass_res.append(f"{r}=={t.assignment[i]}")
            except (IndexError, KeyError):
                pass

    for i in ass_res:
        inx_enum3 = enum3s.index(i)
    enum1[(start_state, end_state)] = Const(f"enum1_{start_state}_{end_state}", Enum1)
    enum2[(start_state, end_state)] = Const(f"enum2_{start_state}_{end_state}", Enum2)
    enum3[(start_state, end_state)] = Const(f"enum3_{start_state}_{end_state}", Enum3)

    res_dict[(start_state, end_state)] = [
            enum1[(start_state, end_state)] == Enum1.constructor(inx_enum1)(),
            enum2[(start_state, end_state)] == Enum2.constructor(inx_enum2)(),
            enum3[(start_state, end_state)] == Enum3.constructor(inx_enum3)(),
            inx_enum1, inx_enum2, inx_enum3
        ]
        
    
    return res_dict


def random_choice_states(auto):
    states = list(range(auto.states))
    half_states = len(states) // 2
    res_states = random.sample(states, half_states)
    # print("picked states: ", res_states)
    return res_states


def random_choice_edges(auto):
    transitions = auto.transitions
    half_size = len(transitions) // 2
    trans = random.sample(transitions, half_size)
    res = dict()
    for i in trans:
        res[(i.start_state, i.end_state)] = i
    return res


def keep_transtitions_related_states(auto, state):
    transitions = auto.transitions
    res = dict()
    for i in transitions:
        if i.start_state == state or i.end_state == state:
            res[(f"s{i.start_state}", f"s{i.end_state}")] = i
    return res

def get_all_transitions(auto):
    transitions = auto.transitions
    res = []
    for i in transitions:
        res.append((f"s{i.start_state}", f"s{i.end_state}"))
    return res

def transform_edge_to_formula_alpha(auto, t, regs, uc, ic):
    val_list = regs + list(uc) + list(ic)
    
    Enum1 = Datatype("Enum1")
    Enum3 = Datatype("Enum3")
    enum1 = {}
    enum3 = {}
    res_dict = dict()

    enum1s = (
            [f"curr--{i}" for i in val_list]
            + [f"curr<>{i}" for i in val_list]
            + ["tt", "ff"]
        )
    

    for i in enum1s:
        Enum1.declare(i)

    Enum1 = Enum1.create()


    enum3s = []
    for i in regs:
        for j in regs:
            enum3s.append(f"{i}=={j}")
        enum3s.append(f"{i}==curr")
    for i in enum3s:
        Enum3.declare(i)

    Enum3 = Enum3.create()

    start_state = f"s{t.start_state}"
    end_state = f"s{t.end_state}"
    # Check if the guard is an And formula
    match = re.search(r'(\w+)\s*([=!<>]=)\s*(\w+)', str(t.guard))
    var = match.group(1) if match else None
    op = match.group(2) if match else None

    if op == "==":
        res_op = '--'
    else:
        res_op = '<>'
    tmp_str = f'curr{res_op}{var}'
    if var == "True":
        inx_enum1 = enum1s.index("tt")
    else:
        inx_enum1 = enum1s.index(tmp_str)
    ass_res = []
    for i, r in enumerate(auto.register_vars):
        ass_res.append(f"{r}=={t.assignment[i]}")
    for i in ass_res:
        inx_enum3 = enum3s.index(i)
    enum1[(start_state, end_state)] = Const(f"enum1_{start_state}_{end_state}", Enum1)

    enum3[(start_state, end_state)] = Const(f"enum3_{start_state}_{end_state}", Enum3)

    res_dict[(start_state, end_state)] = [
            enum1[(start_state, end_state)] == Enum1.constructor(inx_enum1)(),
            enum3[(start_state, end_state)] == Enum3.constructor(inx_enum3)(),
            inx_enum1, inx_enum3, t.letter
        ]
    return res_dict



class Graph:
    """
    A NetworkX-backed graph representation for RAA computation.

    This preserves the original methods and signatures:
    - add_vertex
    - add_edge
    - construct_from_raa
    - get_ds_along_path
    - shortest_path
    - get_final_states
    - get_complement_final_states
    - all_paths_of_length_n
    - to_dot / visualize
    - total_distance_along_path

    Internally uses a nx.DiGraph for storage and algorithms.
    """
    def __init__(self):
        """Initialize an empty directed graph."""
        self.G = nx.DiGraph()

    def add_vertex(self, vertex: Tuple[int, Tuple[Any, ...]]):
        """Add a vertex (state, register_config)."""
        self.G.add_node(vertex)

    def add_edge(self,
                 u: Tuple[int, Tuple[Any, ...]],
                 v: Tuple[int, Tuple[Any, ...]],
                 weight: float,
                 d: Any):
        """Add or update edge (u->v), storing minimal absolute weight and d."""
        self.G.add_node(u)
        self.G.add_node(v)
        abs_w = abs(weight)
        if self.G.has_edge(u, v):
            # keep the smaller weight
            if abs_w < self.G[u][v]['weight']:
                self.G[u][v]['weight'] = abs_w
                self.G[u][v]['d'] = d
        else:
            self.G.add_edge(u, v, weight=abs_w, d=d)

    def construct_from_raa(self,
                            raa,
                            source_state: int,
                            C: Set[Any],
                            k: int):
        """
        BFS over RAA configurations to build graph.
        source_state: initial RAA state
        C: set of possible curr1 values
        k: number of registers
        """
        # starting vertex: all-zero registers
        source = (source_state, tuple([0]*k))
        self.G.add_node(source)
        queue = [source]
        seen = {source}

        # name-order for registers
        reg_names = list(raa.registers.keys())

        while queue:
            state, regs_tuple = queue.pop(0)
            regs = {reg_names[i]: regs_tuple[i] for i in range(k)}

            for t in raa.transitions:
                if t.start_state != state:
                    continue
                for d in C:
                    sat, value = t.is_enabled_fixed_curr1(
                        state, C, d, regs, raa.register_vars, raa.solver
                    )
                    if not sat:
                        continue

                    # simulate register update
                    new_regs = regs.copy()
                    for rname, aval in t.assignment.items():
                        if isinstance(aval, (int, float)):
                            new_regs[rname] = aval
                        elif isinstance(aval, z3.ExprRef):
                            new_regs[rname] = float(raa.solver.model().evaluate(aval))
                        else:
                            # Handle string assignments:
                            # 'c', 'curr', 'curr1' all mean "copy current symbol"
                            if aval in ('curr1', 'curr', 'c', 'c1'):
                                new_regs[rname] = evaluate_z3_expression(value)
                            elif aval in regs:
                                new_regs[rname] = regs[aval]
                            else:
                                new_regs[rname] = aval

                    next_regs = tuple(new_regs[n] for n in reg_names)
                    next_vertex = (t.end_state, next_regs)

                    # compute weight
                    w = t.cost_function[0] * d + t.cost_function[2]
                    self.add_edge((state, regs_tuple), next_vertex, w, d)

                    if next_vertex not in seen:
                        seen.add(next_vertex)
                        queue.append(next_vertex)

        return self

    def get_ds_along_path(self, path: List[Tuple[int, Tuple[Any,...]]]) -> List[Any]:
        """Return list of d's along edges of path."""
        ds = []
        for u, v in zip(path, path[1:]):
            if self.G.has_edge(u, v):
                ds.append(self.G[u][v]['d'])
            else:
                return []
        return ds

    def shortest_path(self,
                      source: Tuple[int, Tuple[Any,...]],
                      target: Tuple[int, Tuple[Any,...]]
                     ) -> Tuple[float, List[Tuple[int, Tuple[Any,...]]]]:
        """Dijkstra: returns (distance, path) or (inf, []) if unreachable."""
        try:
            length, path = nx.single_source_dijkstra(self.G,
                                                     source,
                                                     target,
                                                     weight='weight')
            return length, path
        except nx.NetworkXNoPath:
            return float('inf'), []

    def get_final_states(self, raa) -> List[Tuple[int, Tuple[Any,...]]]:
        """All vertices whose state is in raa.final_states."""
        return [n for n in self.G.nodes if n[0] in raa.final_states]

    def get_complement_final_states(self, raa) -> List[Tuple[int, Tuple[Any,...]]]:
        """All vertices whose state is not in raa.final_states."""
        return [n for n in self.G.nodes if n[0] not in raa.final_states]

    def all_paths_of_length_n(self,
                              source_vertex: Tuple[int, Tuple[Any,...]],
                              n: int,
                              target_vertices: Set[Tuple[int, Tuple[Any,...]]] = None
                             ) -> List[List[Tuple[int, Tuple[Any,...]]]]:
        """Enumerate all paths with exactly n edges."""
        results = []
        def dfs(path):
            if len(path)-1 == n:
                if target_vertices is None or path[-1] in target_vertices:
                    results.append(path.copy())
                return
            for succ in self.G.successors(path[-1]):
                dfs(path + [succ])

        if source_vertex not in self.G:
            raise ValueError("Source vertex not in graph.")
        dfs([source_vertex])
        return results

    def to_dot(self,
               source_vertex=None,
               target_vertices: List[Tuple[int, Tuple[Any,...]]] = None
              ) -> str:
        """Export Graphviz DOT, highlighting source/targets."""
        if target_vertices is None:
            target_vertices = []
        lines = ["digraph G {"]
        for n in self.G.nodes:
            state, regs = n
            label = f"q{state}_" + "_".join(map(str, regs))
            attrs = []
            if n == source_vertex:
                attrs.append('shape=diamond')
                attrs.append('style=filled')
                attrs.append('fillcolor=lightblue')
            if n in target_vertices:
                attrs.append('shape=doublecircle')
                attrs.append('style=filled')
                attrs.append('fillcolor=lightgreen')
            attr_str = ",".join(attrs)
            lines.append(f'  "{n}" [label="{label}"{","+attr_str if attr_str else ""}];')
        for u, v, data in self.G.edges(data=True):
            lines.append(f'  "{u}" -> "{v}" [label="w={data["weight"]:.2f},d={data["d"]}"]')
        lines.append('}')
        return "\n".join(lines)

    def visualize(self,
                  filename: str = "graph",
                  source_vertex=None,
                  target_vertices: List[Tuple[int, Tuple[Any,...]]] = None):
        """Render via Graphviz to PDF."""
        from graphviz import Source
        dot = self.to_dot(source_vertex, target_vertices)
        src = Source(dot)
        src.render(filename, format='pdf', cleanup=True)

    def total_distance_along_path(self, path: List[Tuple[int, Tuple[Any,...]]]) -> float:
        """Sum of weights along path or inf if missing edge."""
        total = 0.0
        for u, v in zip(path, path[1:]):
            if self.G.has_edge(u, v):
                total += self.G[u][v]['weight']
            else:
                return float('inf')
        return total
