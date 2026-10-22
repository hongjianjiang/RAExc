#!/bin/env python3

import signal
import z3
from z3 import Real, Bool, Or, And, Not, Implies, Xor, Datatype, Const, AtMost, AtLeast, Context
from itertools import  product
import re
# Fix import for register_automata.ra
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import register_automata.ra
# import tomitaGen
import time


class Register_Learner(object):
    def __init__(self, n = 1, regs = list(), ic =  list(), uc = list(), neg = list(), pos = list(), edge_limit_type = 'at_most', edge_limit_num = 2, debug = False, batch_size = 1, target = 1):
        self._n = n
        self.regs = regs
        self.ic = ic 
        self.uc = uc
        self.pos = pos
        self.neg = neg
        self._solver = None
        self.debug = debug
        self.outgoing = (edge_limit_type, edge_limit_num)
        self.start = time.time()
        self.batch_size = batch_size
        self.target = target
        self._solver = z3.Solver()

        if self.outgoing[1] > self._n:
            raise Exception(f"Allowed at most {self._n} outgoing edges")
        
        # self.build_form()




    def prefix_list(self, l):
        return [l[:i] for i in range(1, len(l)+1)]  # Start from index 1 to skip empty prefix


    def extract_register(self, r):
        return re.findall('r\d*', r, re.S)[-1]

    def has_numbers(self, inputString):
        return any(char.isdigit() for char in inputString)

    def conv_2_str(self, l):
        if l == []:
            return ""
        return '_'.join(str(i) for i in l)

    
    def extract_re_from_enum1(self, s):
        if "=" in s:
            return s.split("=")[1]
        else:
            return s.split("<")[1]
        
    def extract_re_from_enum2(self, s):
        if "=" in s:
            return s.split("=")[1]
        else:
            return s.split(">")[1]
    # (=, !=, T, \bot)
    def translate_enum_1(self, index, curr, list_interpreted_constants, list_uninterpreted_constants, list_registers):
        all = list_registers + list_uninterpreted_constants + list_interpreted_constants
        all = all + all  # Duplicate the list

        if index < len(all) / 2:
            if all[index] in list_interpreted_constants:
                return float(curr) == all[index]
            else:
                return curr == Real(all[index])
        elif len(all) / 2 <= index < len(all):
            if all[index] in list_interpreted_constants:
                return float(curr) != all[index]
            else:
                return curr != Real(all[index])
        elif index == len(all):
            return True
        else:
            return False

        
    def translate_enum_1_curr(self, index, curr, list_interpreted_constants, list_uninterpreted_constants, list_registers):
        all = list_registers + list_uninterpreted_constants + list_interpreted_constants
        all = all + all
        if (index < (len(all))/2):
            if all[index] in list_interpreted_constants:
                return float(curr) == all[index]
            else:
                return curr == Real(all[index])
        elif (index >= (len(all))/2 and index < len(all)):
            if all[index] in list_interpreted_constants:
                return float(curr) != all[index]
            else:
                return curr != Real(all[index])
        else:
            return True
         
    
    def translate_enum_3(self, index, curr, list_registers):
        d_v = int(index / (len(list_registers) + 1))
        m_v = (index % (len(list_registers) + 1))
        if len(list_registers) == 1:
            if m_v == len(list_registers):
                return Real(f"{list_registers[d_v]}_{curr}") == (curr)
            else:
                return Real(f"{list_registers[d_v]}_{curr}") == Real(list_registers[m_v])
        else:
            fls = [Real(f"{x}_{curr}") == Real(f"{x}") for x in list_registers if x != list_registers[d_v]]
            if m_v == len(list_registers):
                return And(Real(f"{list_registers[d_v]}_{curr}") == (curr), And(fls))
            else:
                return And(Real(f"{list_registers[d_v]}_{curr}") == Real(list_registers[m_v]), And(fls))
               
    def divide(self, enums1, enums3):
        re1 = []
        re2 = []
        re3 = []
        re4 = []

        for i in product(enums1, enums3):
            # equality
            if "--" in i[0]:
                re1.append(i)
            #inequality
            elif "<>" in i[0]: 
                re2.append(i)
            elif "tt" in i[0]:
                re3.append(i)
            else:
                re4.append(i)
        return re1, re2, re3, re4
    
    def build_form(self, pos, neg):
        z3.set_option(rational_to_decimal=True)
        z3.set_option(precision=6)

        Enum1 = Datatype('Enum1')
        Enum3 = Datatype('Enum3')
            
        val_list = self.regs + list(self.uc) + list(self.ic)
        res = []
        enum1s = [f'curr--{i}' for i in val_list] + [f'curr<>{i}' for i in val_list] + ['tt', 'ff']
        for i in enum1s:
            Enum1.declare(i)
        
        Enum1 = Enum1.create()
        for i in range(len(enum1s)):
            for j in range(i+1, len(enum1s)):
                res.append(Enum1.constructor(i)() != Enum1.constructor(j)())
        
        enum3s = []
        for i in self.regs:
            for j in self.regs:
                enum3s.append(f'{i}=={j}')
            enum3s.append(f'{i}==curr')

        for i in enum3s:
            Enum3.declare(i)
        
        Enum3 = Enum3.create()

        for i in range(len(enum3s)):
            for j in range(i+1, len(enum3s)):
                res.append(Enum3.constructor(i)() != Enum3.constructor(j)())

        enum1 = {}  
        enum3 = {} 

        states = [f"s{i}" for i in range(self._n)] # size of automaton 

        for p in states:
            for q in states:
                enum1[(p, q)] = Const(f"enum1_{p}_{q}", Enum1)
                enum3[(p, q)] = Const(f"enum3_{p}_{q}", Enum3)
        # oliver deterministic part

      
        temp_soler = z3.Solver()

        def det_guard2(source_state, destination_state, third_state, enum1_index1, enum1_index2):
            temp_soler.push()
            curr = z3.Real("curr1")
            #print("Guard check: ", enum1_index1, enum1_index2)
            temp_soler.add(And(self.translate_enum_1_curr(enum1_index1, curr, self.ic, self.uc, self.regs),
                               self.translate_enum_1_curr(enum1_index2, curr, self.ic, self.uc, self.regs)))
            if temp_soler.check() == z3.sat:
                temp_soler.pop()
                return And(Implies(And(enum1[(source_state, destination_state)] == Enum1.constructor(enum1_index1)(),
                               enum1[(source_state, third_state)] == Enum1.constructor(enum1_index2)()), False)
                        ,Implies(And(enum1[(source_state, destination_state)] == Enum1.constructor(enum1_index1)(),
                                                    enum1[(source_state, third_state)] == Enum1.constructor(enum1_index2)()), False)
                        )            
            else:
                temp_soler.pop()
                '''print("unsat")
                temp_soler.pop()
                return Implies(And(enum1[(source_state, destination_state)] == Enum1.constructor(enum1_index1)(),enum2[(source_state, destination_state)] == Enum2.constructor(enum2_index1)(),
                               enum1[(source_state, third_state)] == Enum1.constructor(enum1_index2)(),enum2[(source_state, third_state)] == Enum2.constructor(enum2_index2)()), True)'''
                return True
        
        # # deterministic part
        for p in states:
             for q in states:
                 for q_prime in states:
                     if q != q_prime:
                         for i in range(len(enum1s) - 1):
                                 for i1 in range (len(enum1s) - 1):
                                     res.append(det_guard2(p,q,q_prime, i,i1))


        if self.debug:
            print('===========prefix part===========')

        prefix = pos + neg        
        prefix_list = list()
        for u in prefix:
            for t in self.prefix_list(u):
                if self.conv_2_str(t) in prefix_list: 
                    pass 
                else:
                    prefix_list.append(self.conv_2_str(t))

        neg_prefix = list()
        for u in neg:
            for t in self.prefix_list(u):
                if self.conv_2_str(t) in neg_prefix: 
                    pass 
                else:
                    _tmp = self.conv_2_str(t)
                    if (_tmp != ''):
                      neg_prefix.append(self.conv_2_str(t))

        pos_prefix = list()
        for u in pos:
            for t in self.prefix_list(u):
                if self.conv_2_str(t) in pos_prefix: 
                    pass 
                else:
                    _tmp = self.conv_2_str(t)
                    if (_tmp != ''):
                        pos_prefix.append(self.conv_2_str(t))
        

        # for every positive sample/prefix the state x_t_q can be reached
        for t in pos_prefix:
            if self.debug:
                print(And(Or([Bool(f"x_{t}_{q}") for q in states])))
            res.append(And(Or([Bool(f"x_{t}_{q}") for q in states])))


        if self.debug:
            print('===========initial===========')
            print(Bool(f"x__s0"))

        res.append(Bool(f"x__s0"))

        for r in self.regs:
            if self.debug:
                print(Real(f"{r}") == 0)
            res.append(Real(f"{r}") == 0)

        eq_form, neq_form, tt_form, ff_form = self.divide(enum1s, enum3s)

        # TODO also consider the boolean variable?
        def transition_constraint(outgoing):
            flag = outgoing[0]
            num = self._n-outgoing[1]
            result = True
            if flag == 'at_least':
                result = And(result, 
                        And([
                        AtMost(*[And(enum1[("s"+str(m), "s"+str(e))] == Enum1.constructor(len(val_list) * 2  + 1)(),
                         ) for e in range(self._n)],  num)
                        for m in range(self._n)])
                        )
            elif flag == 'at_most':
                result = And(result, 
                        And([
                        AtLeast(*[(And(enum1[("s"+str(m), "s"+str(e))] == Enum1.constructor(len(val_list) * 2  + 1)()
                                       )) for e in range(self._n)],  num)
                        for m in range(self._n)])
                        )
            
            else:
                result = And(result, 
                            And([And(
                            (AtMost(*([(And(enum1[("s"+str(m), "s"+str(e))] == Enum1.constructor(len(val_list) * 2  + 1)(),
                                            )) for e in range(self._n)]),  num)
                            ),
                            (AtLeast(*([(And(enum1[("s"+str(m), "s"+str(e))] == Enum1.constructor(len(val_list) * 2  + 1)()
                                            )) for e in range(self._n)]),  num)
                            )
                            
                            ) 
                            for m in range(self._n)])
                            )
                
            return result
        if self.debug:
            print(f'===========add_{self.outgoing[0]}_{self.outgoing[1]}_transitions part===========')
            print(transition_constraint(self.outgoing))
        res.append(transition_constraint(self.outgoing))
        def add_clauses(q, q_, eq_form, neq_form, tt_form, ff_form, rs, but_last, curr, word, debug):
            for eq_f in eq_form:
                eq_clause = Implies(And(Bool(f"x_{self.conv_2_str(but_last)}_{q}"), 
                                                        enum1[(q, q_)] == Enum1.constructor(enum1s.index(eq_f[0]))(), 
                                                        enum3[(q, q_)] == Enum3.constructor(enum3s.index(eq_f[1]))(),
                                                        self.translate_enum_1(enum1s.index(eq_f[0]), curr, self.ic, self.uc, rs)), 
                                                        And(Bool(f"x_{word}_{q_}"), self.translate_enum_3(enum3s.index(eq_f[1]), curr, rs)))
                eq_clause_2 = Implies(And(Bool(f"x_{self.conv_2_str(but_last)}_{q}"), 
                                                        enum1[(q, q_)] == Enum1.constructor(enum1s.index(eq_f[0]))(), 
                                                        enum3[(q, q_)] == Enum3.constructor(enum3s.index(eq_f[1]))(),
                                                        Not(And(self.translate_enum_1(enum1s.index(eq_f[0]), curr, self.ic, self.uc, rs)))), 
                                                        And(Not(Bool(f"x_{word}_{q_}"))))
                res.append(eq_clause)
                res.append(eq_clause_2)
                if debug:
                    print("eq_form::\n",eq_clause, '\n', eq_clause_2)

            for nf in neq_form:
                neq_clause = Implies(And(Bool(f"x_{self.conv_2_str(but_last)}_{q}"), 
                                                    enum1[(q, q_)] == Enum1.constructor(enum1s.index(nf[0]))(), 
                                                    enum3[(q, q_)] == Enum3.constructor(enum3s.index(nf[1]))(),
                                                    self.translate_enum_1(enum1s.index(nf[0]), curr, self.ic, self.uc, rs)), 
                                                    And(Bool(f"x_{word}_{q_}"), self.translate_enum_3(enum3s.index(nf[1]), curr, rs)))
                neq_clause_2 = Implies(And(Bool(f"x_{self.conv_2_str(but_last)}_{q}"), 
                                                    enum1[(q, q_)] == Enum1.constructor(enum1s.index(nf[0]))(), 
                                                    enum3[(q, q_)] == Enum3.constructor(enum3s.index(nf[1]))(),
                                                    Not(And(self.translate_enum_1(enum1s.index(nf[0]), curr, self.ic, self.uc, rs)))), 
                                                    And(Not(Bool(f"x_{word}_{q_}"))))
                res.append(neq_clause)
                res.append(neq_clause_2)
                if debug:
                    print("neq_form:\n",neq_clause,'\n', neq_clause_2)
                    
            for tt_f in tt_form:
                tt_clause = Implies(And(Bool(f"x_{self.conv_2_str(but_last)}_{q}"), 
                                                        enum1[(q, q_)] == Enum1.constructor(enum1s.index(tt_f[0]))(), 
                                                        enum3[(q, q_)] == Enum3.constructor(enum3s.index(tt_f[1]))(),
                                                        
                                                        ), 
                                                        And(Bool(f"x_{word}_{q_}"), self.translate_enum_3(enum3s.index(tt_f[1]), curr, rs)))
                res.append(tt_clause)
                if debug:
                    print("true_form:\n",tt_clause)

            for ff in ff_form:
                nt_clause = Implies(And(Bool(f"x_{self.conv_2_str(but_last)}_{q}"), 
                                                        enum1[(q, q_)] == Enum1.constructor(enum1s.index(ff[0]))()
                                                        ), 
                                                        Not(Bool(f"x_{word}_{q_}")))
                res.append(nt_clause)
                if debug:
                    print("false_form::",nt_clause)

           
        if self.debug:
            print('===========sample part===========')
        cache_list = []
        for q in states:
            for q_ in states:  
                for p in pos + neg:
                    ps = self.prefix_list(p)
                    rs = self.regs
                    for e in ps:
                        t = self.conv_2_str(e)
                        if t != "":
                            tmp_list = t.split("_")
                            a = tmp_list[-1]
                            u = tmp_list[:-1]
                            add_clauses(q, q_, eq_form, neq_form, tt_form, ff_form, rs, but_last=u, curr=a, word=t, debug=self.debug)
                            rs = [f"{i}_{a}" for i in rs]
                        if q != q_:
                            res.append(Or(Not(Bool(f"x_{t}_{q}")), 
                                                Not(Bool(f"x_{t}_{q_}"))
                                )) 
                            if self.debug:
                                print(f'sampe {q} {q_}:', Or(Not(Bool(f"x_{t}_{q}")), 
                                                Not(Bool(f"x_{t}_{q_}"))
                                ))

        

                                
         # positive examples constraint
        if self.debug:
            print('=======positive examples======')
        
        for u in pos:
            pos_tmp = []
            for i in self.prefix_list(u):
                if i != []:
                    pos_tmp.append(i)
            for q in states:
                res.append(Implies(Bool(f"x_{self.conv_2_str(u)}_{q}"), Bool(f"f_{q}")))

	    # negative examples constraint
        if self.debug:
            print('=======negative examples======')
        for u in neg:
            neg_tmp = []
            for i in self.prefix_list(u):
                if i != []:
                    neg_tmp.append(i)
            for q in states:
                if self.debug:
                    print(Implies(Bool(f"x_{self.conv_2_str(u)}_{q}"), Not(Bool(f"f_{q}"))))
                res.append(Implies(Bool(f"x_{self.conv_2_str(u)}_{q}"), Not(Bool(f"f_{q}"))))
        if self.debug:
            print('=============')
        end = time.time()
        # print("Write time: ", end-self.start)
        return res

    def parse_input_string(self, input_string, model):
        # Handle 'tt' (True) and 'ff' (False) directly
        if input_string.strip() == 'tt':
            return True
        if input_string.strip() == 'ff':
            return False

        # Regular expression pattern to match the input string
        pattern = r'(.+?)\s*(==|--|<>|<<|>>)\s*(.+)'
        match = re.match(pattern, input_string)
        if match:
            left_expression = match.group(1).strip()
            operator = match.group(2).strip()
            right_expression = match.group(3).strip()
            left_expression = self.convert_str_to_z3(left_expression)
            right_expression = self.convert_str_to_z3(right_expression)

            if str(left_expression) in self.uc:
                left_expression = model[left_expression]
            if str(right_expression) in self.uc:
                right_expression = model[right_expression]
            if operator == "--":
                return left_expression == right_expression
            elif operator == "<>":
                return left_expression != right_expression
            elif operator == "==":
                return (left_expression, right_expression)
        else:
            return None


    def convert_str_to_z3(self, expression):
        try:
            num = float(expression)
            return num
        except ValueError:
            return Real(expression)
        
    def insert_new_param(self, s, rs, cs):
        s.append(len(s)+1)
        if len(rs) == 0:
            rs=[['r0']]
        else:
            n_rs = rs[-1]+ [f"r{len(rs[-1])}"]
            rs.append(n_rs)
        if len(cs) == 0:
            cs = [['c0']]
        else:
            n_cs = cs[-1]+ [f"c{len(cs[-1])}"]
            cs.append(n_cs)
        return s, rs, cs

    def get_model(self, timeout=600):
        """
        Attempt to get a model from the solver, with a timeout in seconds (default 3600s = 60min).
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            pos = []
            neg = []
            inx = 0
            for i in range(0, len(self.pos) // self.batch_size):
                tmp_pos = []
                tmp_neg = []
                for j in range(self.batch_size):
                    inx = i * self.batch_size + j
                    tmp_pos.append(self.pos[inx])
                    tmp_neg.append(self.neg[inx])
                # pos.extend(tmp_pos)
                # neg.extend(tmp_neg)
                res = self.build_form(tmp_pos, tmp_neg)
                self._solver.add(*res)
                if self._solver.check() == z3.unsat:
                    print('unsat')
                    return "unsat", i
            
            # if len(self.pos) % self.batch_size != 0:
            #     pos.extend(self.pos[inx + 1 : len(self.pos)])
            # if len(self.neg) % self.batch_size != 0:
            #     neg.extend(self.neg[inx + 1 : len(self.pos)])
            # if len(self.pos) % self.batch_size != 0:
            #     self.build_form(pos, neg)
                
            # if self._solver.check() == z3.unsat:
            #     return "unsat"
                else:
                    model = self._solver.model()
                    final_states = []
                    transition_map = {}
                    for q in range(0, self._n):
                        for q_ in range(0, self._n):
                            transition_map[(q, q_)] = []
                    assignment_map = {}
                    for q in range(0, self._n):
                        for q_ in range(0, self._n):
                            assignment_map[(q, q_)] = {}
                            for r in self.regs:
                                assignment_map[(q, q_)][r] = r
                    for j in model.decls():
                        name = j.name()
                        if name[0] == "f":
                            if model[j] == True:
                                final_states.append(int(name.split("_")[1][1:]))
                        elif name[0] == "e":
                            parsed = self.parse_input_string(str(model[j]), model)
                            if parsed != None:
                                split_name = name.split("_")
                                enum_val = split_name[0][-1]
                                source_state = int(split_name[1][-1])
                                destination_state = int(split_name[2][-1])
                                if enum_val == "3":
                                    reg_name = str(parsed[0])
                                    assignment_map[source_state, destination_state][reg_name] = str(parsed[1])
                                else:
                                    transition_map[source_state, destination_state] = (
                                        transition_map[source_state, destination_state]
                                        + [parsed]
                                    )
                    states = self._n
                    register_vars = []
                    for r in self.regs:
                        register_vars.append(Real(r))
                    transitions = []
                    curr_var = Real("curr")
                    transition_solver = z3.Solver()
                    for q in range(0, self._n):
                        for q_ in range(0, self._n):
                            if len(transition_map[q, q_]) > 1:
                                transition_solver.push()
                                transition_solver.add(And(transition_map[(q, q_)]))
                                if transition_solver.check() == z3.sat:
                                    transitions.append(
                                        register_automata.ra.Transition(
                                            curr_var,
                                            q,
                                            And(transition_map[q, q_]),
                                            assignment_map[q, q_],
                                            q_,
                                        )
                                    )
                                transition_solver.pop()
                            else:
                                if len(transition_map[q, q_]) > 0:
                                    transitions.append(
                                        register_automata.ra.Transition(
                                            curr_var,
                                            q,
                                            transition_map[q, q_][0],
                                            assignment_map[q, q_],
                                            q_,
                                        )
                                    )
                                else:
                                    transitions.append(
                                        register_automata.ra.Transition(
                                            curr_var, q, True, assignment_map[q, q_], q_
                                        )
                                    ) 
                    auto = register_automata.ra.RegisterAutomata(
                        states, transitions, register_vars, final_states
                    )
                    acc = self.consistence_check(auto, self.pos, self.neg)
                    # print(f"Batch {i}:", acc)
                    if acc > self.target or i == 6:
                        return auto, i
                    else:
                        continue
        return "timeout", None

    def consistence_check(self, aut, pos, neg):
        # print(f"------------Consistence_check: Thread {name}------------")
        count1 = 0
        count2 = 0
        pos_err = []
        neg_err = []
        for i in pos:
            if not aut.run(i):
                pos_err.append(i)
                count1 +=1

        for i in neg:
            if aut.run(i):
                neg_err.append(i)
                count2 +=1
        return ((len(pos)-count1)/len(pos) + (len(neg)-count2)/len(neg)) / 2

if __name__ == '__main__':
    start = time.time()
    # from tomitaGen_numpy import Dyck1
    from benchmark_gen import random_sample_generation
    pos, neg = random_sample_generation("mnist", 100 , 10, 10) 
    learner = Register_Learner(n =  4, 
                               regs= ['r0', 'r1'], 
                               ic = [], 
                               uc = [], 
                               pos = pos, 
                               neg = neg,
                            #    pos = [[("push", 1),("pop", 1)]], 
                            #    neg = [[("push", 1),("push", 1)]], 
                               edge_limit_type= 'at_most',
                               edge_limit_num = 2,
                               debug=False,
                               batch_size = 10)
    auto = learner.get_model()
    end = time.time()
    print("total time: ", end - start)
    print(learner.consistence_check(auto, pos, neg))
