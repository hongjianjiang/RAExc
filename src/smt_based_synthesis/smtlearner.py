#!/bin/env python3

import z3
from z3 import (
    Real,
    Bool,
    Or,
    And,
    Not,
    Implies,
    Xor,
    Datatype,
    Const,
    AtMost,
    AtLeast,
    Context,
)
from itertools import product
import re
import time
import argparse
import threading
import register_automata.ra
import csv
import register_automata.utils
import random
import signal


class Register_Learner(object):
    def __init__(
        self,
        n=1,
        regs=list(),
        ic=list(),
        uc=list(),
        neg=list(),
        pos=list(),
        edge_limit_type="at_most",
        edge_limit_num=2,
        auto=None,
        batch_size = 1,
        debug=False,
        strategy = "state",
        target = 1
    ):
        self._n = n
        self.regs = regs
        self.ic = ic
        self.uc = uc
        self.pos = pos
        self.neg = neg
        self._solver = None
        self.debug = debug
        self.outgoing = (edge_limit_type, edge_limit_num)
        self.auto = auto
        self.strategy = strategy
        self.batch_size = batch_size
        self.all_transitions = self.calculate_pair_states()
        self.bottom_transtions = []
        self.target = target
        self._solver = z3.Solver()


        if self.auto:
            self.random_states = utils.random_choice_states(auto)
            self.random_trans = utils.random_choice_edges(auto)
            self.used_state_pairs = []
            self.saved_formulas = dict()
            self.all_transitions = self.calculate_pair_states()
            half = len(self.all_transitions) // 2
            half_choiced_trans = random.sample(self.all_transitions, half)
            if self.strategy == "state":
                for i in self.random_states:
                    for k, v in utils.keep_transtitions_related_states(auto, i).items():
                        self.used_state_pairs.append(k)
                        self.saved_formulas.update(
                            utils.transform_edge_to_formula(auto, v, regs, self.uc, self.ic)
                        )
            else:
                for k, v in self.random_trans.items():
                    self.used_state_pairs.append((f"s{k[0]}", f"s{k[1]}"))
                    self.saved_formulas.update(
                        utils.transform_edge_to_formula(auto, v, regs, self.uc, self.ic)
                    )
             
            
            for i in half_choiced_trans:
                if i not in utils.get_all_transitions(self.auto): 
                    self.bottom_transtions.append(i)
            
            self.search_transitions = [trans for trans in self.all_transitions if trans not in self.bottom_transtions and trans not in self.used_state_pairs]
            # print("total_search", self.all_transitions)
            # print("half trans:", half_choiced_trans)
            # print("fixde ls trains", self.used_state_pairs)
            # print("random_bottom trans:", self.bottom_transtions)
            # print("left search trans:", self.search_transitions)
            
        else:
            self.used_state_pairs = []
            self.saved_formulas = dict()
            self.search_transitions = self.all_transitions


        if self.outgoing[1] > self._n:
            raise Exception(f"Allowed at most {self._n} outgoing edges")

        # self.build_form()

    def prefix_list(self, l):
        return [l[:i] for i in range(len(l) + 1)]

    def extract_register(self, r):
        return re.findall("r\d*", r, re.S)[-1]

    def has_numbers(self, inputString):
        return any(char.isdigit() for char in inputString)

    def calculate_pair_states(self):
        res = list(
            product(
                [f"s{i}" for i in list(range(self._n))],
                [f"s{i}" for i in list(range(self._n))],
            )
        )
        return res

    def conv_2_str(self, l):
        if l == []:
            return ""
        return "_".join(str(i) for i in l)

    def divide1(self, l1, l2, ic, uc, regs):
        re1 = []
        re2 = []
        for i in product(l1, l2):
            if "=" in i[0] and "=" in i[1]:
                re1.append(i)
            elif "NU" in i[0] or "NL" in i[1]:
                re1.append(i)
            elif self.extract_re_from_enum1(i[0]) in [
                str(i) for i in ic
            ] and self.extract_re_from_enum2(i[1]) not in [str(i) for i in ic]:
                re1.append(i)
            elif self.extract_re_from_enum1(
                i[0]
            ) in regs and self.extract_re_from_enum2(i[1]) in [str(i) for i in ic]:
                re1.append(i)
            else:
                re2.append(i)
        return re1, re2

    def divide(self, l1, l2, l3, ic, uc, regs):
        re1 = []
        re2 = []
        for i in product(l1, l2, l3):
            if "=" in i[0] and "=" in i[1]:
                re1.append(i)
            elif "NU" in i[0] or "NL" in i[1]:
                re1.append(i)
            elif self.extract_re_from_enum1(i[0]) in [
                str(i) for i in ic
            ] and self.extract_re_from_enum2(i[1]) not in [str(i) for i in ic]:
                re1.append(i)
            elif self.extract_re_from_enum1(
                i[0]
            ) in regs and self.extract_re_from_enum2(i[1]) in [str(i) for i in ic]:
                re1.append(i)
            else:
                re2.append(i)
        return re1, re2

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

    def translate_enum_1(
        self,
        index,
        curr,
        list_interpreted_constants,
        list_uninterpreted_constants,
        list_registers,
    ):
        all = list_registers + list_uninterpreted_constants + list_interpreted_constants
        all = all + all
        if index < (len(all)) / 2:
            if all[index] in list_interpreted_constants:
                return float(curr) < all[index]
            else:
                return curr < Real(all[index])
        elif index >= (len(all)) / 2 and index != len(all):
            if all[index] in list_interpreted_constants:
                return float(curr) <= all[index]
            else:
                return curr <= Real(all[index])
        else:
            return True

    def translate_enum_1_curr(
        self,
        index,
        curr,
        list_interpreted_constants,
        list_uninterpreted_constants,
        list_registers,
    ):
        all = list_registers + list_uninterpreted_constants + list_interpreted_constants
        all = all + all
        if index < (len(all)) / 2:
            if all[index] in list_interpreted_constants:
                return curr < all[index]
            else:
                return curr < Real(all[index])
        elif index >= (len(all)) / 2 and index != len(all):
            if all[index] in list_interpreted_constants:
                return curr <= all[index]
            else:
                return curr <= Real(all[index])
        else:
            return curr == curr

    def translate_enum_2(
        self,
        index,
        curr,
        list_interpreted_constants,
        list_uninterpreted_constants,
        list_registers,
    ):
        all = list_registers + list_uninterpreted_constants + list_interpreted_constants
        all = all + all
        if index < (len(all)) / 2:
            if all[index] in list_interpreted_constants:
                return float(curr) > all[index]
            else:
                return curr > Real(all[index])
        elif index >= (len(all)) / 2 and index != len(all):
            if all[index] in list_interpreted_constants:
                return float(curr) >= all[index]
            else:
                return curr >= Real(all[index])
        else:
            return True

    def translate_enum_2_curr(
        self,
        index,
        curr,
        list_interpreted_constants,
        list_uninterpreted_constants,
        list_registers,
    ):
        all = list_registers + list_uninterpreted_constants + list_interpreted_constants
        all = all + all
        if index < (len(all)) / 2:
            if all[index] in list_interpreted_constants:
                return curr > all[index]
            else:
                return curr > Real(all[index])
        elif index >= (len(all)) / 2 and index != len(all):
            if all[index] in list_interpreted_constants:
                return curr >= all[index]
            else:
                return curr >= Real(all[index])
        else:
            return curr == curr

    def translate_enum_3(self, index, curr, list_registers):
        d_v = int(index / (len(list_registers) + 1))
        m_v = index % (len(list_registers) + 1)
        if len(list_registers) == 1:
            if m_v == len(list_registers):
                return Real(f"{list_registers[d_v]}_{curr}") == curr
            else:
                return Real(f"{list_registers[d_v]}_{curr}") == Real(
                    list_registers[m_v]
                )
        else:
            fls = [
                Real(f"{x}_{curr}") == Real(f"{x}")
                for x in list_registers
                if x != list_registers[d_v]
            ]
            if m_v == len(list_registers):
                return And(Real(f"{list_registers[d_v]}_{curr}") == curr, And(fls))
            else:
                return And(
                    Real(f"{list_registers[d_v]}_{curr}") == Real(list_registers[m_v]),
                    And(fls),
                )

    def build_form(self, pos, neg):
        z3.set_option(rational_to_decimal=True)
        z3.set_option(precision=6)
        res = []
        Enum1 = Datatype("Enum1")
        Enum2 = Datatype("Enum2")
        Enum3 = Datatype("Enum3")

        val_list = self.regs + list(self.uc) + list(self.ic)

        enum1s = (
            [f"curr<{i}" for i in val_list] + [f"curr<={i}" for i in val_list] + ["NU"]
        )
        for i in enum1s:
            Enum1.declare(i)

        Enum1 = Enum1.create()
        for i in range(len(enum1s)):
            for j in range(i + 1, len(enum1s)):
                res.append(Enum1.constructor(i)() != Enum1.constructor(j)())

        enum2s = (
            [f"curr>{i}" for i in val_list] + [f"curr>={i}" for i in val_list] + ["NL"]
        )

        for i in enum2s:
            Enum2.declare(i)

        Enum2 = Enum2.create()

        for i in range(len(enum2s)):
            for j in range(i + 1, len(enum2s)):
                res.append(Enum2.constructor(i)() != Enum2.constructor(j)())

        enum3s = []
        for i in self.regs:
            for j in self.regs:
                enum3s.append(f"{i}=={j}")
            enum3s.append(f"{i}==curr")

        for i in enum3s:
            Enum3.declare(i)

        Enum3 = Enum3.create()

        for i in range(len(enum3s)):
            for j in range(i + 1, len(enum3s)):
                res.append(Enum3.constructor(i)() != Enum3.constructor(j)())

        enum1 = {}
        enum2 = {}
        enum3 = {}

        states = [f"s{i}" for i in range(self._n)]  # size of automaton

        for p in states:
            for q in states:
                enum1[(p, q)] = Const(f"enum1_{p}_{q}", Enum1)
                enum2[(p, q)] = Const(f"enum2_{p}_{q}", Enum2)
                enum3[(p, q)] = Const(f"enum3_{p}_{q}", Enum3)

        # guard transition
        for p in states:
            res.append(
                Or(
                    [
                        And(
                            enum1[(p, q)] == Enum1.constructor(i)(),
                            enum2[(p, q)] == Enum2.constructor(j)(),
                            enum3[(p, q)] == Enum3.constructor(k)(),
                        )
                        for q in states
                        for i in range(len(enum1s))
                        for j in range(len(enum2s))
                        for k in range(len(enum3s))
                    ]
                )
            )

        # oliver deterministic part

        temp_soler = z3.Solver()

        def det_guard2(
            source_state,
            destination_state,
            third_state,
            enum1_index1,
            enum2_index1,
            enum1_index2,
            enum2_index2,
        ):
            temp_soler.push()
            curr = z3.Real("curr1")
            temp_soler.add(
                And(
                    self.translate_enum_1_curr(
                        enum1_index1, curr, self.ic, self.uc, self.regs
                    ),
                    self.translate_enum_2_curr(
                        enum2_index1, curr, self.ic, self.uc, self.regs
                    ),
                    self.translate_enum_1_curr(
                        enum1_index2, curr, self.ic, self.uc, self.regs
                    ),
                    self.translate_enum_2_curr(
                        enum2_index2, curr, self.ic, self.uc, self.regs
                    ),
                )
            )
            if temp_soler.check() == z3.sat:
                temp_soler.pop()
                return Implies(
                    And(
                        enum1[(source_state, destination_state)]
                        == Enum1.constructor(enum1_index1)(),
                        enum2[(source_state, destination_state)]
                        == Enum2.constructor(enum2_index1)(),
                        enum1[(source_state, third_state)]
                        == Enum1.constructor(enum1_index2)(),
                        enum2[(source_state, third_state)]
                        == Enum2.constructor(enum2_index2)(),
                    ),
                    False,
                )
            else:
                temp_soler.pop()
                """print("unsat")
                temp_soler.pop()
                return Implies(And(enum1[(source_state, destination_state)] == Enum1.constructor(enum1_index1)(),enum2[(source_state, destination_state)] == Enum2.constructor(enum2_index1)(),
                               enum1[(source_state, third_state)] == Enum1.constructor(enum1_index2)(),enum2[(source_state, third_state)] == Enum2.constructor(enum2_index2)()), True)"""
                return True

        # deterministic part
        if self.debug:
            print("--------deterministic part---------")
        for p in states:
            for q in states:
                for q_prime in states:
                    if q != q_prime:
                        for i in range(len(enum1s)):
                            for j in range(len(enum2s)):
                                for i1 in range(len(enum1s)):
                                    for j1 in range(len(enum2s)):
                                        res.append(
                                            det_guard2(p, q, q_prime, i, j, i1, j1)
                                        )

        if self.debug:
            print("===========prefix part===========")

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
                    if _tmp != "":
                        neg_prefix.append(self.conv_2_str(t))

        pos_prefix = list()
        for u in pos:
            for t in self.prefix_list(u):
                if self.conv_2_str(t) in pos_prefix:
                    pass
                else:
                    _tmp = self.conv_2_str(t)
                    if _tmp != "":
                        pos_prefix.append(self.conv_2_str(t))

        for t in pos_prefix:
            if self.debug:
                print(And(Or([Bool(f"x_{t}_{q}") for q in states])))
            res.append(And(Or([Bool(f"x_{t}_{q}") for q in states])))

        if self.debug:
            print("===========initial===========")
            print(Bool(f"x__s0"))

        res.append(Bool(f"x__s0"))

        for r in self.regs:
            if self.debug:
                print(Real(f"{r}") == 0)
            res.append(Real(f"{r}") == 0)

        pos_form, neg_form = self.divide(
            enum1s, enum2s, enum3s, self.ic, self.uc, self.regs
        )

        def transition_constraint(outgoing):
            flag = outgoing[0]
            num = outgoing[1]
            result = True
            if flag == "at_least":
                result = And(
                    result,
                    And(
                        [
                            AtMost(
                                *[
                                    And(
                                        enum1[("s" + str(m), "s" + str(e))]
                                        == Enum1.constructor(0)(),
                                        enum2[("s" + str(m), "s" + str(e))]
                                        == Enum2.constructor(0)(),
                                    )
                                    for e in range(self._n)
                                ],
                                self._n - num,
                            )
                            for m in range(self._n)
                        ]
                    ),
                )
            elif flag == "at_most":
                result = And(
                    result,
                    And(
                        [
                            AtLeast(
                                *[
                                    (
                                        And(
                                            enum1[("s" + str(m), "s" + str(e))]
                                            == Enum1.constructor(0)(),
                                            enum2[("s" + str(m), "s" + str(e))]
                                            == Enum2.constructor(0)(),
                                        )
                                    )
                                    for e in range(self._n)
                                ],
                                self._n - num,
                            )
                            for m in range(self._n)
                        ]
                    ),
                )

            else:
                result = And(
                    result,
                    And(
                        [
                            And(
                                (
                                    AtMost(
                                        *(
                                            [
                                                (
                                                    And(
                                                        enum1[
                                                            ("s" + str(m), "s" + str(e))
                                                        ]
                                                        == Enum1.constructor(0)(),
                                                        enum2[
                                                            ("s" + str(m), "s" + str(e))
                                                        ]
                                                        == Enum2.constructor(0)(),
                                                    )
                                                )
                                                for e in range(self._n)
                                            ]
                                        ),
                                        self._n - num,
                                    )
                                ),
                                (
                                    AtLeast(
                                        *(
                                            [
                                                (
                                                    And(
                                                        enum1[
                                                            ("s" + str(m), "s" + str(e))
                                                        ]
                                                        == Enum1.constructor(0)(),
                                                        enum2[
                                                            ("s" + str(m), "s" + str(e))
                                                        ]
                                                        == Enum2.constructor(0)(),
                                                    )
                                                )
                                                for e in range(self._n)
                                            ]
                                        ),
                                        self._n - num,
                                    )
                                ),
                            )
                            for m in range(self._n)
                        ]
                    ),
                )

            return result

        if self.debug:
            print(
                f"===========add_{self.outgoing[0]}_{self.outgoing[1]}_transitions part==========="
            )
            print(transition_constraint(self.outgoing))

        res.append(transition_constraint(self.outgoing))

            
        def add_clauses_new(q, q_, fs, enum1_inx, enum2_inx, enum3_inx , neg_form, r, u, a, t, debug):
            if debug:
                print(f"cur: {a} word:{t}")
            res.append(And(fs))
            if "NU" in str(fs[0]):
                if "NL" in str(fs[1]):
                    f1 = Implies(
                        And(
                            Bool(f"x_{self.conv_2_str(u)}_{q}"),
                            fs[0], fs[1], fs[2]
                        ),
                        And(
                            Bool(f"x_{t}_{q_}"),
                            self.translate_enum_3(enum3_inx, a, r),
                        ),
                    )
                    res.append(f1)
                    if debug:
                        print(f1)
                else:
                    if "=" in str(fs[1]):
                        f1 = Implies(
                            And(
                                Bool(f"x_{self.conv_2_str(u)}_{q}"),
                                fs[0], fs[1], fs[2],
                                self.translate_enum_2(
                                    enum2_inx, a, self.ic, self.uc, r
                                ),
                            ),
                            And(
                                Bool(f"x_{t}_{q_}"),
                                self.translate_enum_3(enum3_inx, a, r),
                            ),
                        )

                        f2 = Implies(
                            And(
                                Bool(f"x_{self.conv_2_str(u)}_{q}"),
                                fs[0], fs[1],
                                Not(
                                    self.translate_enum_2(
                                        enum2_inx, a, self.ic, self.uc, r
                                    )
                                ),
                            ),
                            (Not(Bool(f"x_{t}_{q_}"))),
                        )

                    else:
                        f1 = Implies(
                            And(
                                Bool(f"x_{self.conv_2_str(u)}_{q}"),
                                fs[0], fs[1], fs[2],
                                self.translate_enum_2(
                                    enum2_inx, a, self.ic, self.uc, r
                                ),
                            ),
                            And(
                                Bool(f"x_{t}_{q_}"),
                                self.translate_enum_3(enum3_inx, a, r),
                            ),
                        )

                        f2 = Implies(
                            And(
                                Bool(f"x_{self.conv_2_str(u)}_{q}"),
                                fs[0], fs[1],
                                Not(
                                    self.translate_enum_2(
                                        enum2_inx, a, self.ic, self.uc, r
                                    )
                                ),
                            ),
                            (Not(Bool(f"x_{t}_{q_}"))),
                        )
                    res.extend([f1, f2])
                    if debug:
                        print(f1)
                        print(f2)

            elif "=" not in str(fs[0]):
                f1 = Implies(
                    And(
                        Bool(f"x_{self.conv_2_str(u)}_{q}"),
                        fs[0], fs[1], fs[2],
                        self.translate_enum_1(
                            enum1_inx, a, self.ic, self.uc, r
                        ),
                        self.translate_enum_2(
                            enum2_inx, a, self.ic, self.uc, r
                        ),
                    ),
                    And(
                        Bool(f"x_{t}_{q_}"),
                        self.translate_enum_3(enum3_inx, a, r),
                    ),
                )

                f2 = Implies(
                    And(
                        Bool(f"x_{self.conv_2_str(u)}_{q}"),
                        fs[0], fs[1],
                        Or(
                            Not(
                                self.translate_enum_1(
                                    enum1_inx, a, self.ic, self.uc, r
                                )
                            ),
                            Not(
                                self.translate_enum_2(
                                    enum2_inx, a, self.ic, self.uc, r
                                )
                            ),
                        ),
                    ),
                    (Not(Bool(f"x_{t}_{q_}"))),
                )
                res.extend([f1, f2])
                if debug:
                    print(f1)
                    print(f2)
            else:
                if "NL" in str(fs[1]):
                    f1 = Implies(
                        And(
                            Bool(f"x_{self.conv_2_str(u)}_{q}"),
                            fs[0], fs[1], fs[2],
                            self.translate_enum_1(
                                enum1_inx, a, self.ic, self.uc, r
                            ),
                            # self.translate_enum_2(enum2_inx, a, self.ic, self.uc, r)
                        ),
                        And(
                            Bool(f"x_{t}_{q_}"),
                            self.translate_enum_3(enum3_inx, a, r),
                        ),
                    )

                    f2 = Implies(
                        And(
                            Bool(f"x_{self.conv_2_str(u)}_{q}"),
                            fs[0], fs[1],
                            Not(
                                self.translate_enum_1(
                                    enum1_inx, a, self.ic, self.uc, r
                                )
                            ),
                        ),
                        (Not(Bool(f"x_{t}_{q_}"))),
                    )

                else:
                    f1 = Implies(
                        And(
                            Bool(f"x_{self.conv_2_str(u)}_{q}"),
                            fs[0], fs[1], fs[2],
                            self.translate_enum_1(
                                enum1_inx, a, self.ic, self.uc, r
                            ),
                            self.translate_enum_2(
                                enum2_inx, a, self.ic, self.uc, r
                            ),
                        ),
                        And(
                            Bool(f"x_{t}_{q_}"),
                            self.translate_enum_3(enum3_inx, a, r),
                        ),
                    )

                    f2 = Implies(
                        And(
                            Bool(f"x_{self.conv_2_str(u)}_{q}"),
                            fs[0], fs[1],
                            Or(
                                Not(
                                    self.translate_enum_1(
                                        enum1_inx, a, self.ic, self.uc, r
                                    )
                                ),
                                Not(
                                    self.translate_enum_2(
                                        enum2_inx, a, self.ic, self.uc, r
                                    )
                                ),
                            ),
                        ),
                        (Not(Bool(f"x_{t}_{q_}"))),
                    )
                res.extend([f1, f2])
                if debug:
                    print(f1)
                    print(f2)

        def add_bot_clauses(q, q_, pos_form, neg_form, r, u, a, t, debug):   
                form = Implies(
                        And(
                            Bool(f"x_{self.conv_2_str(u)}_{q}"),
                                            enum1[(q, q_)]
                                            == Enum1.constructor(0)(),
                                            enum2[(q, q_)]
                                            == Enum2.constructor(0)(),
                        ),
                        (Not(Bool(f"x_{t}_{q_}"))),
                    )
                res.append(form)
                if self.debug:
                    print(f"bottom formula: {form}")
                 
        def add_clauses(q, q_, pos_form, neg_form, r, u, a, t, debug):
            if debug:
                print(f"cur: {a} word:{t}")

            for f in pos_form:
                if f[0] == "NU":
                    if f[1] == "NL":

                        f = Implies(
                            And(
                                Bool(f"x_{self.conv_2_str(u)}_{q}"),
                                enum1[(q, q_)]
                                == Enum1.constructor(enum1s.index(f[0]))(),
                                enum2[(q, q_)]
                                == Enum2.constructor(enum2s.index(f[1]))(),
                                enum3[(q, q_)]
                                == Enum3.constructor(enum3s.index(f[2]))(),
                            ),
                            And(
                                Bool(f"x_{t}_{q_}"),
                                self.translate_enum_3(enum3s.index(f[2]), a, r),
                            ),
                        )
                        res.append(f)
                        if debug:
                            print(f)
                    else:
                        if "=" in f[1]:
                            f1 = Implies(
                                And(
                                    Bool(f"x_{self.conv_2_str(u)}_{q}"),
                                    enum1[(q, q_)]
                                    == Enum1.constructor(enum1s.index(f[0]))(),
                                    enum2[(q, q_)]
                                    == Enum2.constructor(enum2s.index(f[1]))(),
                                    enum3[(q, q_)]
                                    == Enum3.constructor(enum3s.index(f[2]))(),
                                    self.translate_enum_2(
                                        enum2s.index(f[1]), a, self.ic, self.uc, r
                                    ),
                                ),
                                And(
                                    Bool(f"x_{t}_{q_}"),
                                    self.translate_enum_3(enum3s.index(f[2]), a, r),
                                ),
                            )

                            f2 = Implies(
                                And(
                                    Bool(f"x_{self.conv_2_str(u)}_{q}"),
                                    enum1[(q, q_)]
                                    == Enum1.constructor(enum1s.index(f[0]))(),
                                    enum2[(q, q_)]
                                    == Enum2.constructor(enum2s.index(f[1]))(),
                                    Not(
                                        self.translate_enum_2(
                                            enum2s.index(f[1]), a, self.ic, self.uc, r
                                        )
                                    ),
                                ),
                                (Not(Bool(f"x_{t}_{q_}"))),
                            )

                        else:
                            f1 = Implies(
                                And(
                                    Bool(f"x_{self.conv_2_str(u)}_{q}"),
                                    enum1[(q, q_)]
                                    == Enum1.constructor(enum1s.index(f[0]))(),
                                    enum2[(q, q_)]
                                    == Enum2.constructor(enum2s.index(f[1]))(),
                                    enum3[(q, q_)]
                                    == Enum3.constructor(enum3s.index(f[2]))(),
                                    self.translate_enum_2(
                                        enum2s.index(f[1]), a, self.ic, self.uc, r
                                    ),
                                ),
                                And(
                                    Bool(f"x_{t}_{q_}"),
                                    self.translate_enum_3(enum3s.index(f[2]), a, r),
                                ),
                            )

                            f2 = Implies(
                                And(
                                    Bool(f"x_{self.conv_2_str(u)}_{q}"),
                                    enum1[(q, q_)]
                                    == Enum1.constructor(enum1s.index(f[0]))(),
                                    enum2[(q, q_)]
                                    == Enum2.constructor(enum2s.index(f[1]))(),
                                    Not(
                                        self.translate_enum_2(
                                            enum2s.index(f[1]), a, self.ic, self.uc, r
                                        )
                                    ),
                                ),
                                (Not(Bool(f"x_{t}_{q_}"))),
                            )
                        res.extend([f1, f2])
                        if debug:
                            print(f1)
                            print(f2)

                elif "=" not in f[0]:
                    f1 = Implies(
                        And(
                            Bool(f"x_{self.conv_2_str(u)}_{q}"),
                            enum1[(q, q_)] == Enum1.constructor(enum1s.index(f[0]))(),
                            enum2[(q, q_)] == Enum2.constructor(enum2s.index(f[1]))(),
                            enum3[(q, q_)] == Enum3.constructor(enum3s.index(f[2]))(),
                            self.translate_enum_1(
                                enum1s.index(f[0]), a, self.ic, self.uc, r
                            ),
                            self.translate_enum_2(
                                enum2s.index(f[1]), a, self.ic, self.uc, r
                            ),
                        ),
                        And(
                            Bool(f"x_{t}_{q_}"),
                            self.translate_enum_3(enum3s.index(f[2]), a, r),
                        ),
                    )

                    f2 = Implies(
                        And(
                            Bool(f"x_{self.conv_2_str(u)}_{q}"),
                            enum1[(q, q_)] == Enum1.constructor(enum1s.index(f[0]))(),
                            enum2[(q, q_)] == Enum2.constructor(enum2s.index(f[1]))(),
                            Or(
                                Not(
                                    self.translate_enum_1(
                                        enum1s.index(f[0]), a, self.ic, self.uc, r
                                    )
                                ),
                                Not(
                                    self.translate_enum_2(
                                        enum2s.index(f[1]), a, self.ic, self.uc, r
                                    )
                                ),
                            ),
                        ),
                        (Not(Bool(f"x_{t}_{q_}"))),
                    )
                    res.extend([f1, f2])
                    if debug:
                        print(f1)
                        print(f2)
                else:
                    if f[1] == "NL":
                        f1 = Implies(
                            And(
                                Bool(f"x_{self.conv_2_str(u)}_{q}"),
                                enum1[(q, q_)]
                                == Enum1.constructor(enum1s.index(f[0]))(),
                                enum2[(q, q_)]
                                == Enum2.constructor(enum2s.index(f[1]))(),
                                enum3[(q, q_)]
                                == Enum3.constructor(enum3s.index(f[2]))(),
                                self.translate_enum_1(
                                    enum1s.index(f[0]), a, self.ic, self.uc, r
                                ),
                                # self.translate_enum_2(enum2s.index(f[1]), a, self.ic, self.uc, r)
                            ),
                            And(
                                Bool(f"x_{t}_{q_}"),
                                self.translate_enum_3(enum3s.index(f[2]), a, r),
                            ),
                        )

                        f2 = Implies(
                            And(
                                Bool(f"x_{self.conv_2_str(u)}_{q}"),
                                enum1[(q, q_)]
                                == Enum1.constructor(enum1s.index(f[0]))(),
                                enum2[(q, q_)]
                                == Enum2.constructor(enum2s.index(f[1]))(),
                                Not(
                                    self.translate_enum_1(
                                        enum1s.index(f[0]), a, self.ic, self.uc, r
                                    )
                                ),
                            ),
                            (Not(Bool(f"x_{t}_{q_}"))),
                        )

                    else:
                        f1 = Implies(
                            And(
                                Bool(f"x_{self.conv_2_str(u)}_{q}"),
                                enum1[(q, q_)]
                                == Enum1.constructor(enum1s.index(f[0]))(),
                                enum2[(q, q_)]
                                == Enum2.constructor(enum2s.index(f[1]))(),
                                enum3[(q, q_)]
                                == Enum3.constructor(enum3s.index(f[2]))(),
                                self.translate_enum_1(
                                    enum1s.index(f[0]), a, self.ic, self.uc, r
                                ),
                                self.translate_enum_2(
                                    enum2s.index(f[1]), a, self.ic, self.uc, r
                                ),
                            ),
                            And(
                                Bool(f"x_{t}_{q_}"),
                                self.translate_enum_3(enum3s.index(f[2]), a, r),
                            ),
                        )

                        f2 = Implies(
                            And(
                                Bool(f"x_{self.conv_2_str(u)}_{q}"),
                                enum1[(q, q_)]
                                == Enum1.constructor(enum1s.index(f[0]))(),
                                enum2[(q, q_)]
                                == Enum2.constructor(enum2s.index(f[1]))(),
                                Or(
                                    Not(
                                        self.translate_enum_1(
                                            enum1s.index(f[0]), a, self.ic, self.uc, r
                                        )
                                    ),
                                    Not(
                                        self.translate_enum_2(
                                            enum2s.index(f[1]), a, self.ic, self.uc, r
                                        )
                                    ),
                                ),
                            ),
                            (Not(Bool(f"x_{t}_{q_}"))),
                        )
                    res.extend([f1, f2])
                    if debug:
                        print(f1)
                        print(f2)

            for f in neg_form:
                if self.has_numbers(f[0]) and self.has_numbers(f[1]):
                    if self.extract_register(f[0]) != self.extract_register(f[1]):
                        f1 = Implies(
                            And(
                                Bool(f"x_{self.conv_2_str(u)}_{q}"),
                                enum1[(q, q_)]
                                == Enum1.constructor(enum1s.index(f[0]))(),
                                enum2[(q, q_)]
                                == Enum2.constructor(enum2s.index(f[1]))(),
                                enum3[(q, q_)]
                                == Enum3.constructor(enum3s.index(f[2]))(),
                                self.translate_enum_1(
                                    enum1s.index(f[0]), a, self.ic, self.uc, r
                                ),
                                self.translate_enum_2(
                                    enum2s.index(f[1]), a, self.ic, self.uc, r
                                ),
                            ),
                            And(
                                Bool(f"x_{t}_{q_}"),
                                self.translate_enum_3(enum3s.index(f[2]), a, r),
                            ),
                        )

                        res.append(f1)
                        form = Implies(
                            And(
                                Bool(f"x_{self.conv_2_str(u)}_{q}"),
                                enum1[(q, q_)]
                                == Enum1.constructor(enum1s.index(f[0]))(),
                                enum2[(q, q_)]
                                == Enum2.constructor(enum2s.index(f[1]))(),
                                Or(
                                    Not(
                                        self.translate_enum_1(
                                            enum1s.index(f[0]), a, self.ic, self.uc, r
                                        )
                                    ),
                                    Not(
                                        self.translate_enum_2(
                                            enum2s.index(f[1]), a, self.ic, self.uc, r
                                        )
                                    ),
                                ),
                            ),
                            (Not(Bool(f"x_{t}_{q_}"))),
                        )
                    else:
                        form = Implies(
                            And(
                                Bool(f"x_{self.conv_2_str(u)}_{q}"),
                                enum1[(q, q_)]
                                == Enum1.constructor(enum1s.index(f[0]))(),
                                enum2[(q, q_)]
                                == Enum2.constructor(enum2s.index(f[1]))(),
                            ),
                            (Not(Bool(f"x_{t}_{q_}"))),
                        )
                else:
                    form = Implies(
                        And(
                            Bool(f"x_{self.conv_2_str(u)}_{q}"),
                            enum1[(q, q_)] == Enum1.constructor(enum1s.index(f[0]))(),
                            enum2[(q, q_)] == Enum2.constructor(enum2s.index(f[1]))(),
                        ),
                        (Not(Bool(f"x_{t}_{q_}"))),
                    )

                res.append(form)
                if debug:
                    print(form)

            if self.debug:
                print(
                    Implies(
                        And([Bool(f"x_{t}_r_{q}") for q in states]), Bool(f"x_{t}_r")
                    )
                )

        if self.debug:
            print("===========sample part===========")

        for k, v in self.saved_formulas.items():
            q = k[0]
            q_ = k[1]
            
            for p in pos + neg:
                if self.debug:
                    print(f"new pos sample {p}")
                ps = self.prefix_list(p)
                rs = self.regs
                for e in ps:
                    t = self.conv_2_str(e)
                    if t != "":
                        tmp_list = t.split("_")
                        a = tmp_list[-1]
                        u = tmp_list[:-1]
                        add_clauses_new(q, q_, v[:3], v[3], v[4], v[5], neg_form, rs, u, a, t, self.debug)
                        rs = [f"{i}_{a}" for i in rs]
                    if q != q_:
                        res.append(
                            Or(Not(Bool(f"x_{t}_{q}")), Not(Bool(f"x_{t}_{q_}")))
                        )
                        if self.debug:
                            print(
                                f"sampe {q} {q_}:",
                                Or(
                                    Not(Bool(f"x_{t}_{q}")),
                                    Not(Bool(f"x_{t}_{q_}")),
                                ),
                            )

        for tuple_states in self.bottom_transtions:
            q = tuple_states[0]
            q_ = tuple_states[1]
            form = And(
                                            enum1[(q, q_)]
                                            == Enum1.constructor(0)(),
                                            enum2[(q, q_)]
                                            == Enum2.constructor(0)(),
                                        )
            if self.debug:
                print(f"bottom transition: {form}")
            for p in pos + neg:
                if self.debug:
                    print(f"new pos sample {p}")
                ps = self.prefix_list(p)
                rs = self.regs
                for e in ps:
                    t = self.conv_2_str(e)
                    if t != "":
                        tmp_list = t.split("_")
                        a = tmp_list[-1]
                        u = tmp_list[:-1]
                        add_bot_clauses(q, q_, pos_form, neg_form, rs, u, a, t, self.debug)
                        rs = [f"{i}_{a}" for i in rs]

                    if q != q_:
                        res.append(
                            Or(Not(Bool(f"x_{t}_{q}")), Not(Bool(f"x_{t}_{q_}")))
                        )
                        if self.debug:
                            print(
                                f"sampe {q} {q_}:",
                                Or(
                                    Not(Bool(f"x_{t}_{q}")),
                                    Not(Bool(f"x_{t}_{q_}")),
                                ),
                            )
            res.append(form)
        for i, tuple_states in enumerate(self.search_transitions):
            q = tuple_states[0]
            q_ = tuple_states[1]        
            percent = (i + 1) / len(self.search_transitions) * 100
            # print(f"Learner Progress: {percent:.2f}%", end='\r')
            
            for p in pos + neg:
                if self.debug:
                    print(f"new pos sample {p}")
                ps = self.prefix_list(p)
                rs = self.regs
                for e in ps:
                    t = self.conv_2_str(e)
                    if t != "":
                        tmp_list = t.split("_")
                        a = tmp_list[-1]
                        u = tmp_list[:-1]
                        add_clauses(q, q_, pos_form, neg_form, rs, u, a, t, self.debug)
                        rs = [f"{i}_{a}" for i in rs]
                    if q != q_:
                        res.append(
                            Or(Not(Bool(f"x_{t}_{q}")), Not(Bool(f"x_{t}_{q_}")))
                        )
                        if self.debug:
                            print(
                                f"sampe {q} {q_}:",
                                Or(
                                    Not(Bool(f"x_{t}_{q}")),
                                    Not(Bool(f"x_{t}_{q_}")),
                                ),
                            )

        # positive examples constraint
        if self.debug:
            print("=======positive examples======")
        for u in pos:
            pos_tmp = []
            for i in self.prefix_list(u):
                if i != []:
                    pos_tmp.append(i)
            pos_form = [
                True
            ]  # [Not(Bool(f"x_{self.conv_2_str(i)}_r")) for i in pos_tmp]
            for q in states:
                pos_form.append(
                    Implies(Bool(f"x_{self.conv_2_str(u)}_{q}"), Bool(f"f_{q}"))
                )
                if self.debug:
                    print(And(pos_form))
                res.append(And(pos_form))

        if self.debug:
            print("=======negative examples======")

        # negative examples constraint
        for u in neg:
            neg_tmp = []
            for i in self.prefix_list(u):
                if i != []:
                    neg_tmp.append(i)
            neg_form = [(Bool(f"x_{self.conv_2_str(i)}_r")) for i in neg_tmp]
            for q in states:
                if self.debug:
                    print(
                        Implies(
                            Bool(f"x_{self.conv_2_str(u)}_{q}"), Not(Bool(f"f_{q}"))
                        )
                    )
                res.append(
                    Implies(Bool(f"x_{self.conv_2_str(u)}_{q}"), Not(Bool(f"f_{q}")))
                )
        # print("---------------")
        return res

    def parse_input_string(self, input_string, model):
        # Regular expression pattern to match the input string
        pattern = r"(.+?)\s*(==|<=|<|>=|>)\s*(.+)"

        # Using regex to match the pattern
        match = re.match(pattern, input_string)

        if match:
            # Extracting left expression, operator, and right expression
            left_expression = match.group(1).strip()
            operator = match.group(2).strip()
            right_expression = match.group(3).strip()
            left_expression = self.convert_str_to_z3(left_expression)
            right_expression = self.convert_str_to_z3(right_expression)
            if str(left_expression) in self.uc:
                left_expression = model[left_expression]
            if str(right_expression) in self.uc:
                right_expression = model[right_expression]
            if operator == ">=":
                return left_expression >= right_expression
            elif operator == ">":
                return left_expression > right_expression
            elif operator == "<=":
                return left_expression <= right_expression
            elif operator == "<":
                return left_expression < right_expression
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
        s.append(len(s) + 1)
        if len(rs) == 0:
            rs = [["r0"]]
        else:
            n_rs = rs[-1] + [f"r{len(rs[-1])}"]
            rs.append(n_rs)
        if len(cs) == 0:
            cs = [["c0"]]
        else:
            n_cs = cs[-1] + [f"c{len(cs[-1])}"]
            cs.append(n_cs)
        return s, rs, cs

    def get_model(self, timeout=3600):
        """
        Attempt to get a model from the solver, with a timeout in seconds (default 3600s = 60min).
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            tmp_pos = []
            tmp_neg = []
            inx = 0
            for i in range(0, len(self.pos) // self.batch_size):
                # if i == 0:
                #     tmp_pos.extend(self.pos[0:1])
                #     tmp_neg.extend(self.neg[0:1])
                # else:
                print(i)
                pre_inx = i * self.batch_size
                post_inx = (i+1) * self.batch_size
                tmp_pos.extend(
                    self.pos[pre_inx : post_inx]
                )
                tmp_neg.extend(
                    self.neg[pre_inx : post_inx]
                )
                res_form = self.build_form(tmp_pos, tmp_neg)
                self._solver.add(*res_form)
                if self._solver.check() == z3.unsat:
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
                    auto =  register_automata.ra.RegisterAutomata(
                        states, transitions, register_vars, final_states
                    )
                    
                    auto = register_automata.ra.RegisterAutomata(
                        states, transitions, register_vars, final_states
                    )
                    acc = self.consistence_check(auto, self.pos, self.neg)
                    print(f"Batch {i}:", acc, 'time used:', time.time() - start_time)
                    if acc > 0.9 or i == 6:
                        return auto, i
                    else:
                        continue
        else:         
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
                count1 += 1

        for i in neg:
            if aut.run(i):
                neg_err.append(i)
                count2 += 1

        return ((len(pos) - count1) / len(pos) + (len(neg) - count2) / len(neg)) / 2

