import numpy as np
from typing import List, Tuple
import argparse
import sys
from typing import List, Optional
import os
import csv


class Language:
    def __init__(self, num_pos=0, num_neg=0, max_len=0):
        self.pos = []
        self.neg = []
        self.transition_matrix = None
        self.pos, self.neg = ([], [])#self.generate_samples(num_pos, num_neg, max_len)
        self.states = 0
        self.ic = []
        self.uc = []
        self.regs = ["r0"]
        self.edge_limit_type = 'exactly'
        self.edge_limit_num = 1

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pass

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        pass

    def consecutive_pairs(self, input_list: List[int]) -> List[Tuple[int, int]]:
        return [(input_list[i], input_list[i + 1]) for i in range(len(input_list) - 1)]



class L1(Language):
    def __init__(self, num_pos, num_neg, max_len):

        super().__init__(num_pos, num_neg, max_len)
        self.states = 2

    def generate_samples(self, num_pos, num_neg, max_length):
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, nl = self.generate_sequence(max_length)
            while not (nl == 0) or sequence in pos_sample:
                sequence, nl = self.generate_sequence(max_length)
            pos_sample.append(sequence)
        for i in range(num_neg):
            sequence, nl = self.generate_sequence(max_length)
            while (nl == 0) or sequence in neg_sample or len(sequence) <= 1:
                sequence, nl = self.generate_sequence(max_length)
            neg_sample.append(sequence)
        return (pos_sample, neg_sample)

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.99, 0.01],
                [0.0, 1],
            ]
        )

        current_state = 0
        length = np.random.randint(1, max_length + 1)

        # Generate state transitions
        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = (
                    valid_probs[valid_states] / valid_probs[valid_states].sum()
            )
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state

        # Generate sequence based on pairs of states
        ts = self.consecutive_pairs(states)
        a = np.random.randint(-200, 200)
        for t in ts:
            if t == (0, 0):
                sequence.append(a)
            elif t == (0, 1):
                b = np.random.randint(-200, 200)
                while a == b:
                    b = np.random.randint(-200, 200)
                sequence.append(b)
            else:
                b = np.random.randint(-200, 200)
                sequence.append(b)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0


        
class L2(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):

        super().__init__(num_pos, num_neg, max_len)
        self.states = 4
        self.regs = ['r0', 'r1']

    def is_l2_language(self, lst: List[float]) -> bool:
        if len(set(lst)) == 2:
            if lst[0] != lst[1]:
                return lst == [lst[0], lst[1]] * (len(lst) // 2)
            else:
                return False
        elif lst == []:
            return True
        else:
            return False

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []

        for inx in range(num_pos):
            sequence, nl = self.generate_sequence(max_length)
            while not self.is_l2_language(sequence) or sequence in pos_sample:
                sequence, nl = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            
        for inx in range(num_neg):
            sequence, nl = self.generate_sequence(max_length)
            while self.is_l2_language(sequence) or sequence in neg_sample:
                sequence, nl = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            
        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0, 0, 0.95, 0.05],
                [0.0, 0.95, 0.0, 0.05],
                [0, 0.0, 0.0, 1.0],
            ]
        )

        current_state = 0
        length = np.random.randint(2, max_length + 1)
        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state

        a = np.random.randint(-200, 200)
        b = np.random.randint(-200, 200)
        while (b == a):
            b = np.random.randint(-200, 200)

        ts = self.consecutive_pairs(states)
        for t in ts:
            if t == (0, 1):
                sequence.append(a)
            elif t == (1, 2):
                sequence.append(b)
            elif t == (1, 3):
                c = np.random.randint(-200, 200)
                sequence.append(c)
            elif t == (2, 3):
                f = np.random.randint(-200, 200)
                sequence.append(f)
            elif t == (2, 1):
                sequence.append(a)
            else:
                f = np.random.randint(-200, 200)
                sequence.append(f)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0



        
class L3(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):

        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 4
        self.edge_limit_num = 2
        self.edge_limit_type = "exactly"

    def is_l3_language(self, lst: List[float]) -> bool:
        if not lst:
            return False

        # Preserve order of first appearance to define (a, b)
        seen = []
        for x in lst:
            if x not in seen:
                seen.append(x)
                if len(seen) > 2:
                    # More than two distinct symbols => reject
                    return False

        if len(seen) == 1:
            # Unary case: accept iff odd length
            return (len(lst) % 2) == 1

        # Exactly two distinct symbols: first is 'a', second is 'b'
        a, b = seen[0], seen[1]
        count_a = sum(1 for x in lst if x == a)
        count_b = sum(1 for x in lst if x == b)
        flag1 =  (count_a % 2 == 1) and (count_b % 2 == 0)
        
        a, b = seen[1], seen[0]
        count_a = sum(1 for x in lst if x == a)
        count_b = sum(1 for x in lst if x == b)
        flag2 =  (count_a % 2 == 1) and (count_b % 2 == 0)
        return flag1 or flag2
    
    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for inx in range(num_pos):
            sequence, nl,a,b = self.generate_sequence(max_length)
            while not (self.is_l3_language(sequence)) or sequence in pos_sample:
                sequence, nl,a,b = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            
        for i in range(num_neg):

            sequence, nl,a,b = self.generate_sequence(max_length)

            while (self.is_l3_language(sequence)) or sequence in neg_sample:
                sequence, nl,a,b = self.generate_sequence(max_length)
            
            neg_sample.append(sequence)
        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 0.5, 0, 0.5], 
                [0.5, 0, 0.5, 0],
                [0, 0.5, 0, 0.5],
                [0.5, 0, 0.5, 0],
            ]
        )

        current_state = 0
        length = np.random.randint(1, max_length + 1)

        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state

        a = np.random.randint(-200, 200)
        b = np.random.randint(-200, 200)
        while b == a:
            b = np.random.randint(-200, 200)
        ts = self.consecutive_pairs(states)
        for t in ts:
            if t == (0, 1):
                sequence.append(b)
            elif t == (0, 3):
                sequence.append(a)
            elif t == (1, 0):
                sequence.append(b)
            elif t == (1, 2):
                sequence.append(a)
            elif t == (2, 3):
                sequence.append(b)
            elif t == (2, 1):
                sequence.append(a)
            elif t == (3, 2):
                sequence.append(b)
            elif t == (3, 0):
                sequence.append(a)
            else:
                print(t)
                c = np.random.randint(-200, 200)
                sequence.append(c)

        if ts:
            return sequence, ts[-1][1],a,b
        else:
            return sequence, 0, a,b




class L4(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):

        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0', 'r1']
        self.states = 4
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def is_l4_language(self, sequence: List[float]) -> bool:
        for i in range(0, len(sequence) - 2):
            if sequence[i] == sequence[i + 1] and  sequence[i + 1] == sequence[i + 2]:
                return False
        return True

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for inx in range(num_pos):
            sequence, nl = self.generate_sequence(max_length)
            while not self.is_l4_language(sequence) or sequence in pos_sample:
                sequence, nl = self.generate_sequence(max_length)

            pos_sample.append(sequence)
            print(1, inx)
        for inx in range(num_neg):
            sequence, nl = self.generate_sequence(max_length)
            while self.is_l4_language(sequence) or sequence in neg_sample:
                sequence, nl = self.generate_sequence(max_length)

            neg_sample.append(sequence)
            print(2, inx)
        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 1, 0.0, 0.0],
                [0, 0.5, 0.5, 0.0],
                [0, 0.5, 0, 0.5],
                [0, 0, 0.0, 1],
            ]
        )

        current_state = 0
        length = np.random.randint(1, max_length + 1)

        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state

        r0 = np.random.randint(-20, 20)
        # b = np.random.randint(-200, 200)
        # while a ==b :
        #     b = np.random.randint(-200, 200)

        ts = self.consecutive_pairs(states)
        for t in ts:
            if t == (0, 1):
                sequence.append(r0)
            elif t == (1, 1):
                c = np.random.randint(-20, 20) 
                while c == r0:
                    c = np.random.randint(-20, 20)
                r0 = c
                sequence.append(c)
                # a = b
            elif t == (1, 2):
                sequence.append(r0)
            elif t == (2, 3):
                # b = np.random.randint(-200, 200)
                # while a == b:
                #     b = np.random.randint(-200, 200)
                sequence.append(r0)
                # a = b
            elif t == (2, 1):
                c = np.random.randint(-20, 20) 
                while c == r0:
                    c = np.random.randint(-20, 20)
                r0 = c
                sequence.append(c)
            elif t == (2, 3):
                sequence.append(r0)
            elif t == (3, 3):
                c = np.random.randint(-20, 20)
                sequence.append(c)
            else:
                print("Here")

        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0



class L5(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 4
        self.edge_limit_num = 2
        self.edge_limit_type = "exactly"

    def is_l5_language(self, sequence: List[float]) -> bool:
        if len(set(sequence)) == 1:
            return len(sequence) % 2 == 0
        elif len(set(sequence)) > 2 or len(set(sequence)) == 0:
            return False
        a, b = list(set(sequence))[:2]
        if sequence.count(a) % 2 ==0 and sequence.count(b) % 2 == 0:
            return True
        else:
            return False

    def generate_samples(self, num_pos: int, num_neg: int, max_length: int) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, nl = self.generate_sequence(max_length)
            while not self.is_l5_language(sequence) or sequence in pos_sample:
                sequence, nl = self.generate_sequence(max_length)
            print(1, i)
            pos_sample.append(sequence)
        for i in range(num_neg):
            sequence, nl = self.generate_sequence(max_length)
            while self.is_l5_language(sequence) or sequence in neg_sample:
                sequence, nl = self.generate_sequence(max_length)
            print(2, i)
            neg_sample.append(sequence)
            
        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array([
            [0.0, 0.5, 0.0, 0.5],
            [0.5, 0.0, 0.5, 0],
            [0.0, 0.5, 0.0, 0.5],
            [0.5, 0, 0.5, 0],

        ])

        current_state = 0
        length = np.random.randint(1, max_length + 1)

        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state

        a = np.random.randint(-10, 10)
        b = np.random.randint(-10, 10)
        while a == b:
            b = np.random.randint(-10, 10)
        ts = self.consecutive_pairs(states)
        for t in ts:
            if t == (0, 1):
                sequence.append(b)
            elif t == (0, 3):
                sequence.append(a)
            elif t == (1, 0):
                sequence.append(b)
            elif t == (1, 2):
                sequence.append(a)
            elif t == (2, 1):
                sequence.append(a)
            elif t == (2, 3):
                sequence.append(b)
            elif t == (3, 2):
                sequence.append(b)
            elif t == (3, 0):
                sequence.append(a)
            else:
                # c = np.
                # sequence.append(c)
                print(123)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0

class L6(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ["r0", "r1"]
        self.states = 3
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def is_l6_language(self, seq: list) -> bool:
        """
        L6: Strings where (# of 1s - # of 0s) ≡ 0 (mod 3).
        Works for any sequence with only two unique symbols.
        """
        if not seq:
            return True  # empty string: 0-0 = 0

        unique = list(set(seq))
        if len(unique) == 1:
            return len(seq) % 3 == 0  # all same symbol
        elif len(unique) == 2:
            a = seq[0]
            for i in seq:
                if i != a:
                    b = i
                    break
            count_a = seq.count(a)
            count_b = seq.count(b)
            return abs((count_a - count_b)) % 3 == 0
        else:
            return False


    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for inx in range(num_pos):
            sequence, last_node = self.generate_sequence(max_length)
            while (
                    not self.is_l6_language(sequence)
                    or sequence in pos_sample
            ):
                sequence, last_node = self.generate_sequence(max_length)

            # print(1, inx)
            pos_sample.append(sequence)
        for inx in range(num_neg):
            sequence, last_node = self.generate_sequence(max_length)
            while self.is_l6_language(sequence) or sequence in neg_sample:
                sequence, last_node = self.generate_sequence(max_length)
            # print(2 ,inx)
            neg_sample.append(sequence)
        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 0.5, 0.5],
                [0.5, 0, 0.5],
                [0.5, 0.5, 0],


            ]
        )

        current_state = 0
        length = np.random.randint(1, max_length + 1)

        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state

        a =np.random.randint(0, 20)
        b =np.random.randint(0, 20)
        while (a == b):
            b =np.random.randint(0, 15)
        ts = self.consecutive_pairs(states)
        # 0 -> read first a
        for t in ts:
            # 1 -> (1,0) await b
            if t == (0, 1):
                sequence.append(b)
            # 2 -> (2,0) await b
            elif t == (0, 2):
                sequence.append(a)
            # 4 -> (1,1)
            # elif t == (0, 3):
            #     c = np.random.randint(0, 15)
            #     while c == a or c == b:
            #         c = np.random.randint(0, 15)
            #     sequence.append(c)
            elif t == (1, 0):
                sequence.append(a)
            elif t == (1, 2):
                sequence.append(b)
            # elif t == (1,3):
            #     c = np.random.randint(0, 15)
            #     while c == a or c == b:
            #         c = np.random.randint(0, 15)
            #     sequence.append(c)
            elif t == (2, 0):
                sequence.append(b)
            # 1 -> (1,0) await b
            elif t == (2, 1):
                sequence.append(a)
            # elif t == (2, 3):
            #     c = np.random.randint(0, 15)
            #     while c == a or c == b:
            #         c = np.random.randint(0, 15)
            #     sequence.append(c)
            # 5 -> (0,1)
            else:
                tmp = np.random.randint(0, 20)
                while tmp == a or tmp == b:
                    tmp = np.random.randint(0, 20)
                sequence.append(tmp)
        #print(sequence, "tsts", ts[-1][1], ts[-1])
        return sequence, ts[-1][1] if ts else 0

class L7(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):

        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 5
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def matches_pattern(self, lst: List[float], eps: float = 1e-9) -> bool:
        """
        Accepts sequences of the form a* b* a* b* over at most two distinct values.
        - The first element defines 'a'
        - The first different value defines 'b'
        - Empty list is accepted (matches a*)
        - Equality between floats uses |x - y| <= eps
        """
        if not lst:
            return True

        def same(x: float, y: float) -> bool:
            return abs(x - y) <= eps

        a = lst[0]
        b: Optional[float] = None

        # states: 0 = in first a*, 1 = in first b*, 2 = in second a*, 3 = in final b*
        state = 0

        for x in lst:
            if same(x, a):
                if state == 0:
                    # stay in first a*
                    pass
                elif state == 1:
                    # b* -> a*
                    state = 2
                elif state == 2:
                    # stay in second a*
                    pass
                else:  # state == 3
                    # cannot go back to a after final b*
                    return False
            else:
                # x != a, must be b (and only b)
                if b is None:
                    b = x  # first different value becomes b
                elif not same(x, b):
                    # third distinct value -> reject
                    return False

                # x == b
                if state == 0:
                    # a* -> b*
                    state = 1
                elif state == 1:
                    # stay in first b*
                    pass
                elif state == 2:
                    # a* -> final b*
                    state = 3
                else:  # state == 3
                    # stay in final b*
                    pass

        return True


    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []

        for i in range(num_pos):
            sequence, ln = self.generate_sequence(max_length)
            while not (self.matches_pattern(sequence)) or sequence in pos_sample:
                sequence, ln = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            print(1, i)

        for i in range(num_neg):
            sequence, ln = self.generate_sequence(max_length)
            while (self.matches_pattern(sequence)) or sequence in neg_sample:
                sequence, ln = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            print(2, i)
            
        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.5,0.5, 0, 0, 0],
                [0, 0.5,0.5, 0, 0],
                [0, 0, 0.5,0.5, 0],
                [0, 0, 0, 0.9, 0.1],
                [0, 0, 0, 0, 1],
            ]
        )

        current_state = 0
        length = np.random.randint(1, max_length + 1)

        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state

        a = np.random.randint(0, 200)
        b = np.random.randint(0, 200)
        while a == b:
            b = np.random.randint(0, 200)
        ts = self.consecutive_pairs(states)
        for t in ts:
            if t == (0, 0):
                sequence.append(a)
            elif t == (0, 1):
                sequence.append(b)
            elif t == (1, 1):
                # b = np.random.uniform(a, 10)
                sequence.append(b)
            elif t == (1, 2):
                # b = np.random.uniform(0, a)
                # while b == a:
                #     b = np.random.uniform(0, a)
                sequence.append(a)
            elif t == (2, 2):
                sequence.append(a)
            elif t == (2, 3):
                sequence.append(b)
            elif t == (3, 3):
                sequence.append(b)
            elif t == (3, 4):
                sequence.append(a)
            elif t == (4, 4):
                c = np.random.choice([a, b])
                sequence.append(c)
            # elif t == (3, 4):
            #     sequence.append(b)
            # elif t == (3, 7):
            #     tmp = np.random.randint(-200, 200)
            #     while tmp == a or tmp == b:
            #         tmp = np.random.randint(-200, 200)
            #     sequence.append(tmp)
            # elif t == (4, 4):
            #     sequence.append(b)
            # elif t == (4, 8):
            #     tmp = np.random.randint(-200, 200)
            #     while tmp == b:
            #         tmp = np.random.randint(-200, 200)
            #     sequence.append(tmp)
            # else:
            #     tmp = np.random.randint(-200, 200)
            #     sequence.append(tmp)
        return sequence, ts[-1][1] if ts else 0





class S1(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 2
        self.edge_limit_num = 1
        self.edge_limit_type = "exactly"

    def is_increasing_array(self, arr: List[float]) -> bool:
        for i in range(1, len(arr)):
            if arr[i] <= arr[i - 1]:
                return False
        return True

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for inx in range(num_pos):
            sequence = self.generate_sequence(max_length)
            while not self.is_increasing_array(sequence) or sequence in pos_sample:
                sequence = self.generate_sequence(max_length)
            
            pos_sample.append(sequence)
        for i in range(num_neg):
            sequence = self.generate_sequence(max_length)
            while self.is_increasing_array(sequence) or sequence in neg_sample:
                sequence = self.generate_sequence(max_length)
            
            neg_sample.append(sequence)
        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> List[float]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.98, 0.02],
                [0.0, 1],
            ]
        )

        current_state = 0
        length = np.random.randint(1, max_length + 1)

        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state

        a = np.random.randint(0,10)
        ts = self.consecutive_pairs(states)
        for t in ts:
            if t == (0, 0):
                inc = np.random.randint(5, 20)
                a +=  inc
                sequence.append(a)
            else:  
                b = a - np.random.randint(0, 20)
                sequence.append(b)
             
        return sequence


class S2(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 2
        self.edge_limit_num = 1
        self.edge_limit_type = "exactly"

    def is_decreasing_array(self, arr: List[float]) -> bool:
        for i in range(1, len(arr)):
            if arr[i] >= arr[i - 1]:
                return False
        return True

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for inx in range(num_pos):
            sequence = self.generate_sequence(max_length)
            while not self.is_decreasing_array(sequence) or sequence in pos_sample:
                sequence = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            

        for inx in range(num_neg):
            sequence = self.generate_sequence(max_length)
            while self.is_decreasing_array(sequence) or sequence in neg_sample:
                sequence = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            
        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> List[float]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.98, 0.02],
                [0.0, 1],
            ]
        )

        current_state = 0
        length = np.random.randint(1, max_length + 1)

        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state

        a = np.random.randint(0,10)
        ts = self.consecutive_pairs(states)
        for t in ts:
            if t == (0, 0):
                inc = np.random.randint(5, 20)
                a -=  inc
                sequence.append(a)
            else:
                inc = np.random.randint(10, 20)
                choice = np.random.choice([0, inc])
                b = a + choice

                sequence.append(b)

        return sequence
    
class S3(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 2
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def is_weakly_increasing_array(self, arr: List[float]) -> bool:
        if len(arr) == 1:
            return True
        elif len(arr) > 1 :
            # if arr[-1] > arr[-2]:
            for i in range(1, len(arr)): 
                if arr[i] < arr[i - 1]:
                    return False
            return True
            # else:
                # return False




    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for inx in range(num_pos):
            sequence = self.generate_sequence(max_length)
            # inc = np.random.randint(5, 20)
            # sequence.append(sequence[-1] + inc)
            while not self.is_weakly_increasing_array(sequence) or sequence in pos_sample:
                sequence = self.generate_sequence(max_length)
                # inc = np.random.randint(5, 20)
                # sequence.append(sequence[-1] + inc)
            pos_sample.append(sequence)
#             
        for i in range(num_neg):
            sequence = self.generate_sequence(max_length)
            while self.is_weakly_increasing_array(sequence) or sequence in neg_sample:
                sequence = self.generate_sequence(max_length)
            neg_sample.append(sequence)
#             
        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> List[float]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.98, 0.02],
                [0, 1]
                # [0.0, 1],
            ]
        )

        current_state = 0
        length = np.random.randint(1, max_length + 1)

        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state

        a = np.random.randint(0,10)
        ts = self.consecutive_pairs(states)
        for t in ts:
            if t == (0, 0):
                d = np.random.randint(4, 20)
                inc = np.random.choice([0, d])
                a +=  inc
                sequence.append(a)
            elif t == (0, 1):
                inc = np.random.randint(4, 20)
                a = a + inc
                sequence.append(a)
            elif t == (1, 1):
                inc = np.random.randint(4, 10)
                a = a - inc
                sequence.append(a)

            # else:
            #     # c = sequence[-1] if sequence else 0
            #     d = np.random.randint(5,10)
            #     e = a - d
            #     # e = random.choice([c - d])
            #     sequence.append(e)
        return sequence

class S4(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 2
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def is_weakly_increasing_array(self, arr: List[float]) -> bool:
        if len(arr) == 1:
            return True
        elif len(arr) > 1 :
            # if arr[-1] < arr[-2]:
            for i in range(1, len(arr)):
                if arr[i] > arr[i - 1]:
                    return False
            return True
            # else:
                # return False
    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for inx in range(num_pos):
            sequence = self.generate_sequence(max_length)
            # inc = np.random.randint(5, 20)
            # sequence.append(sequence[-1] - inc)
            while not self.is_weakly_increasing_array(sequence) or sequence in pos_sample:
                sequence = self.generate_sequence(max_length)
                # inc = np.random.randint(5, 20)
                # sequence.append(sequence[-1] - inc)
            pos_sample.append(sequence)
            # print(1, inx)
#             
        for i in range(num_neg):
            sequence = self.generate_sequence(max_length)
            while self.is_weakly_increasing_array(sequence) or sequence in neg_sample:
                sequence = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # print(2, i)
        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> List[float]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.8, 0.2],
                [0, 1]
                # [0.0, 1],
            ]
        )

        current_state = 0
        length = np.random.randint(1, max_length + 1)

        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state

        a = np.random.randint(0,10)
        ts = self.consecutive_pairs(states)
        for t in ts:
            if t == (0, 0):
                d = np.random.randint(4, 20)
                inc = np.random.choice([0, d])
                a -=  inc
                sequence.append(a)
            elif t == (0, 1):
                inc = np.random.randint(4, 20)
                a = a - inc
                sequence.append(a)
            elif t == (1, 1):
                inc = np.random.randint(1, 10)
                a = a + inc
                sequence.append(a)

            # else:
            #     # c = sequence[-1] if sequence else 0
            #     d = np.random.randint(5,10)
            #     e = a - d
            #     # e = random.choice([c - d])
            #     sequence.append(e)
        return sequence


class S5(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 4
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def upanddown(self, lst: List[float]) -> bool:
        if len(lst) < 3:
            return False

        # Find the minimum point (valley)
        min_idx = lst.index(max(lst))

        # Check if minimum is not at the edges
        if min_idx == 0 or min_idx == len(lst) - 1:
            return False

        # Check if sequence goes down before minimum and up after minimum
        for i in range(min_idx):
            if i > 0 and lst[i] <= lst[i - 1]:
                return False

        for i in range(min_idx + 1, len(lst)):
            if lst[i] >= lst[i - 1]:
                return False

        return True

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)

            while not self.upanddown(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
                # print(sequence)

            pos_sample.append(sequence)
            # print(1, i)


        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.upanddown(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            #
            # print(2, i)

        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 1, 0, 0],
                [0.0, 0.5, 0.5, 0],
                [0.0, 0, 0.95, 0.05],
                [0, 0, 0, 1],
            ]
        )

        current_state = 0
        length = np.random.randint(3, max_length)

        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state

        r0 = 0
        ts = self.consecutive_pairs(states)
        for t in ts:
            # print(t)
            if t == (0, 1):
                inc = np.random.randint(2, 4)
                r0 += inc
                sequence.append(r0)
            elif t == (1, 1):
                # p = np.random.uniform(0,1)
                inc = np.random.randint(2, 4)
                # if p < 0.5:
                #     r0 = r0 + 2
                # else:
                r0 = r0 + inc
                sequence.append(r0)
            elif t == (1, 2):
                inc = np.random.randint(2, 4)
                r0 -= inc
                sequence.append(r0)
            elif t == (2, 2):
                inc = np.random.randint(2, 4)
                r0 -= inc
                sequence.append(r0)
            elif t == (2, 3):
                r0 = r0 + np.random.randint(2, 4)
                sequence.append(r0)
            else:
                inc = np.random.randint(0, 5)
                p = np.random.uniform(0,1)
                if p < 0.2:
                    r0 = r0
                elif p < 0.6:
                    r0 = r0 + inc
                else:
                    r0 = r0 - inc
                sequence.append(r0)

        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0


class S6(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 4
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def downandup(self, lst: List[float]) -> bool:
        if len(lst) < 3:
            return False

        # Find the minimum point (valley)
        min_idx = lst.index(min(lst))

        # Check if minimum is not at the edges
        if min_idx == 0 or min_idx == len(lst) - 1:
            return False

        # Check if sequence goes down before minimum and up after minimum
        for i in range(min_idx):
            if i > 0 and lst[i] >= lst[i - 1]:
                return False

        for i in range(min_idx + 1, len(lst)):
            if lst[i] <= lst[i - 1]:
                return False

        return True

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)

            while not self.downandup(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
                # print(sequence)

            pos_sample.append(sequence)
            print(1, i)


        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.downandup(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            #
            print(2, i)

        return pos_sample, neg_sample

    
    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 1, 0, 0],
                [0.0, 0.5, 0.5, 0],
                [0.0, 0, 0.95, 0.05],
                [0, 0, 0, 1],
            ]
        )

        current_state = 0
        length = np.random.randint(3, max_length)

        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state

        r0 = 0
        ts = self.consecutive_pairs(states)
        for t in ts:
            # print(t)
            if t == (0, 1):
                inc = np.random.randint(2, 5)
                r0 -= inc
                sequence.append(r0)
            elif t == (1, 1):
                # p = np.random.uniform(0,1)
                inc = np.random.randint(2, 5)
                # if p < 0.5:
                #     r0 = r0 + 2
                # else:
                r0 = r0 - inc
                sequence.append(r0)
            elif t == (1, 2):
                inc = np.random.randint(2, 5)
                r0 += inc
                sequence.append(r0)
            elif t == (2, 2):
                inc = np.random.randint(2, 5)
                r0 += inc
                sequence.append(r0)
            elif t == (2, 3):
                r0 = r0 - np.random.randint(2, 5)
                sequence.append(r0)
            else:
                inc = np.random.randint(2, 5)
                p = np.random.uniform(0,1)
                if p < 0.2:
                    r0 = r0
                elif p < 0.6:
                    r0 = r0 + inc
                else:
                    r0 = r0 - inc
                sequence.append(r0)

        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0




class LStock(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0', 'r1']
        self.states = 5
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def is_higher_peaks_and_troughs(self, lst: List[float]) -> bool:
        if len(lst) == 0:
            return False
        if len(lst) == 1:
            return True
        if len(lst) == 2:
            return lst[0] < lst[1]

        peaks, troughs = [], []
        for i in range(1, len(lst) - 1):
            if lst[i] > lst[i - 1] and lst[i] > lst[i + 1]:
                peaks.append(lst[i])
            elif lst[i] < lst[i - 1] and lst[i] < lst[i + 1]:
                troughs.append(lst[i])

        if lst[-1] < lst[-2]:
            troughs.append(lst[-1])
        elif lst[-1] > lst[-2]:
            peaks.append(lst[-1])

        # peaks strictly decreasing
        for i in range(1, len(peaks)):
            if peaks[i] <= peaks[i - 1]:
                return False
        # troughs strictly decreasing
        for i in range(1, len(troughs)):
            if troughs[i] <= troughs[i - 1]:
                return False
        return True

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.is_higher_peaks_and_troughs(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            print(1, i)
            

        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.is_higher_peaks_and_troughs(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            print(2, i)
            
        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0],
                [0.0, 0.5, 0.5, 0.0, 0],
                [0.0, 0, 0.5, 0.5, 0], 
                [0.0, 0, 0.95, 0, 0.15],
                [0.0, 0, 0, 0, 1]
            ]
        )

        current_state = 0
        length = np.random.randint(3, max_length)

        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state
        r0 = 0
        r1 = 0 
        ts = self.consecutive_pairs(states)
        for t in ts:
            if t == (0, 1):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            elif t == (1, 1):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            elif t == (1, 2):
                inc = np.random.randint(5, 20)
                r1 -=  inc
                sequence.append(r1)
            elif t == (2, 2):
                inc = np.random.randint(5, 20)
                r1 -= inc
                sequence.append(r1)
            elif t == (2, 3):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            elif t == (3, 2):
                inc = np.random.randint(5, 20)
                r1 += inc
                sequence.append(r1)
            elif t == (3, 4):         
                c = r0 + np.random.randint(5, 20)
                sequence.append(c)
            else:
                p = np.random.uniform(0,1)
                if p < 0.5:
                    r0 = r0 - np.random.randint(5, 20)
                else:
                    r0 = r0 + np.random.randint(5, 20)
                # r0 = r0 + np.random.randint(5, 7)
                sequence.append(r0)

        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0
    
class LStock1(Language):
    """
    Language: higher peaks AND lower troughs.
    Peaks strictly increasing; troughs strictly decreasing.
    """
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0', 'r1']
        self.states = 5
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def is_higher_peaks_and_lower_troughs(self, lst: List[float]) -> bool:
        if len(lst) == 0:
            return False
        if len(lst) == 1:
            return lst[0] > 0
        if len(lst) == 2:
            return lst[0] < lst[1]

        peaks, troughs = [], []
        for i in range(1, len(lst) - 1):
            if lst[i] > lst[i - 1] and lst[i] > lst[i + 1]:
                peaks.append(lst[i])
            elif lst[i] < lst[i - 1] and lst[i] < lst[i + 1]:
                troughs.append(lst[i])

        # classify last endpoint
        if lst[-1] < lst[-2]:
            troughs.append(lst[-1])
        elif lst[-1] > lst[-2]:
            peaks.append(lst[-1])

        # peaks strictly increasing
        for i in range(1, len(peaks)):
            if peaks[i] <= peaks[i - 1]:
                return False
        # troughs strictly decreasing
        for i in range(1, len(troughs)):
            if troughs[i] >= troughs[i - 1]:
                return False
        return True

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.is_higher_peaks_and_lower_troughs(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            # print(1, i)
            

        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.is_higher_peaks_and_lower_troughs(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # 
            
        return pos_sample, neg_sample
    
    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0],
                [0.0, 0.5, 0.5, 0.0, 0],
                [0.0, 0, 0.5, 0.5, 0], 
                [0.0, 0, 0.9, 0, 0.1],
                [0.0, 0, 0, 0, 1]
            ]
        )

        current_state = 0
        length = np.random.randint(3, max_length)

        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state
        r0 = 0
        r1 = 0 
        ts = self.consecutive_pairs(states)
        for t in ts:
            if t == (0, 1):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            elif t == (1, 1):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            elif t == (1, 2):
                inc = np.random.randint(5, 20)
                r1 -=  inc
                sequence.append(r1)
            elif t == (2, 2):
                inc = np.random.randint(5, 20)
                r1 -= inc
                sequence.append(r1)
            elif t == (2, 3):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            elif t == (3, 2):
                inc = np.random.randint(5, 20)
                r1 -= inc
                sequence.append(r1)
            elif t == (3, 4):         
                c = r0 + np.random.randint(5, 20)
                sequence.append(c)
            else:
                tmp = np.random.randint(0,100)
                sequence.append(tmp)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0
        
class LStock2(Language):
    """
    Language: lower peaks AND lower troughs.
    Both peaks and troughs strictly decreasing.
    """
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0', 'r1']
        self.states = 5
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def is_lower_peaks_and_lower_troughs(self, lst: List[float]) -> bool:
        if len(lst) == 0:
            return False
        if len(lst) == 1:
            return True
        if len(lst) == 2:
            return lst[0] < lst[1]

        peaks, troughs = [], []
        for i in range(1, len(lst) - 1):
            if lst[i] > lst[i - 1] and lst[i] > lst[i + 1]:
                peaks.append(lst[i])
            elif lst[i] < lst[i - 1] and lst[i] < lst[i + 1]:
                troughs.append(lst[i])

        if lst[-1] < lst[-2]:
            troughs.append(lst[-1])
        elif lst[-1] > lst[-2]:
            peaks.append(lst[-1])

        # peaks strictly decreasing
        for i in range(1, len(peaks)):
            if peaks[i] >= peaks[i - 1]:
                return False
        # troughs strictly decreasing
        for i in range(1, len(troughs)):
            if troughs[i] >= troughs[i - 1]:
                return False
        return True

    
    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.is_lower_peaks_and_lower_troughs(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            print(1, i)
            

        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.is_lower_peaks_and_lower_troughs(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            
            
        return pos_sample, neg_sample
    
    
    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0],
                [0.0, 0.5, 0.5, 0.0, 0],
                [0.0, 0, 0.5, 0.5, 0], 
                [0.0, 0, 0.9, 0, 0.1],
                [0.0, 0, 0, 0, 1]
            ]
        )

        current_state = 0
        length = np.random.randint(3, max_length)

        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state
        r0 = 0
        r1 = 0 
        ts = self.consecutive_pairs(states)
        for t in ts:
            if t == (0, 1):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            elif t == (1, 1):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            elif t == (1, 2):
                inc = np.random.randint(5, 20)
                r1 -=  inc
                sequence.append(r1)
            elif t == (2, 2):
                inc = np.random.randint(5, 20)
                r1 -= inc
                sequence.append(r1)
            elif t == (2, 3):
                inc = np.random.randint(5, 20)
                r0 -= inc
                sequence.append(r0)
            elif t == (3, 2):
                inc = np.random.randint(5, 20)
                r1 -= inc
                sequence.append(r1)
            elif t == (3, 4):         
                c = r0 + np.random.randint(5, 20)
                sequence.append(c)
            else:
                tmp = np.random.randint(0,100)
                sequence.append(tmp)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0
    
class L2A(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 3
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def two_a(self, lst: List[float]) -> bool:
        if len(set(lst)) != 1:
            return False
        else:
            return len(lst) % 2 == 0
    
    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.two_a(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            # print(1, i)
            

        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.two_a(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # print(2, i)
            
        return pos_sample, neg_sample
    
    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 1, 0],
                [0.95, 0, 0.05],
                [0.0, 0.0, 1],
            ]
        )

        current_state = 0
        length = np.random.randint(1, max_length + 1)

        # Generate state transitions
        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = (
                    valid_probs[valid_states] / valid_probs[valid_states].sum()
            )
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state

        # Generate sequence based on pairs of states
        ts = self.consecutive_pairs(states)
        a = np.random.randint(-200, 200)
        for t in ts:
            if t == (0, 1):
                sequence.append(a)
            elif t == (1, 2):
                b = np.random.randint(-200, 200)
                while a == b:
                    b = np.random.randint(-200, 200)
                sequence.append(b)
            elif t == (1, 0):
                sequence.append(a)
            else:
                b = np.random.randint(-200, 200)
                sequence.append(b)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0
        
class S7(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0', 'r1']
        self.states = 4
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def up_down_up(self, lst: List[float]) -> bool:
        # Need at least 4 points to realize: up, down, up (3 transitions)
        if len(lst) < 4:
            return False

        diffs = [lst[i] - lst[i - 1] for i in range(1, len(lst))]

        # No flat differences
        if any(d == 0 for d in diffs):
            return False

        # The pattern must start up, then down, then up
        pattern = [True, False, True]  # True = up, False = down

        # We only care about the first 3 transitions
        for i in range(3):
            d = diffs[i]
            expected_up = pattern[i]
            if expected_up and d <= 0:
                return False
            if not expected_up and d >= 0:
                return False

        # Remaining transitions (if any) must continue the "up" trend
        return all(d > 0 for d in diffs[3:])

    
    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.up_down_up(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            # print(1, i)
            

        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.up_down_up(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # print(2, i)
            
        return pos_sample, neg_sample

    
    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 0.9, 0, 0, 0.1],
                [0.0, 0.1, 0.9, 0, 0],
                [0.0, 0, 0.1, 0.9, 0],
                [0.0, 0, 0, 0.95, 0.05],
                [0, 0, 0, 0, 1],
            ]
        )

        current_state = 0
        length = np.random.randint(3, max_length)

        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state

        r0 = 0
        ts = self.consecutive_pairs(states)
        for t in ts:
            # print(t)
            
            if t == (0, 1):
                inc = np.random.randint(1, 10)
                r0 += inc
                sequence.append(r0)
            elif t == (0, 4):
                inc = np.random.randint(1, 10)
                r0 -= inc
                sequence.append(r0)
            elif t == (1, 1):
                # p = np.random.uniform(0,1)
                inc = np.random.randint(1, 10)
                # if p < 0.5:
                #     r0 = r0 + 2
                # else:
                r0 = r0 + inc
                sequence.append(r0)
            elif t == (1, 2):
                inc = np.random.randint(1, 10)
                r0 -= inc
                sequence.append(r0)
            elif t == (2, 2):
                inc = np.random.randint(1, 10)
                r0 -= inc
                sequence.append(r0)
            elif t == (2, 3):
                r0 = r0 + np.random.randint(1, 10)
                sequence.append(r0)
            elif t == (3,3):
                inc = np.random.randint(1, 10)
                r0 = r0 + inc
                sequence.append(r0)
            elif t == (3,4):
                r0 = r0 - np.random.randint(1, 10)
                sequence.append(r0)
            else:
                inc = np.random.randint(1, 10)
                p = np.random.uniform(0,1)
                if p < 0.2:
                    r0 = r0
                elif p < 0.6:
                    r0 = r0 + inc
                else:
                    r0 = r0 - inc
                sequence.append(r0)

        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0


class S8(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 6
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def up_down_up_down_prefix(self, lst: List[float]) -> bool:
        """
        Begins with: up, down, up, down (first 4 transitions).
        Allows extra points afterward (no restriction on the tail).
        """
        if len(lst) < 5:
            return False
        diffs = [lst[i] - lst[i-1] for i in range(1, len(lst))]
        if any(d == 0 for d in diffs[:4]):
            return False
        pattern = [True, False, True, False]
        for d, expect_up in zip(diffs[:4], pattern):
            if expect_up and d <= 0:
                return False
            if not expect_up and d >= 0:
                return False
        return True
    
    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.up_down_up_down_prefix(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            print(1, i)
            

        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.up_down_up_down_prefix(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            print(2, i)
            
        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 0.9, 0, 0, 0, 0.1],
                [0.0, 0.1, 0.9, 0, 0, 0],
                [0.0, 0, 0.1, 0.9, 0, 0],
                [0.0, 0, 0, 0.1, 0.9, 0],
                [0.0, 0, 0, 0, 0.95, 0.05],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        current_state = 0
        length = np.random.randint(3, max_length)

        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state

        r0 = 0
        ts = self.consecutive_pairs(states)
        for t in ts:
            # print(t)
            
            if t == (0, 1):
                inc = np.random.randint(1, 10)
                r0 += inc
                sequence.append(r0)
            elif t == (0, 4):
                inc = np.random.randint(1, 10)
                r0 -= inc
                sequence.append(r0)
            elif t == (1, 1):
                # p = np.random.uniform(0,1)
                inc = np.random.randint(1, 10)
                # if p < 0.5:
                #     r0 = r0 + 2
                # else:
                r0 = r0 + inc
                sequence.append(r0)
            elif t == (1, 2):
                inc = np.random.randint(1, 10)
                r0 -= inc
                sequence.append(r0)
            elif t == (2, 2):
                inc = np.random.randint(1, 10)
                r0 -= inc
                sequence.append(r0)
            elif t == (2, 3):
                r0 = r0 + np.random.randint(1, 10)
                sequence.append(r0)
            elif t == (3,3):
                inc = np.random.randint(1, 10)
                r0 = r0 + inc
                sequence.append(r0)
            elif t == (3,4):
                r0 = r0 - np.random.randint(1, 10)
                sequence.append(r0)
            elif t == (4,4):
                inc = np.random.randint(1, 10)
                r0 = r0 - inc
                sequence.append(r0)
            elif t == (4, 5):
                r0 = r0 + np.random.randint(1, 10)
                sequence.append(r0)
            else:
                inc = np.random.randint(1, 10)
                p = np.random.uniform(0,1)
                if p < 0.2:
                    r0 = r0
                elif p < 0.6:
                    r0 = r0 + inc
                else:
                    r0 = r0 - inc
                sequence.append(r0)

        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0

        
class S53(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0', 'r1']
        self.states = 6
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def up_down_up_down_up_prefix(self, lst: List[float]) -> bool:
        """
        Starts with: up, down, up, down, up, down (first 6 transitions).
        Allows extra points afterward (no restriction on the tail).
        """
        if len(lst) < 6:
            return False
        diffs = [lst[i] - lst[i - 1] for i in range(1, len(lst))]
        if any(d == 0 for d in diffs[:5]):
            return False
        pattern = [True, False, True, False, True]
        for d, expect_up in zip(diffs[:5], pattern):
            if expect_up and d <= 0:
                return False
            if not expect_up and d >= 0:
                return False
        return True
        
    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.up_down_up_down_up_prefix(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            print(1, i)
            

        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.up_down_up_down_up_prefix(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            print(2, i)
            
        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.5, 0.5, 0.0, 0, 0, 0 ],
                [0.0, 0.5, 0.5, 0, 0, 0],
                [0.0, 0.0, 0.5, 0.5, 0, 0],
                [0.0, 0.0, 0.0, 0.5, 0.5, 0],
                [0.0, 0.0, 0.0, 0.0, 0.95, 0.05],
                [0.0, 0.0, 0.0, 0.0, 0, 1]
                # [0 , 0, 0, 1]
              ]
        )

        current_state = 0
        length = np.random.randint(3, max_length)

        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state
        r0 = 0
        r1 = 0 
        ts = self.consecutive_pairs(states)
        for t in ts:
            if t == (0, 0):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            elif t == (0, 1):
                inc = np.random.randint(5, 20)
                r1 = r0 - inc
                sequence.append(r1)
            elif t == (1, 1):
                inc = np.random.randint(5, 20)
                r1 -= inc
                sequence.append(r1)
            elif t == (1, 2):
                r0 = r1 + np.random.randint(5, 20)
                sequence.append(r0)
            elif t == (2, 2):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            elif t == (2, 3):
                inc = np.random.randint(5, 20)
                r1 = r0 - inc
                sequence.append(r1)
            elif t == (3, 3):
                inc = np.random.randint(5, 20)
                r1 -= inc
                sequence.append(r1)
            elif t == (3, 4):
                r0 = r1 + np.random.randint(5, 20)
                sequence.append(r0)
            elif t == (4, 4):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            else:
                r0 = r0 - np.random.randint(5, 20)
                sequence.append(r0)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0
        
class S54(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0', 'r1']
        self.states = 7
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def up_down_up_down_up_down_prefix(self, lst: List[float]) -> bool:
        """
        Starts with: up, down, up, down, up, down (first 6 transitions).
        Allows extra points afterward (no restriction on the tail).
        """
        if len(lst) < 7:
            return False
        diffs = [lst[i] - lst[i - 1] for i in range(1, len(lst))]
        if any(d == 0 for d in diffs[:6]):
            return False
        pattern = [True, False, True, False, True, False]
        for d, expect_up in zip(diffs[:6], pattern):
            if expect_up and d <= 0:
                return False
            if not expect_up and d >= 0:
                return False
        return True
        
    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.up_down_up_down_up_down_prefix(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            print(1, i)
            

        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.up_down_up_down_up_down_prefix(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            print(2, i)
            
        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.5,0.5, 0.0, 0, 0, 0, 0],
                [0.0, 0.5,0.5, 0, 0, 0, 0],
                [0.0, 0.0, 0.5,0.5, 0, 0, 0],
                [0.0, 0.0, 0.0, 0.5,0.5, 0, 0],
                [0.0, 0.0, 0.0, 0.0, 0.5,0.5, 0],
                [0.0, 0.0, 0.0, 0.0, 0, 0.95,0.05],
                [0.0, 0.0, 0.0, 0.0, 0, 0, 1]
              ]
        )

        current_state = 0
        length = np.random.randint(3, max_length)

        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state
        r0 = 0
        r1 = 0 
        ts = self.consecutive_pairs(states)
        for t in ts:
            if t == (0, 0):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            elif t == (0, 1):
                inc = np.random.randint(5, 20)
                r1 = r0 - inc
                sequence.append(r1)
            elif t == (1, 1):
                inc = np.random.randint(5, 20)
                r1 -= inc
                sequence.append(r1)
            elif t == (1, 2):
                r0 = r1 + np.random.randint(5, 20)
                sequence.append(r0)
            elif t == (2, 2):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            elif t == (2, 3):
                inc = np.random.randint(5, 20)
                r1 = r0 - inc
                sequence.append(r1)
            elif t == (3, 3):
                inc = np.random.randint(5, 20)
                r1 -= inc
                sequence.append(r1)
            elif t == (3, 4):
                r0 = r1 + np.random.randint(5, 20)
                sequence.append(r0)
            elif t == (4, 4):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            elif t == (4, 5):
                inc = np.random.randint(5, 20)
                r1 = r0 - inc
                sequence.append(r1)
            elif t == (5, 5):
                inc = np.random.randint(5, 20)
                r1 -= inc
                sequence.append(r1)
            # elif t == (5, 6):
            #     r0 = r1 + np.random.randint(5, 20)
            #     sequence.append(r0)
            # elif t == (6, 6):
            #     inc = np.random.randint(5, 20)
            #     r0 += inc
            #     sequence.append(r0)
            else:
                r1 = r1 + np.random.randint(5, 20)
                sequence.append(r1)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0

class S55(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0', 'r1']
        self.states = 8
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def up_down_up_down_up_down_up_prefix(self, lst: List[float]) -> bool:
        """
        Starts with: up, down, up, down, up, down (first 6 transitions).
        Allows extra points afterward (no restriction on the tail).
        """
        if len(lst) < 8:
            return False
        diffs = [lst[i] - lst[i - 1] for i in range(1, len(lst))]
        if any(d == 0 for d in diffs[:7]):
            return False
        pattern = [True, False, True, False, True, False, True]
        for d, expect_up in zip(diffs[:7], pattern):
            if expect_up and d <= 0:
                return False
            if not expect_up and d >= 0:
                return False
        return True
        
    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.up_down_up_down_up_down_up_prefix(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            print(1, i)
            

        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.up_down_up_down_up_down_up_prefix(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            print(2, i)
            
        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.5,0.5, 0.0, 0, 0, 0, 0, 0],
                [0.0, 0.5,0.5, 0, 0, 0, 0, 0],
                [0.0, 0.0, 0.5,0.5, 0, 0, 0, 0],
                [0.0, 0.0, 0.0, 0.5,0.5, 0, 0, 0],
                [0.0, 0.0, 0.0, 0.0, 0.5,0.5, 0, 0],
                [0.0, 0.0, 0.0, 0.0, 0, 0.5,0.5, 0],
                [0.0, 0.0, 0.0, 0.0, 0, 0, 0.95,0.05],
                [0.0, 0.0, 0.0, 0.0, 0, 0, 0, 1]
              ]
        )

        current_state = 0
        length = np.random.randint(3, max_length)

        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states]
            normalized_probs /= normalized_probs.sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state
        r0 = 0
        r1 = 0 
        ts = self.consecutive_pairs(states)
        for t in ts:
            if t == (0, 0):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            elif t == (0, 1):
                inc = np.random.randint(5, 20)
                r1 = r0 - inc
                sequence.append(r1)
            elif t == (1, 1):
                inc = np.random.randint(5, 20)
                r1 -= inc
                sequence.append(r1)
            elif t == (1, 2):
                r0 = r1 + np.random.randint(5, 20)
                sequence.append(r0)
            elif t == (2, 2):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            elif t == (2, 3):
                inc = np.random.randint(5, 20)
                r1 = r0 - inc
                sequence.append(r1)
            elif t == (3, 3):
                inc = np.random.randint(5, 20)
                r1 -= inc
                sequence.append(r1)
            elif t == (3, 4):
                r0 = r1 + np.random.randint(5, 20)
                sequence.append(r0)
            elif t == (4, 4):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            elif t == (4, 5):
                inc = np.random.randint(5, 20)
                r1 = r0 - inc
                sequence.append(r1)
            elif t == (5, 5):
                inc = np.random.randint(5, 20)
                r1 -= inc
                sequence.append(r1)
            elif t == (5, 6):
                r0 = r1 + np.random.randint(5, 20)
                sequence.append(r0)
            elif t == (6, 6):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            # elif t == (5, 6):
            #     r0 = r1 + np.random.randint(5, 20)
            #     sequence.append(r0)
            # elif t == (6, 6):
            #     inc = np.random.randint(5, 20)
            #     r0 += inc
            #     sequence.append(r0)
            else:
                r0 = r0 - np.random.randint(5, 20)
                sequence.append(r0)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0
  
# class LStock3(Language):
#     """
#     Language: lower peaks AND lower troughs.
#     Both peaks and troughs strictly decreasing.
#     """
#     def __init__(self, num_pos: int, num_neg: int, max_len: int):
#         super().__init__(num_pos, num_neg, max_len)
#         self.regs = ['r0', 'r1']
#         self.states = 5
#         self.edge_limit_num = 2
#         self.edge_limit_type = "at_most"

#     def is_lower_peaks_and_higher_troughs(self, lst: List[float]) -> bool:
#         if len(lst) == 0:
#             return False
#         if len(lst) == 1:
#             return True
#         if len(lst) == 2:
#             return lst[0] < lst[1]

#         peaks, troughs = [], []
#         for i in range(1, len(lst) - 1):
#             if lst[i] > lst[i - 1] and lst[i] > lst[i + 1]:
#                 peaks.append(lst[i])
#             elif lst[i] < lst[i - 1] and lst[i] < lst[i + 1]:
#                 troughs.append(lst[i])

#         if lst[-1] < lst[-2]:
#             troughs.append(lst[-1])
#         elif lst[-1] > lst[-2]:
#             peaks.append(lst[-1])

#         # peaks strictly decreasing
#         for i in range(1, len(peaks)):
#             if peaks[i] >= peaks[i - 1]:
#                 return False
#         # troughs strictly decreasing
#         for i in range(1, len(troughs)):
#             if troughs[i] <= troughs[i - 1]:
#                 return False
#         return True

    
#     def generate_samples(
#             self, num_pos: int, num_neg: int, max_length: int
#     ) -> Tuple[List[List[float]], List[List[float]]]:
#         pos_sample = []
#         neg_sample = []
#         for i in range(num_pos):
#             sequence, last_guy = self.generate_sequence(max_length)
#             while not self.is_lower_peaks_and_higher_troughs(sequence) or sequence in pos_sample:
#                 sequence, last_guy = self.generate_sequence(max_length)
#             pos_sample.append(sequence)
#             print(1, i)
            

#         for i in range(num_neg):
#             sequence, last_guy = self.generate_sequence(max_length)
#             while self.is_lower_peaks_and_higher_troughs(sequence) or sequence in neg_sample:
#                 sequence, last_guy = self.generate_sequence(max_length)
#             neg_sample.append(sequence)
#             
            
#         return pos_sample, neg_sample
    
    
#     def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
#         """
#         Encourage successive swings that gradually diminish in amplitude.
#         """
#         transition_matrix = np.array([
#             [0.0, 1.0, 0.0, 0.0, 0],  # 0 -> 1
#             [0.0, 0.5, 0.5, 0.0, 0],  # 1 -> 1,2
#             [0.0, 0, 0, 1, 0],  # 2 -> 2,3
#             [0.0, 0.0, 0.9, 0, 0.1],  # 3 -> 2,3 (damped oscillation)
#             [0.0, 0.0, 0, 0, 1]  # 4 -> 3,4 (damped oscillation)
            
 
            
#         ])

#         current_state = 0
#         length = int(np.random.randint(3, max_length))
#         states = [current_state]
#         for _ in range(length):
#             probs = transition_matrix[current_state]
#             valid = np.where(probs > 0)[0]
#             p = probs[valid] / probs[valid].sum()
#             nxt = int(np.random.choice(valid, p=p))
#             states.append(nxt)
#             current_state = nxt

#         r0 = 0
#         r1 = 0
#         sequence: List[float] = []
#         # Start with a moderate amplitude and shrink it to bias toward lower peaks/troughs
#         # amp = int(np.random.randint(10, 30))
#         for (a, b) in self.consecutive_pairs(states):
#             inc = max(1, int(np.random.randint(5, 20)))
#             if (a, b) in {(0, 1), (1, 1)}:
#                 r0 += inc; sequence.append((r0))
#             elif (a, b) in {(1, 2)}:
#                 r1 -= inc; sequence.append((r1))
#             elif (a, b) in {(2, 3)}:
#                 c = np.random.uniform(r1, r0)
#                 r0 = c; sequence.append((r0))
#             elif (a, b) in {(3, 2)}:
#                 # print(r1, r0)
#                 c = np.random.uniform(r1, r0)
#                 r1 = c; sequence.append((r1))
#             elif (a, b) == (3, 4):
#                 c = r0 + inc
#                 sequence.append((c))
#             else:
#                 sequence.append((np.random.randint(-50, 50)))
#             # gradually reduce amplitude
#             # amp = max(5, amp - 1)
#         return sequence, states[-1]



def new_generate_language(language_name: str, num_pos: int = 50000, num_neg: int = 50000, max_len: int = 50, save: bool = False, printout: bool = False) -> List[str]:
    # Discover all subclasses of the Language class
    language_classes = {cls.__name__: cls for cls in Language.__subclasses__()}

    if language_name not in language_classes:
        raise ValueError(f"Unsupported language: {language_name}. Available languages: {list(language_classes.keys())}")

    # Initialize the selected language
    language_class = language_classes[language_name]
    lang = language_class(num_pos, num_neg, max_len)

    if save:
        dataset_dir = "dataset/"
        os.makedirs(dataset_dir, exist_ok=True)
        filename = os.path.join(dataset_dir, f"{language_name}.csv")
        test_filename = os.path.join(dataset_dir, f"test_{language_name}.csv")


        # Generate new samples
        lang.pos, lang.neg = lang.generate_samples(num_pos, num_neg, max_len)
        res = []
        for i in lang.pos:
            if printout:
                print(','.join(str(x) for x in i))
            tmp = i + [1]
            res.append(tmp)

        for i in lang.neg:
            if printout:
                print(','.join(str(x) for x in i))
            tmp = i + [0]
            res.append(tmp)

        # random.shuffle(res)
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(res)
        # Generate new samples
        pos, neg = lang.generate_samples(1000, 1000, 100)
        res1 = []
        for i in pos:
            tmp = i + [1]
            res1.append(tmp)

        for i in neg:
            tmp = i + [0]
            res1.append(tmp)

        with open(test_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(res1)
    else:
        lang.pos, lang.neg = lang.generate_samples(num_pos, num_neg, max_len)
#         lang.pos.sort(key=lambda x: len(x))
#         lang.neg.sort(key=lambda x: len(x))

        res = lang.pos + lang.neg

        # sort pos and neg with len
        for i in lang.pos:
            tmp = i + [1]
            res.append(tmp)

        for i in lang.neg:
            tmp = i + [0]
            res.append(tmp)
        res.sort(key=lambda x: len(x) if isinstance(x, list) else 0)
        for i in res:
            if printout:
                print(','.join(str(x) for x in i))
    return res

def new_generate_language_l(language_name: str, num_pos: int = 500, num_neg: int = 0, max_len: int = 20) -> List[str]:
    # Discover all subclasses of the Language class
    language_classes = {cls.__name__: cls for cls in Language.__subclasses__()}

    if language_name not in language_classes:
        raise ValueError(f"Unsupported language: {language_name}. Available languages: {list(language_classes.keys())}")

    # Initialize the selected language
    language_class = language_classes[language_name]
    lang = language_class(num_pos, num_neg, max_len)
    return lang

def random_sample_generation(language_name: str, num_pos: int = 500, num_neg: int = 500, max_len: int = 10) -> List[str]:
    # Discover all subclasses of the Language class
    language_classes = {cls.__name__: cls for cls in Language.__subclasses__()}

    if language_name not in language_classes:
        raise ValueError(f"Unsupported language: {language_name}. Available languages: {list(language_classes.keys())}")

    # Initialize the selected language
    language_class = language_classes[language_name]
    lang = language_class(num_pos, num_neg, max_len)

    # Generate new samples
    lang.pos, lang.neg = lang.generate_samples(num_pos, num_neg, max_len)

    return lang.pos + lang.neg


def get_available_languages():
    """Get a list of all available language classes."""
    language_classes = {cls.__name__: cls for cls in Language.__subclasses__()}
    return sorted(language_classes.keys())



def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Generate samples for different formal languages')
    parser.add_argument('language', nargs='?', default='L2',
                       help='Language to generate (default: L2). Use --list to see available languages.')
    parser.add_argument('--num-pos', type=int, default=500,
                       help='Number of positive samples to generate (default: 50)')
    parser.add_argument('--num-neg', type=int, default=0,
                       help='Number of negative samples to generate (default: 0)')
    parser.add_argument('--max-len', type=int, default=50,
                       help='Maximum sequence length (default: 50)')
    parser.add_argument('--list', action='store_true',
                       help='List all available languages and exit')
    parser.add_argument('--save', action='store_true',
                       help='Save the generated samples to a file')
    parser.add_argument('--printout', action='store_true',
                          help='Print the generated samples to the console')

    args = parser.parse_args()

    # List available languages if requested
    if args.list:
        available_languages = get_available_languages()
        print("Available languages:")
        for lang in available_languages:
            print(f"  {lang}")
        sys.exit(0)

    # Validate language name
    available_languages = get_available_languages()
    if args.language not in available_languages:
        print(f"Error: '{args.language}' is not a valid language.")
        print(f"Available languages: {', '.join(available_languages)}")
        print("Use --list to see all available languages.")
        sys.exit(1)

    try:
        lang = new_generate_language(args.language, num_pos=args.num_pos, num_neg=args.num_neg, max_len=args.max_len, save=args.save, printout=args.printout)

    except Exception as e:
        print(f"Error generating language samples: {e}")
        sys.exit(1)


def generate_only_positive_samples(language_name: str, num_pos: int = 5000, max_len: int = 50) -> List[str]:
    # Discover all subclasses of the Language class
    language_classes = {cls.__name__: cls for cls in Language.__subclasses__()}

    if language_name not in language_classes:
        raise ValueError(f"Unsupported language: {language_name}. Available languages: {list(language_classes.keys())}")

    # Initialize the selected language
    language_class = language_classes[language_name]
    lang = language_class(num_pos, 0, max_len)

    # Generate new samples
    lang.pos, lang.neg = lang.generate_samples(num_pos, 0, max_len)

    return lang.pos



def random_sample_generation(language_name: str, num_pos: int = 500, num_neg: int = 500, max_len: int = 10) -> List[str]:
    # Discover all subclasses of the Language class
    language_classes = {cls.__name__: cls for cls in Language.__subclasses__()}

    if language_name not in language_classes:
        raise ValueError(f"Unsupported language: {language_name}. Available languages: {list(language_classes.keys())}")

    language_class = language_classes[language_name]
    lang = language_class(num_pos, num_neg, max_len)

    # Generate new samples
    lang.pos, lang.neg = lang.generate_samples(num_pos, num_neg, max_len)

    return lang.pos + lang.neg


if __name__ == "__main__":
    main()
    