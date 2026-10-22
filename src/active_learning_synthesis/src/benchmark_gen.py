import numpy as np
from typing import List, Tuple
import argparse
import sys
from typing import List, Optional



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


class DistinctK(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.k = 8
        self.regs = [f"r{i}" for i in range(self.k)]
        self.states = self.k + 2
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def is_valid(self, lst: List[float]) -> bool:
        if len(lst) != self.k:
            return False
        return len(set(lst)) == self.k

    def generate_samples(
        self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:

        pos_sample = []
        neg_sample = []

        for _ in range(num_pos):
            sequence, _ = self.generate_sequence(max_length)
            while not self.is_valid(sequence) or sequence in pos_sample:
                sequence, _ = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            # print(1, len(pos_sample))
        for _ in range(num_neg):
            sequence, _ = self.generate_sequence(max_length)
            while self.is_valid(sequence) or sequence in neg_sample:
                sequence, _ = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # print(2, len(neg_sample))
        return pos_sample, neg_sample


    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:

        sequence = []

        # Build transition matrix automatically
        n = self.k + 2
        transition_matrix = np.zeros((n, n))

        transition_matrix[0][1] = 1

        for i in range(1, self.k):
            transition_matrix[i][i+1] = 0.5
            transition_matrix[i][n-1] = 0.5

        transition_matrix[self.k][n-1] = 1
        transition_matrix[n-1][n-1] = 1

        current_state = 0
        length = np.random.randint(1, max_length + 1)

        states = [current_state]

        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states] / valid_probs[valid_states].sum()

            next_state = np.random.choice(valid_states, p=normalized_probs)

            states.append(next_state)
            current_state = next_state

        ts = self.consecutive_pairs(states)

        # generate k distinct symbols
        # Generate a strictly increasing list of k symbols
        start = np.random.randint(-200, 200 - self.k)
        step = np.random.randint(1, 10, size=self.k)
        symbols = [start]
        for s in step:
            symbols.append(symbols[-1] + s)
        symbols = symbols[:self.k]

        for t in ts:
            i, j = t

            if i == 0 and j == 1:
                sequence.append(symbols[0])

            elif 1 <= i < self.k and j == i + 1:
                sequence.append(symbols[i])

            elif 1 <= i <= self.k and j == self.k + 1:
                sequence.append(symbols[i-1])

            else:
                sequence.append(np.random.randint(-200, 200))

        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0
        
        

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


class L20A(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 21                 # 20-cycle + 1 sink
        self.edge_limit_num = 1
        self.edge_limit_type = "exactly"

    def twenty_a(self, lst: List[float]) -> bool:
        if len(set(lst)) != 1:
            return False
        else:
            return len(lst) % 20 == 0

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []

        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.twenty_a(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)

        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.twenty_a(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)

        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []

        # States: 0..19 are the cycle; 20 is the sink
        transition_matrix = np.zeros((21, 21), dtype=float)

        # Build the 20-cycle transitions with 0.9, plus 0.1 to sink
        for s in range(20):
            nxt = (s + 1) % 20
            transition_matrix[s, nxt] = 0.99
            transition_matrix[s, 20] = 0.01

        # Sink self-loop
        transition_matrix[20, 20] = 1.0

        current_state = 0
        length = np.random.randint(1, max_length + 1)

        # Generate state transitions
        states = [current_state]
        for _ in range(length):
            valid_probs = transition_matrix[current_state]
            valid_states = np.where(valid_probs > 0)[0]
            normalized_probs = valid_probs[valid_states] / valid_probs[valid_states].sum()
            next_state = np.random.choice(valid_states, p=normalized_probs)
            states.append(next_state)
            current_state = next_state

        # Emit values based on consecutive state pairs
        ts = self.consecutive_pairs(states)
        a = np.random.randint(-400, 400)

        for (u, v) in ts:
            # The 20 cycle edges are exactly u in [0..19] and v == (u+1)%20
            if (u < 20) and (v == (u + 1) % 20):
                sequence.append(a)
            else:
                b = np.random.randint(-400, 400)
                while a == b:
                    b = np.random.randint(-400, 400)
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
        self.regs = ['r0', 'r1']
        self.states = 5
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def is_l3_language(self, lst: List[float]) -> bool:
        if len(lst) == 0:
            return False
        unique = list(set(lst))
        if len(unique) != 2:
            return len(lst) % 2 == 1
        if len(unique) > 2:
            return False
        a = lst[0]
        b = [i for i in unique if i not in [a]][0]
        count_a = lst.count(a)
        count_b = lst.count(b)
        # Odd number of a's, even number of b's
        return (count_a % 2 == 1) and (count_b % 2 == 0)

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
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.3, 0.7, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.9, 0.1],
                [0.0, 0.9, 0.0, 0.0, 0.0, 0.1],
                [0.0, 0.0, 0.9, 0.0, 0.0, 0.1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
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
                sequence.append(a)
            elif t == (1, 2):
                sequence.append(b)
            elif t == (1, 3):
                sequence.append(a)
            elif t == (2, 4):
                sequence.append(b)
            elif t == (2, 5):
                c = np.random.randint(-200, 200)
                while c == b:
                    c = np.random.randint(-200, 200)
                sequence.append(c)
            elif t == (3, 1):
                sequence.append(a)
            elif t == (3, 5):
                c = np.random.randint(-200, 200)
                while c == a:
                    c = np.random.randint(-200, 200)
                sequence.append(c)
            elif t == (4, 2):
                sequence.append(b)
            elif t == (4, 5):
                c = np.random.randint(-200, 200)
                while c == b:
                    c = np.random.randint(-200, 200)
                sequence.append(c)
            elif t == (5, 5):
                c = np.random.randint(-200, 200)
                sequence.append(c)
            else:
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
            # print(1, inx)
        for inx in range(num_neg):
            sequence, nl = self.generate_sequence(max_length)
            while self.is_l4_language(sequence) or sequence in neg_sample:
                sequence, nl = self.generate_sequence(max_length)

            neg_sample.append(sequence)
            # print(2, inx)
        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.5, 0, 0.5, 0.0],
                [0.5, 0, 0, 0.5],
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

        a = np.random.randint(-200, 200)
        b = np.random.randint(-200, 200)
        while a ==b :
            b = np.random.randint(-200, 200)

        ts = self.consecutive_pairs(states)
        for t in ts:
            if t == (0, 0):
                sequence.append(b)
            elif t == (0, 1):
                sequence.append(a)
                # a = b
            elif t == (1, 0):
                sequence.append(b)
            elif t == (1, 2):
                # b = np.random.randint(-200, 200)
                # while a == b:
                #     b = np.random.randint(-200, 200)
                sequence.append(a)
                # a = b
            elif t == (2, 3):
                sequence.append(a)
            elif t == (2, 0):
                sequence.append(b)
            elif t == (3, 3):
                c = np.random.choice([a,b])
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

            pos_sample.append(sequence)
        for i in range(num_neg):
            sequence, nl = self.generate_sequence(max_length)
            while self.is_l5_language(sequence) or sequence in neg_sample:
                sequence, nl = self.generate_sequence(max_length)

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
        self.regs = ['r0', 'r1']
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


        for i in range(num_neg):
            sequence, ln = self.generate_sequence(max_length)
            while (self.matches_pattern(sequence)) or sequence in neg_sample:
                sequence, ln = self.generate_sequence(max_length)
            neg_sample.append(sequence)

        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.5, 0.5, 0, 0, 0],
                [0, 0.5, 0.5, 0, 0],
                [0, 0, 0.5, 0.5, 0],
                [0, 0, 0, 0.5, 0.5],
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
            elif t == (3, 3):
                sequence.append(b)
            elif t == (3, 4):
                sequence.append(a)
            elif t == (4, 4):
                c = np.random.randint(0, 200)
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
            inc = np.random.randint(5, 20)
            sequence.append(sequence[-1] + inc)
            while not self.is_weakly_increasing_array(sequence) or sequence in pos_sample:
                sequence = self.generate_sequence(max_length)
                inc = np.random.randint(5, 20)
                sequence.append(sequence[-1] + inc)
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
                inc = np.random.randint(1, 10)
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
            inc = np.random.randint(5, 20)
            sequence.append(sequence[-1] - inc)
            while not self.is_weakly_increasing_array(sequence) or sequence in pos_sample:
                sequence = self.generate_sequence(max_length)
                inc = np.random.randint(5, 20)
                sequence.append(sequence[-1] - inc)
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
        sequence = [1]
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

        r0 = 1
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
            # print(1, i)


        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.downandup(sequence) or sequence in neg_sample:
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



class S9(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0', 'r1']
        self.states = 5
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def is_higher_peaks_and_troughs(self, lst: List[float]) -> bool:
        if len(lst) == 0:
            return False
        elif len(lst) == 1:
            return lst[0] > 0
        elif len(lst) == 2:
            return lst[0] < lst[1]
        peaks = []
        troughs = []

        for i in range(1, len(lst) - 1):
            if lst[i] > lst[i - 1] and lst[i] > lst[i + 1]:
                peaks.append(lst[i])
            elif lst[i] < lst[i - 1] and lst[i] < lst[i + 1]:
                troughs.append(lst[i])
        if lst[-1] < lst[-2]:
            if lst[-1] not in troughs:
                troughs.append(lst[-1])
        elif lst[-1] > lst[-2]:
            if lst[-1] not in peaks:
                peaks.append(lst[-1])
        # Check if peaks are in increasing order
        for i in range(1, len(peaks)):
            if peaks[i] <= peaks[i - 1]:
                return False

        # Check if troughs are in increasing order
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
            # print(1, i)


        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.is_higher_peaks_and_troughs(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            #

        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.5, 0.5, 0.0, 0.0, 0],
                [0.0, 0.5, 0.5, 0.0, 0],
                [0.0, 0, 0, 1, 0],
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
            if t == (0, 0):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            elif t == (0, 1):
                inc = np.random.randint(5, 20)
                r1 -= inc
                sequence.append(r1)
            elif t == (1, 1):
                inc = np.random.randint(5, 20)
                r1 -= inc
                sequence.append(r1)
            elif t == (1, 2):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            elif t == (2, 3):
                inc = np.random.randint(5, 20)
                r1 += inc
                sequence.append(r1)
            elif t == (3, 2):
                inc = np.random.randint(5, 20)
                r0 += inc
                sequence.append(r0)
            elif t == (3, 4):
                c = r1 - np.random.randint(5, 20)
                sequence.append(c)
            else:
                tmp = np.random.randint(-20,20)
                sequence.append(tmp)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0

class S10(Language):
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

class S11(Language):
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
            # print(1, i)


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

        if np.random.uniform() < 0.5:
            sequence = [1]
        else:
            sequence = [0]
        transition_matrix = np.array(
            [
                [0, 0.95, 0.05, 0, 0, 0],
                [0, 0.05, 0, 0.95, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0.05, 0.95, 0],
                [0, 0, 0, 0, 0.95, 0.05],
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
        r0 = 1
        r1 = 1
        ts = self.consecutive_pairs(states)
        for t in ts:
            if t == (0, 1):
                inc = np.random.randint(1,5)
                r0 += inc
                sequence.append(r0)
            elif t == (0, 2):
                inc = np.random.randint(1,5)
                r0 -= inc
                sequence.append(r0)
            elif t == (1, 1):
                inc = np.random.randint(1,5)
                r0 += inc
                sequence.append(r0)
            elif t == (1, 3):
                r1 = r0 - np.random.randint(1,5)
                sequence.append(r1)
            elif t == (2, 2):
                inc = np.random.randint(1,5)
                if np.random.uniform() < 0.5:
                    r1 = r1
                else:
                    r1 -= inc
                sequence.append(r1)
            # elif t == (2, 3):
            #     r0 = r1 + np.random.randint(1,5)
            #     sequence.append(r0)
            elif t == (3, 3):
                inc = np.random.randint(1,5)
                r1 -= inc
                sequence.append(r1)
            elif t == (3, 4):
                # if np.random.uniform() < 0.5:
                #     r1 = r1
                # else:
                r0 = r1 + np.random.randint(1,5)
                sequence.append(r0)
            elif t == (4, 4):
                inc = np.random.randint(1,5)
                r0 += inc
                sequence.append(r0)
            elif t == (4, 5):
                r1 = r0 - np.random.randint(1,5)
                sequence.append(r1)
            elif t == (5, 5):
                if np.random.uniform() < 0.3:
                    r1 = r1
                elif np.random.uniform() < 0.6:
                    r1 = r1 + np.random.randint(1,5)
                else:
                    r1 = r1 - np.random.randint(1,5)
                sequence.append(r1)
            elif t == (2,2):
                if np.random.uniform() < 0.3:
                    r0 = r0
                elif np.random.uniform() < 0.6:
                    r0 = r0 + np.random.randint(1,5)
                else:
                    r0 = r0 - np.random.randint(1,5)
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
            # print(1, i)


        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.up_down_up_down_prefix(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # print(2, i)

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


class L2AB(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0', 'r1']
        self.states = 5
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def two_ab(self, lst: List[float]) -> bool:
        # Must have at least two distinct symbols (a and b)
        if len(set(lst)) != 2:
            return False
        # Must have even length (since it's (ab)*)
        if len(lst) % 2 != 0:
            return False
        # Extract the two symbols
        a, b = lst[0], lst[1]
        # Must start with a and alternate a,b,a,b,...
        for i in range(len(lst)):
            expected = a if i % 2 == 0 else b
            if lst[i] != expected:
                return False
        return True

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.two_ab(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            # print(1, i)


        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.two_ab(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # print(2, i)

        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 0.9, 0, 0, 0.1],
                [0, 0, 0.9, 0, 0.1],
                [0, 0, 0, 0.9, 0.1],
                [0.9, 0, 0, 0, 0.1],
                [0, 0.9, 0, 0, 1],

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
        b = np.random.randint(-200, 200)
        while a == b:
            b = np.random.randint(-200, 200)
        for t in ts:
            if t == (0, 1):
                sequence.append(a)
            elif t == (1, 2):
                sequence.append(b)
            elif t == (2, 3):
                sequence.append(a)
            elif t == (3, 0):
                sequence.append(b)
            else:
                c = np.random.randint(-200, 200)
                sequence.append(c)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0


class L3AB(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0','r1']
        self.states = 7
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def three_ab(self, lst: List[float]) -> bool:
        # (ababab)* requires exactly two distinct symbols
        if len(set(lst)) != 2:
            return False
        # Length must be a multiple of 6
        if len(lst) % 6 != 0:
            return False

        a, b = lst[0], lst[1]
        pattern = [a, b, a, b, a, b]  # one block

        # Check that each 6-element block matches the pattern
        for i in range(0, len(lst), 6):
            if lst[i:i+6] != pattern:
                return False
        return True

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.three_ab(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            # print(1, i)


        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.three_ab(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # print(2, i)

        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 0.9, 0, 0, 0, 0, 0.1],
                [0, 0, 0.9, 0, 0, 0, 0.1],
                [0, 0, 0, 0.9, 0, 0, 0.1],
                [0, 0, 0, 0, 0.9, 0, 0.1],
                [0, 0, 0, 0, 0, 0.9, 0.1],
                [0.9, 0, 0, 0, 0, 0, 0.1],
                [0, 0, 0, 0, 0, 0, 1],
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
        b = np.random.randint(-200, 200)
        while a == b:
            b = np.random.randint(-200, 200)
        for t in ts:
            if t == (0, 1):
                sequence.append(a)
            elif t == (1, 2):
                sequence.append(b)
            elif t == (2, 3):
                sequence.append(a)
            elif t == (3, 4):
                sequence.append(b)
            elif t == (4,5):
                sequence.append(a)
            elif t == (5,0):
                sequence.append(b)
            else:
                c = np.random.randint(-200, 200)
                sequence.append(c)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0


class L3A(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 4
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def three_a(self, lst: List[float]) -> bool:
        if len(set(lst)) != 1:
            return False
        else:
            return len(lst) % 3 == 0

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.three_a(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            # print(1, i)


        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.three_a(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # print(2, i)

        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 0.9, 0.1],
                [0.9, 0.0, 0.0, 0.1],
                [0.0, 0.0, 0.0, 1],
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
            elif t == (2, 3):
                b = np.random.randint(-200, 200)
                while a == b:
                    b = np.random.randint(-200, 200)
                sequence.append(b)
            elif t == (2, 0):
                sequence.append(a)
            elif t == (1, 2):
                sequence.append(a)
            else:
                b = np.random.randint(-200, 200)
                while a == b:
                    b = np.random.randint(-200, 200)
                sequence.append(b)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0


class L4A(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 5
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def three_a(self, lst: List[float]) -> bool:
        if len(set(lst)) != 1:
            return False
        else:
            return len(lst) % 4 == 0

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.three_a(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            # print(1, i)


        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.three_a(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # print(2, i)

        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 1, 0, 0, 0],
                [0, 0, 0.9, 0, 0.1],
                [0, 0.0, 0.0, 0.9, 0.1],
                [0.9, 0.0, 0.0, 0, 0.1],
                [0.0, 0.0, 0.0, 0.0, 1],
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
                sequence.append(a)
            elif t == (2, 3):
                sequence.append(a)
            elif t == (3, 0):
                sequence.append(a)
            else:
                b = np.random.randint(-200, 200)
                while a == b:
                    b = np.random.randint(-200, 200)
                sequence.append(b)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0


class L4AB(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0', 'r1']
        self.states = 9
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def four_ab(self, lst: List[float]) -> bool:
        # (ababab)* requires exactly two distinct symbols
        if len(set(lst)) != 2:
            return False
        # Length must be a multiple of 6
        if len(lst) % 6 != 0:
            return False

        a, b = lst[0], lst[1]
        pattern = [a, b, a, b, a, b, a, b]  # one block

        # Check that each 6-element block matches the pattern
        for i in range(0, len(lst), 8):
            if lst[i:i+8] != pattern:
                return False
        return True

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.four_ab(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            # print(1, i)


        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.four_ab(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # print(2, i)

        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 0.9, 0, 0, 0, 0, 0, 0, 0.1],
                [0, 0, 0.9, 0, 0, 0, 0, 0, 0.1],
                [0, 0, 0, 0.9, 0, 0, 0, 0, 0.1],
                [0, 0, 0, 0, 0.9, 0, 0, 0, 0.1],
                [0, 0, 0, 0, 0, 0.9, 0, 0, 0.1],
                [0, 0, 0, 0, 0, 0, 0.9, 0, 0.1],
                [0, 0, 0, 0, 0, 0, 0, 0.9, 0.1],
                [0.9, 0, 0, 0, 0, 0, 0, 0, 0.1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
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
        b = np.random.randint(-200, 200)
        while a == b:
            b = np.random.randint(-200, 200)
        for t in ts:
            if t == (0, 1):
                sequence.append(a)
            elif t == (1, 2):
                sequence.append(b)
            elif t == (2, 3):
                sequence.append(a)
            elif t == (3, 4):
                sequence.append(b)
            elif t == (4,5):
                sequence.append(a)
            elif t == (5,6):
                sequence.append(b)
            elif t == (6,7):
                sequence.append(a)
            elif t == (8,0):
                sequence.append(b)
            else:
                c = np.random.randint(-200, 200)
                sequence.append(c)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0



class L5A(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 6
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def three_a(self, lst: List[float]) -> bool:
        if len(set(lst)) != 1:
            return False
        else:
            return len(lst) % 5 == 0

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.three_a(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            # print(1, i)


        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.three_a(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # print(2, i)

        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0.9, 0, 0, 0.1],
                [0, 0.0, 0.0, 0.9, 0, 0.1],
                [0, 0.0, 0.0, 0, 0.9, 0.1],
                [0.9, 0.0, 0.0, 0.0, 0,  0.1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1]
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
                sequence.append(a)
            elif t == (2, 3):
                sequence.append(a)
            elif t == (3, 4):
                sequence.append(a)
            elif t == (4, 0):
                sequence.append(a)
            else:
                b = np.random.randint(-200, 200)
                while a == b:
                    b = np.random.randint(-200, 200)
                sequence.append(b)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0



class L5A(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 6
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def three_a(self, lst: List[float]) -> bool:
        if len(set(lst)) != 1:
            return False
        else:
            return len(lst) % 5 == 0

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.three_a(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            # print(1, i)


        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.three_a(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # print(2, i)

        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0.9, 0, 0, 0.1],
                [0, 0.0, 0.0, 0.9, 0, 0.1],
                [0, 0.0, 0.0, 0, 0.9, 0.1],
                [0.9, 0.0, 0.0, 0.0, 0,  0.1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1]
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
                sequence.append(a)
            elif t == (2, 3):
                sequence.append(a)
            elif t == (3, 4):
                sequence.append(a)
            elif t == (4, 0):
                sequence.append(a)
            else:
                b = np.random.randint(-200, 200)
                while a == b:
                    b = np.random.randint(-200, 200)
                sequence.append(b)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0


class L6A(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 7
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def three_a(self, lst: List[float]) -> bool:
        if len(set(lst)) != 1:
            return False
        else:
            return len(lst) % 6 == 0

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.three_a(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            # print(1, i)


        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.three_a(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # print(2, i)

        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0.9, 0, 0, 0, 0.1],
                [0, 0.0, 0.0, 0.9, 0, 0, 0.1],
                [0, 0.0, 0.0, 0, 0.9, 0, 0.1],
                [0, 0.0, 0.0, 0.0, 0, 0.9, 0.1],
                [0.9, 0.0, 0.0, 0.0, 0.0, 0,  0.1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0, 1]
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
                sequence.append(a)
            elif t == (2, 3):
                sequence.append(a)
            elif t == (3, 4):
                sequence.append(a)
            elif t == (4, 5):
                sequence.append(a)
            elif t == (5, 0):
                sequence.append(a)
            else:
                b = np.random.randint(-200, 200)
                while a == b:
                    b = np.random.randint(-200, 200)
                sequence.append(b)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0



class L7A(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 8
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def three_a(self, lst: List[float]) -> bool:
        if len(set(lst)) != 1:
            return False
        else:
            return len(lst) % 7 == 0

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.three_a(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            # print(1, i)


        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.three_a(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # print(2, i)

        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 1, 0, 0, 0, 0, 0,0 ],
                [0, 0, 0.9, 0, 0, 0, 0, 0.1],
                [0, 0.0, 0.0, 0.9, 0, 0,0, 0.1],
                [0, 0.0, 0.0, 0, 0.9, 0, 0, 0.1],
                [0, 0.0, 0.0, 0.0, 0, 0.9, 0, 0.1],
                [0, 0.0, 0.0, 0.0, 0.0, 0,0.9,  0.1],
                [0.9, 0.0, 0.0, 0.0, 0.0, 0, 0,0.1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 1]
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
                sequence.append(a)
            elif t == (2, 3):
                sequence.append(a)
            elif t == (3, 4):
                sequence.append(a)
            elif t == (4, 5):
                sequence.append(a)
            elif t == (5, 6):
                sequence.append(a)
            elif t == (6, 0):
                sequence.append(a)
            else:
                b = np.random.randint(-200, 200)
                while a == b:
                    b = np.random.randint(-200, 200)
                sequence.append(b)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0

class L8A(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 9
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def three_a(self, lst: List[float]) -> bool:
        if len(set(lst)) != 1:
            return False
        else:
            return len(lst) % 8 == 0

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.three_a(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            # print(1, i)


        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.three_a(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # print(2, i)

        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 1, 0, 0, 0, 0, 0, 0,0 ],
                [0, 0, 0.9, 0, 0, 0, 0, 0, 0.1],
                [0, 0.0, 0.0, 0.9, 0, 0,0, 0, 0.1],
                [0, 0.0, 0.0, 0, 0.9, 0, 0, 0, 0.1],
                [0, 0.0, 0.0, 0.0, 0, 0.9, 0, 0, 0.1],
                [0, 0.0, 0.0, 0.0, 0.0, 0,0.9,  0, 0.1],
                [0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.9, 0.1],
                [0.9, 0.0, 0.0, 0.0, 0.0, 0, 0, 0,0.1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 1]
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
                sequence.append(a)
            elif t == (2, 3):
                sequence.append(a)
            elif t == (3, 4):
                sequence.append(a)
            elif t == (4, 5):
                sequence.append(a)
            elif t == (5, 6):
                sequence.append(a)
            elif t == (6, 7):
                sequence.append(a)
            elif t == (7, 0):
                sequence.append(a)
            else:
                b = np.random.randint(-200, 200)
                while a == b:
                    b = np.random.randint(-200, 200)
                sequence.append(b)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0



class L9A(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 10
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def three_a(self, lst: List[float]) -> bool:
        if len(set(lst)) != 1:
            return False
        else:
            return len(lst) % 9 == 0

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.three_a(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            # print(1, i)


        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.three_a(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # print(2, i)

        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 1, 0, 0, 0, 0, 0, 0,0 ,0 ],
                [0, 0, 0.9, 0, 0, 0, 0, 0, 0, 0.1],
                [0, 0.0, 0.0, 0.9, 0, 0,0, 0, 0, 0.1],
                [0, 0.0, 0.0, 0, 0.9, 0, 0, 0, 0, 0.1],
                [0, 0.0, 0.0, 0.0, 0, 0.9, 0, 0, 0, 0.1],
                [0, 0.0, 0.0, 0.0, 0.0, 0,0.9,  0, 0, 0.1],
                [0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.9, 0, 0.1],
                [0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.9, 0.1],
                [0.9, 0.0, 0.0, 0.0, 0.0, 0, 0, 0,0, 0.1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 1]
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
                sequence.append(a)
            elif t == (2, 3):
                sequence.append(a)
            elif t == (3, 4):
                sequence.append(a)
            elif t == (4, 5):
                sequence.append(a)
            elif t == (5, 6):
                sequence.append(a)
            elif t == (6, 7):
                sequence.append(a)
            elif t == (7, 8):
                sequence.append(a)
            elif t == (8, 0):
                sequence.append(a)
            else:
                b = np.random.randint(-200, 200)
                while a == b:
                    b = np.random.randint(-200, 200)
                sequence.append(b)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0

class L10A(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 11
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def three_a(self, lst: List[float]) -> bool:
        if len(set(lst)) != 1:
            return False
        else:
            return len(lst) % 10 == 0

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.three_a(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            # print(1, i)


        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.three_a(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # print(2, i)

        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 1, 0, 0, 0, 0, 0, 0,0 ,0, 0 ],
                [0, 0, 0.99, 0, 0, 0, 0, 0, 0, 0, 0.01],
                [0, 0.0, 0.0, 0.99, 0, 0,0, 0, 0, 0, 0.01],
                [0, 0.0, 0.0, 0, 0.99, 0, 0, 0, 0, 0, 0.01],
                [0, 0.0, 0.0, 0.0, 0, 0.99, 0, 0, 0, 0, 0.01],
                [0, 0.0, 0.0, 0.0, 0.0, 0,0.99,  0, 0, 0, 0.01],
                [0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.99, 0, 0, 0.01],
                [0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.99, 0, 0.01],
                [0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.99, 0.01],
                [0.99, 0.0, 0.0, 0.0, 0.0, 0, 0, 0,0, 0, 0.01],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0,0, 1]
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
                sequence.append(a)
            elif t == (2, 3):
                sequence.append(a)
            elif t == (3, 4):
                sequence.append(a)
            elif t == (4, 5):
                sequence.append(a)
            elif t == (5, 6):
                sequence.append(a)
            elif t == (6, 7):
                sequence.append(a)
            elif t == (7, 8):
                sequence.append(a)
            elif t == (8, 9):
                sequence.append(a)
            elif t == (9, 0):
                sequence.append(a)
            else:
                b = np.random.randint(-200, 200)
                while a == b:
                    b = np.random.randint(-200, 200)
                sequence.append(b)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0

class L11A(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 12
        self.edge_limit_num = 1
        self.edge_limit_type = "exactly"

    def three_a(self, lst: List[float]) -> bool:
        if len(set(lst)) != 1:
            return False
        else:
            return len(lst) % 11 == 0
    
    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []
        for i in range(num_pos):
            sequence, last_guy = self.generate_sequence(max_length)
            while not self.three_a(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            # print(1, sequence)
            

        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.three_a(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # print(2, sequence)
            
        return pos_sample, neg_sample
    
    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 1, 0, 0, 0, 0, 0, 0,0 ,0, 0,0],
                [0, 0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0,0.1],
                [0, 0.0, 0.0, 0.9, 0, 0,0, 0, 0, 0,0, 0.1],
                [0, 0.0, 0.0, 0, 0.9, 0, 0, 0, 0, 0,0, 0.1],
                [0, 0.0, 0.0, 0.0, 0, 0.9, 0, 0, 0, 0, 0,0.1],
                [0, 0.0, 0.0, 0.0, 0.0, 0,0.9,  0, 0, 0, 0,0.1],
                [0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.9, 0, 0, 0,0.1],
                [0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.9, 0, 0,0.1],
                [0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.9, 0,0.1],
                [0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0.9,0.1],
                [0.9, 0.0, 0.0, 0.0, 0.0, 0, 0, 0,0,0, 0, 0.1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 1]
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
        a = np.random.randint(-400, 400)
        for t in ts:
            if t == (0, 1):
                sequence.append(a)
            elif t == (1, 2):
                sequence.append(a)
            elif t == (2, 3):
                sequence.append(a)
            elif t == (3, 4):
                sequence.append(a)
            elif t == (4, 5):
                sequence.append(a)
            elif t == (5, 6):   
                sequence.append(a)
            elif t == (6, 7):   
                sequence.append(a)
            elif t == (7, 8):   
                sequence.append(a)
            elif t == (8, 9):   
                sequence.append(a)
            elif t == (9, 10):
                sequence.append(a)
            elif t == (10, 0):
                sequence.append(a)
            else:
                b = np.random.randint(-400, 400)
                while a == b:
                    b = np.random.randint(-400, 400)
                sequence.append(b)
        if ts:
            return sequence, ts[-1][1]
        else:
            return sequence, 0
        
def new_generate_language(language_name: str, num_pos: int = 100, num_neg: int = 0, max_len: int = 30) -> List[str]:
    # Discover all subclasses of the Language class
    language_classes = {cls.__name__: cls for cls in Language.__subclasses__()}

    if language_name not in language_classes:
        raise ValueError(f"Unsupported language: {language_name}. Available languages: {list(language_classes.keys())}")

    # Initialize the selected language
    language_class = language_classes[language_name]
    lang = language_class(num_pos, num_neg, max_len)


    # Generate new samples
    lang.pos, lang.neg = lang.generate_samples(num_pos, num_neg, max_len)

    for i in lang.pos:
        print(','.join(str(x) for x in i))
    for i in lang.neg:
        print(','.join(str(x) for x in i))
    return lang

def random_sample_generation(language_name: str, num_pos: int = 500, num_neg: int = 500, max_len: int = 30) -> List[str]:
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

    # Generate language samples
    #     print(f"Generating {args.num_pos} positive and {args.num_neg} negative samples for language '{args.language}'")
    #     print(f"Maximum sequence length: {args.max_len}")

    try:
        lang = new_generate_language(args.language, num_pos=args.num_pos, num_neg=args.num_neg, max_len=args.max_len)

        # Print statistics
        if hasattr(lang, 'pos') and lang.pos:
            max_length = max([len(i) for i in lang.pos])
        #             print(f"Maximum length of generated positive sequences: {max_length}")

        pos_count = len(lang.pos) if hasattr(lang, 'pos') else 0
        neg_count = len(lang.neg) if hasattr(lang, 'neg') else 0

        # print(f"Successfully generated {pos_count} positive and {neg_count} negative samples.")
        # print(f"Dataset saved to: dataset/{args.language}.csv")

    except Exception as e:
        print(f"Error generating language samples: {e}")
        sys.exit(1)


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
    import random
    random.shuffle(lang.pos)
    random.shuffle(lang.neg)
    return (lang.pos, lang.neg)


if __name__ == "__main__":
    main()
    