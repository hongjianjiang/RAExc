import numpy as np
from typing import List, Tuple
import argparse
import sys
from typing import List, Optional
import random


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
    
    def membership(self, lst: List[float]) -> bool:
        pass

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []

        for inx in range(num_pos):
            sequence, nl = self.generate_sequence(max_length)
            while not self.membership(sequence) or sequence in pos_sample:
                sequence, nl = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            
        for inx in range(num_neg):
            sequence, nl = self.generate_sequence(max_length)
            while self.membership(sequence) or sequence in neg_sample:
                sequence, nl = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            
        return pos_sample, neg_sample

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        pass

    def consecutive_pairs(self, input_list: List[int]) -> List[Tuple[int, int]]:
        return [(input_list[i], input_list[i + 1]) for i in range(len(input_list) - 1)]



class L1(Language):
    def __init__(self, num_pos, num_neg, max_len):
        super().__init__(num_pos, num_neg, max_len)
        self.states = 2
    
    def membership(self, lst: List[float]) -> bool:
        return len(set(lst)) == 1
    
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
        self.regs = ['r0']
        self.states = 4
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def membership(self, lst: List[float]) -> bool:
        if len(set(lst)) == 2:
            if lst[0] != lst[1]:
                return lst == [lst[0], lst[1]] * (len(lst) // 2)
            else:
                return False
        elif lst == []:
            return True
        else:
            return False
        
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

    def membership(self, lst: List[float]) -> bool:
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
                sequence.append(a)
            elif t == (0, 3):
                sequence.append(b)
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
            return sequence, ts[-1][1]
        else:
            return sequence, 0

class L4(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):

        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0', 'r1']
        self.states = 4
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def membership(self, sequence: List[float]) -> bool:
        for i in range(0, len(sequence) - 2):
            if sequence[i] == sequence[i + 1] and  sequence[i + 1] == sequence[i + 2]:
                return False
        return True
 
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

        r0 = np.random.randint(-200, 200)
        # b = np.random.randint(-200, 200)
        # while a == b :
        #     b = np.random.randint(-200, 200)

        ts = self.consecutive_pairs(states)
        for t in ts:
            if t == (0, 1):
                sequence.append(r0)
            elif t == (1, 1):
                c = np.random.randint(-200, 200) 
                while c == r0:
                    c = np.random.randint(-200, 200)
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
                c = np.random.randint(-200, 200) 
                while c == r0:
                    c = np.random.randint(-200, 200)
                r0 = c
                sequence.append(c)
            elif t == (2, 3):
                sequence.append(r0)
            elif t == (3, 3):
                c = np.random.randint(-200, 200)
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
        self.edge_limit_type = "at_most"

    def membership(self, sequence: List[float]) -> bool:
        if len(set(sequence)) == 1:
            return len(sequence) % 2 == 0
        elif len(set(sequence)) > 2 or len(set(sequence)) == 0:
            return False
        a, b = list(set(sequence))[:2]
        if sequence.count(a) % 2 ==0 and sequence.count(b) % 2 == 0:
            return True
        else:
            return False
 
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

        a = np.random.randint(0, 10)
        b = np.random.randint(0, 10)
        while a == b:
            b = np.random.randint(0, 10)
        ts = self.consecutive_pairs(states)
        for t in ts:
            if t == (0, 1):
                sequence.append(a)
            elif t == (0, 3):
                sequence.append(b)
            elif t == (1, 0):
                sequence.append(a)
            elif t == (1, 2):
                sequence.append(b)
            elif t == (2, 1):
                sequence.append(b)
            elif t == (2, 3):
                sequence.append(a)
            elif t == (3, 2):
                sequence.append(a)
            elif t == (3, 0):
                sequence.append(b)
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

    def membership(self, seq: list) -> bool:
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

    def membership(self, lst: List[float], eps: float = 1e-9) -> bool:
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

    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.7, 0.3, 0, 0, 0],
                [0, 0.7, 0.3, 0, 0],
                [0, 0, 0.7, 0.3, 0],
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
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def membership(self, arr: List[float]) -> bool:
        for i in range(1, len(arr)):
            if arr[i] <= arr[i - 1]:
                return False
        return True
 
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
                b = a - np.random.randint(3, 20)
                sequence.append(b)
             
        return sequence, ts[-1][1] if ts else 0


class Sn(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int, n: int = 10):
        super().__init__(num_pos, num_neg, max_len)
        self.n = n
        self.regs = ['r0']
        self.states = n + 2
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def membership(self, arr: List[float]) -> bool:
        """Check if array is length n AND strictly increasing."""
        if len(arr) != self.n:
            return False
        for i in range(1, len(arr)):
            if arr[i] <= arr[i - 1]: # Must be strictly greater
                return False
        return True
 
    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        """
        Generates a sequence where the first n elements are strictly increasing.
        If the sequence continues beyond n, the automaton goes to the sink state.
        """
        n = self.n
        current_state = 0
        sequence = []

        length = np.random.randint(1, max_length + 1)

        # initial value
        prev = np.random.randint(-10000, 10000)

        for i in range(length):
            if current_state < n:
                # generate strictly increasing value
                step = np.random.randint(3, 10)
                value = prev + step

                sequence.append(value)
                prev = value
                current_state += 1

            else:
                # sink behavior: break increasing property
                value = prev - np.random.randint(3, 10)
                sequence.append(value)
                current_state = n + 1
                prev = value

        return sequence, current_state
    
class S2(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 2
        self.edge_limit_num = 1
        self.edge_limit_type = "exactly"

    def membership(self, arr: List[float]) -> bool:
        for i in range(1, len(arr)):
            if arr[i] >= arr[i - 1]:
                return False
        return True

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
                inc = np.random.randint(5, 20)
                a -=  inc
                sequence.append(a)
            else:
                inc = np.random.randint(10, 20)
                choice = np.random.choice([2, inc])
                b = a + choice

                sequence.append(b)

        return sequence, len(sequence)
    
class S3(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 2
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def membership(self, arr: List[float]) -> bool:
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
        return sequence, len(sequence)

class S4(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 2
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def membership(self, arr: List[float]) -> bool:
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
        return sequence, len(sequence)

class S5(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 4
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def membership(self, lst: List[float]) -> bool:
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

    def membership(self, lst: List[float]) -> bool:
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
 
    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0.0, 0.5, 0.5, 0, 0, 0],
                [0.0, 0, 0.5, 0.5, 0, 0],
                [0.0, 0, 0, 0.5, 0.5, 0],
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
        self.regs = ['r0']
        self.states = 5
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def membership(self, lst: List[float]) -> bool:
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

    def membership(self, lst: List[float]) -> bool:
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
 
    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0.0, 0.5, 0.5, 0, 0, 0],
                [0.0, 0, 0.5, 0.5, 0, 0],
                [0.0, 0, 0, 0.5, 0.5, 0],
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
        r1 = 0 
        ts = self.consecutive_pairs(states)
        for t in ts:
            # print(t)
            if t == (0, 1):
                inc = np.random.randint(2, 5)
                r0 += inc
                sequence.append(r0)
            elif t == (1, 1):
                # p = np.random.uniform(0,1)
                inc = np.random.randint(2, 5)
                # if p < 0.5:
                #     r0 = r0 + 2
                # else:
                r0 = r0 + inc
                sequence.append(r0)
            elif t == (1, 2):
                inc = np.random.randint(2, 5)
                r1 -=  inc
                sequence.append(r1)
            elif t == (2, 2):
                inc = np.random.randint(2, 5)
                r1 -= inc
                sequence.append(r1)
            elif t == (2, 3):
                r0 = r0 + np.random.randint(2, 5)
                sequence.append(r0)
            elif t == (3,3):
                inc = np.random.randint(2,5)
                r0 = r0 + inc
                sequence.append(r0)
            elif t == (3,4):
                r0 = r0 - np.random.randint(2,5)
                sequence.append(r0)
            elif t == (4,4):
                inc = np.random.randint(2,5)
                r0 = r0 - inc
                sequence.append(r0)
            elif t == (4,5):
                r0 = r0 + np.random.randint(2,5)
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
        

class S9(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0', 'r1']
        self.states = 5
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def membership(self, lst: List[float]) -> bool:
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
                inc = np.random.randint(5, 7)
                r0 += inc
                sequence.append(r0)
            elif t == (1, 1):
                inc = np.random.randint(5, 7)
                r0 += inc
                sequence.append(r0)
            elif t == (1, 2):
                inc = np.random.randint(5, 7)
                r1 -=  inc
                sequence.append(r1)
            elif t == (2, 2):
                inc = np.random.randint(5, 7)
                r1 -= inc
                sequence.append(r1)
            elif t == (2, 3):
                inc = np.random.randint(5, 7)
                r0 += inc
                sequence.append(r0)
            elif t == (3, 2):
                inc = np.random.randint(5, 7)
                r1 -= inc
                sequence.append(r1)
            elif t == (3, 4):         
                c = r0 + np.random.randint(5, 7)
                sequence.append(c)
            else:
                tmp = np.random.randint(-100,100)
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

    def membership(self, lst: List[float]) -> bool:
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
            elif t == (0, 4):
                inc = np.random.randint(5, 20)
                r0 -= inc
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

    def membership(self, lst: List[float]) -> bool:
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
            while not self.membership(sequence) or sequence in pos_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            pos_sample.append(sequence)
            # print(1, i)
            

        for i in range(num_neg):
            sequence, last_guy = self.generate_sequence(max_length)
            while self.membership(sequence) or sequence in neg_sample:
                sequence, last_guy = self.generate_sequence(max_length)
            neg_sample.append(sequence)
            # print(2, i)
            
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
            elif t == (0, 4):
                inc = np.random.randint(5, 20)
                r0 -= inc
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
  
class S12(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0', 'r1']
        self.states = 6
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def membership(self, lst: List[float]) -> bool:
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
 
    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.7, 0.3, 0.0, 0, 0, 0 ],
                [0.0, 0.7, 0.3, 0, 0, 0],
                [0.0, 0.0, 0.7, 0.3, 0, 0],
                [0.0, 0.0, 0.0, 0.7, 0.3, 0],
                [0.0, 0.0, 0.0, 0.0, 0.7, 0.3],
                [0.0, 0.0, 0.0, 0.0, 0, 1]
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
        
class S13(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0', 'r1']
        self.states = 7
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def membership(self, lst: List[float]) -> bool:
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
 
    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.7, 0.3, 0.0, 0, 0, 0, 0],
                [0.0, 0.7, 0.3, 0, 0, 0, 0],
                [0.0, 0.0, 0.7, 0.3, 0, 0, 0],
                [0.0, 0.0, 0.0, 0.7, 0.3, 0, 0],
                [0.0, 0.0, 0.0, 0.0, 0.7, 0.3, 0],
                [0.0, 0.0, 0.0, 0.0, 0, 0.7, 0.3],
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

class S14(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0', 'r1']
        self.states = 8
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def membership(self, lst: List[float]) -> bool:
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
 
    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        sequence = []
        transition_matrix = np.array(
            [
                [0.7, 0.3, 0.0, 0, 0, 0, 0, 0],
                [0.0, 0.7, 0.3, 0, 0, 0, 0, 0],
                [0.0, 0.0, 0.7, 0.3, 0, 0, 0, 0],
                [0.0, 0.0, 0.0, 0.7, 0.3, 0, 0, 0],
                [0.0, 0.0, 0.0, 0.0, 0.7, 0.3, 0, 0],
                [0.0, 0.0, 0.0, 0.0, 0, 0.7, 0.3, 0],
                [0.0, 0.0, 0.0, 0.0, 0, 0, 0.7, 0.3],
                [0.0, 0.0, 0.0, 0.0, 0, 0, 0, 1],
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

class L2A(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 3
        self.edge_limit_num = 1
        self.edge_limit_type = "exactly"

    def membership(self, lst: List[float]) -> bool:
        if len(set(lst)) != 1:
            return False
        else:
            return len(lst) % 2 == 0
 
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
 
 
class DistinctK(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.k = 7
        self.regs = [f"r{i}" for i in range(self.k)]
        self.states = self.k + 2
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def membership(self, lst: List[float]) -> bool:
        if len(lst) != self.k:
            return False
        return len(set(lst)) == self.k


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
        symbols = []
        while len(symbols) < self.k:
            x = np.random.randint(-200, 200)
            if x not in symbols:
                symbols.append(x)

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
        
class LnA(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int, n: int = 10):
        super().__init__(num_pos, num_neg, max_len)
        self.n = n
        self.regs = ['r0']
        self.states = n + 1
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"

    def membership(self, lst: List[float]) -> bool:
        if len(set(lst)) != 1:
            return False
        else:
            return len(lst) == self.n
 
    def generate_sequence(self, max_length: int) -> Tuple[List[float], int]:
        """
        Generates a sequence that follows a path toward acceptance or the sink.
        To get a positive sample, we must follow the chain for exactly n steps.
        """
        n = self.n
        # States: 0...n-1 (chain), n (accepting/terminal), n+1 (sink)
        current_state = 0
        sequence = []
        
        # We pick a fixed 'a' value for the potential positive sequence
        a = np.random.randint(-10000, 10000)
        
        # Decide length: to get variety, we pick a length up to max_length
        # Note: three_a only returns true if length == self.n
        length = np.random.randint(1, max_length + 1)
        
        for i in range(length):
            # TRANSITION LOGIC:
            # If we are in the chain (0 to n-1), we move to the next state
            if current_state < n:
                # To make it "exactly n", the n-th transition must land in the accept state
                # Otherwise, we stay in the chain.
                # Here we simulate the logic of the target automaton:
                current_state += 1
                sequence.append(a)
            else:
                # If we are already at n or in the sink (n+1), any further input
                # takes us to the sink state with a different value.
                current_state = n + 1
                b = np.random.randint(-10000, 10000)
                while b == a:
                    b = np.random.randint(-10000, 10000)
                sequence.append(b)
        
        return sequence, current_state
 
class mnist(Language):
    def __init__(self, num_pos: int, num_neg: int, max_len: int):
        super().__init__(num_pos, num_neg, max_len)
        self.regs = ['r0']
        self.states = 4
        self.edge_limit_num = 2
        self.edge_limit_type = "at_most"
        self.ic = [8]
        # Load samples from CSV
        self.pos, self.neg = self.load_from_csv()

    def load_from_csv(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Load positive and negative samples from mnist_test.csv, where the first element is the label and the next 28 are the input vector (integers)."""
        import os
        csv_path = os.path.join(os.path.dirname(__file__),  "mnist_test.csv")
        print(f"Loading samples from {csv_path}...")
        pos_samples = []
        neg_samples = []
        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 29:
                    continue  # skip malformed lines
                label = int(parts[0])
                input_vec = [int(x) for x in parts[1:29]]
                if label == 1:
                    pos_samples.append(input_vec)
                else:
                    neg_samples.append(input_vec)
        return pos_samples, neg_samples

    def generate_samples(
            self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        # Draw a uniform random subset of the loaded samples, so that repeated
        # calls (e.g. fresh validation samples) are independent draws
        import random
        pos = random.sample(self.pos, min(num_pos, len(self.pos))) if num_pos > 0 else list(self.pos)
        neg = random.sample(self.neg, min(num_neg, len(self.neg))) if num_neg > 0 else list(self.neg)
        return pos, neg

def new_generate_language(language_name: str, num_pos: int = 100, num_neg: int = 0, max_len: int = 20, n: int = 3) -> List[str]:
    # Discover all subclasses of the Language class
    language_classes = {cls.__name__: cls for cls in Language.__subclasses__()}

    if language_name not in language_classes:
        raise ValueError(f"Unsupported language: {language_name}. Available languages: {list(language_classes.keys())}")

    # Initialize the selected language
    language_class = language_classes[language_name]
    lang = language_class(num_pos, num_neg, max_len, n) if language_name == "LnA"  or language_name =='Sn' else language_class(num_pos, num_neg, max_len)


    # Generate new samples
    lang.pos, lang.neg = lang.generate_samples(num_pos, num_neg, max_len)
    # for i in lang.pos:
    #     print(','.join(str(x) for x in i))
    # for i in lang.neg:
    #     print(','.join(str(x) for x in i))
    return lang

def random_sample_generation(language_name: str, num_pos: int = 500, num_neg: int = 500, max_len: int = 10, n: int = 4) -> List[str]:
    # Discover all subclasses of the Language class
    language_classes = {cls.__name__: cls for cls in Language.__subclasses__()}

    if language_name not in language_classes:
        raise ValueError(f"Unsupported language: {language_name}. Available languages: {list(language_classes.keys())}")

    # Initialize the selected language
    language_class = language_classes[language_name]
    if language_name == "LnA" or language_name == 'Sn': 
        lang = language_class(num_pos, num_neg, max_len, n)
    else:
        lang = language_class(num_pos, num_neg, max_len)

    # Generate new samples
    lang.pos, lang.neg = lang.generate_samples(num_pos, num_neg, max_len)
    import random
    random.shuffle(lang.pos)
    random.shuffle(lang.neg)
    return (lang.pos, lang.neg)

def membership_query(sample, language_name):
    language_classes = {cls.__name__: cls for cls in Language.__subclasses__()}
    if language_name not in language_classes:
        raise ValueError(f"Unsupported language: {language_name}. Available languages: {list(language_classes.keys())}")
    language_class = language_classes[language_name]
    lang = language_class(0, 0, 0)  # We only need the membership function
    return lang.membership(sample)

def _cli() -> None:
    """Print fresh samples, one comma-separated sequence per line.

    Used by the active learner (RNNSimulator.gen_test_samples) for its
    sampling-based equivalence checks.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("language")
    parser.add_argument("--num-pos", type=int, required=True)
    parser.add_argument("--num-neg", type=int, required=True)
    parser.add_argument("--max-len", type=int, required=True)
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args()
    pos, neg = random_sample_generation(args.language, args.num_pos, args.num_neg,
                                        args.max_len, n=args.n)
    for sample in pos + neg:
        print(','.join(str(x) for x in sample))


if __name__ == "__main__" and len(sys.argv) > 1:
    _cli()
elif __name__ == "__main__":
    # generate dataset for LnA with n = 100, and max_len = 110, and save into a csv file, add the label in the end
    pos, neg = random_sample_generation("LnA", num_pos=20000, num_neg=20000, max_len=110, n=100)
    
    # shuffle and split into train and test, 80% for train, 20% for test
    train_pos = pos[:int(0.8 * len(pos))]
    test_pos = pos[int(0.8 * len(pos)):]
    train_neg = neg[:int(0.8 * len(neg))]
    test_neg = neg[int(0.8 * len(neg)):]
    
    with open("LnA_train.csv", "w") as f:
        for i, sample in enumerate(train_pos):
            f.write(','.join(str(x) for x in sample) + ',1\n')
        for i, sample in enumerate(train_neg):
            f.write(','.join(str(x) for x in sample) + ',0\n')
    
    with open("LnA_test.csv", "w") as f:
        for i, sample in enumerate(test_pos):
            f.write(','.join(str(x) for x in sample) + ',1\n')
        for i, sample in enumerate(test_neg):
            f.write(','.join(str(x) for x in sample) + ',0\n')
    
