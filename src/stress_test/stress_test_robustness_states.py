#!/usr/bin/env python3
"""
Stress Test: Robustness Check Efficiency based on Number of States

This script evaluates how robustness checking performance scales with 
the number of states in the automaton. It uses `create_exactly_n_same_word_ra`
to generate automata with 1 to N states and measures:
- Robustness check time
- Memory usage (optional)
- Scalability characteristics

Usage:
    python stress_test_robustness_states.py --min_states 1 --max_states 10
    python stress_test_robustness_states.py --states 1 2 3 4 5 6 7 8 9 10
    python stress_test_robustness_states.py --quick
"""

import argparse
import csv
import os
import sys
import time
import traceback
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import random
import math

sys.path.append(os.path.dirname(__file__))

import z3
from register_automata.ra import RegisterAutomata, Transition, load_smt_dra
# from register_automata.metric_automata import create_l1_raa, create_hamming_raa, create_edit_raa
from register_automata.raa import RAA
from robustness_check import get_metric_raa
from register_automata.robustness import RobustnessChecker


def create_strictly_increasing_ra(n: int) -> RegisterAutomata:
    """
    Accepts exactly length n sequences where symbols are strictly increasing.
    Logic: x_1 < x_2 < ... < x_n
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    curr = z3.Real("curr")
    r0 = z3.Real("r0")

    sink = n + 1
    states = n + 2 
    final_states = [n]
    register_vars = [r0]

    transitions = []

    # 1. Sink self-loop
    transitions.append(
        Transition(curr, sink, True, {}, sink)
    )

    if n == 0:
        transitions.append(Transition(curr, 0, True, {}, sink))
        return RegisterAutomata(states, transitions, register_vars, final_states=[0])

    # 2. Step 0: Read first symbol, store in r0
    transitions.append(
        Transition(curr, start_state=0, guard=True, assignment={"r0": "c"}, end_state=1)
    )

    # 3. Steps 1 to n-1: Enforce curr > r0 and UPDATE r0
    for i in range(1, n):
        # Success: strictly increasing
        transitions.append(
            Transition(
                curr,
                start_state=i,
                guard=(curr > r0),
                assignment={"r0": "c"}, # Crucial: update r0 to the new peak
                end_state=i + 1
            )
        )
        # Failure: not strictly increasing
        transitions.append(
            Transition(curr, start_state=i, guard=(curr <= r0), assignment={}, end_state=sink)
        )

    # 4. Post-acceptance: any further input goes to sink
    transitions.append(
        Transition(curr, start_state=n, guard=True, assignment={}, end_state=sink)
    )

    return RegisterAutomata(states, transitions, register_vars, final_states)

def generate_test_vectors_increasing(n: int, num_vectors: int = 100) -> List[List[float]]:
    vectors = []
    for _ in range(num_vectors):
        vec = sorted(random.sample(range(-1000, 1000), n))
        vectors.append([float(x) for x in vec])
    return vectors

def create_exactly_n_same_word_ra(n: int) -> RegisterAutomata:
    """
    Accepts exactly length n sequences where all symbols are equal.
    - n == 0: accepts only the empty word
    - n >= 1: accepts sequences [x, x, ..., x] (n times) for any real x

    States: 0..n are "progress" states, final state is n, sink is n+1.
    Register: r0 stores the first symbol.
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    curr = z3.Real("curr")
    r0 = z3.Real("r0")

    sink = n + 1
    states = n + 2  # 0..n plus sink
    final_states = [n]
    register_vars = [r0]

    transitions = []

    # Sink self-loop: once rejected, always rejected
    transitions.append(
        Transition(curr, sink, True, {}, sink)
    )

    if n == 0:
        # Accept empty word: final state is 0.
        # Any input symbol from state 0 goes to sink.
        transitions.append(
            Transition(curr, 0, True, {}, sink)
        )
        return RegisterAutomata(states, transitions, register_vars, final_states=[0])

    # Step 0: read first symbol, store it in r0, go to state 1
    transitions.append(
        Transition(
            curr,
            start_state=0,
            guard=True,
            assignment={"r0": "c"},   # "c" means copy current symbol into r0
            end_state=1
        )
    )

    # Steps 1..n-1: enforce curr == r0
    for i in range(1, n):
        transitions.append(
            Transition(
                curr,
                start_state=i,
                guard=(curr == r0),
                assignment={},         # keep r0
                end_state=i + 1
            )
        )
        # mismatch -> sink (keeps automaton total/deterministic)
        transitions.append(
            Transition(
                curr,
                start_state=i,
                guard=(curr != r0),
                assignment={},
                end_state=sink
            )
        )

    # After consuming n symbols (state n is final), any further symbol -> sink
    transitions.append(
        Transition(curr, start_state=n, guard=True, assignment={}, end_state=sink)
    )

    return RegisterAutomata(states, transitions, register_vars, final_states)


def create_n_register_ra(n: int) -> RegisterAutomata:
    """
    Creates an automaton with exactly n registers for testing register scaling.
    
    Accepts sequences of length n where all symbols are equal.
    Each symbol is stored in a different register (r0, r1, ..., r(n-1)).
    
    - n == 0: accepts only the empty word, no registers
    - n == 1: accepts [x] for any x, stores in r0
    - n >= 2: accepts [x, x, ..., x] (n times), stores in r0..r(n-1)
              and verifies each new symbol equals r0
    
    States: 0..n are "progress" states, final state is n, sink is n+1.
    Registers: r0, r1, ..., r(n-1)
    
    Args:
        n: Number of registers (and also sequence length accepted)
    
    Returns:
        RegisterAutomata with n registers
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    curr = z3.Real("curr")
    
    sink = n + 1
    states = n + 2  # 0..n plus sink
    final_states = [n]
    
    # Create n registers
    register_vars = [z3.Real(f"r{i}") for i in range(max(1, n))]  # At least 1 register
    r0 = register_vars[0] if register_vars else z3.Real("r0")
    
    transitions = []

    # Sink self-loop: once rejected, always rejected
    transitions.append(
        Transition(curr, sink, True, {}, sink)
    )

    if n == 0:
        # Accept empty word: final state is 0.
        # Any input symbol from state 0 goes to sink.
        transitions.append(
            Transition(curr, 0, True, {}, sink)
        )
        return RegisterAutomata(states, transitions, [z3.Real("r0")], final_states=[0])

    # Step 0: read first symbol, store it in r0, go to state 1
    transitions.append(
        Transition(
            curr,
            start_state=0,
            guard=True,
            assignment={"r0": "c"},   # "c" means copy current symbol into r0
            end_state=1
        )
    )

    # Steps 1..n-1: enforce curr == r0 AND store in ri
    for i in range(1, n):
        # Match transition: curr == r0, store in r_i
        assignment = {f"r{i}": "c"}  # Store current symbol in register i
        transitions.append(
            Transition(
                curr,
                start_state=i,
                guard=(curr == r0),
                assignment=assignment,
                end_state=i + 1
            )
        )
        # Mismatch -> sink (keeps automaton total/deterministic)
        transitions.append(
            Transition(
                curr,
                start_state=i,
                guard=(curr != r0),
                assignment={},
                end_state=sink
            )
        )

    # After consuming n symbols (state n is final), any further symbol -> sink
    transitions.append(
        Transition(curr, start_state=n, guard=True, assignment={}, end_state=sink)
    )

    return RegisterAutomata(states, transitions, register_vars, final_states)


def create_n_register_distinct_ra(n: int) -> RegisterAutomata:
    """
    Creates an automaton with n registers that accepts sequences where 
    the first n symbols are stored and all subsequent symbols must NOT 
    match any stored value (i.e., must be distinct from all registers).
    
    - n == 0: accepts any sequence (no checks)
    - n == 1: accepts sequences where symbol 2+ differ from symbol 1
    - n >= 2: stores first n symbols, then requires all following symbols 
              to be different from all stored values
    
    States: 
        0..n-1: reading and storing phase
        n: accepting state (after storing n symbols)
        n+1: sink state
    
    Registers: r0, r1, ..., r(n-1)
    
    Args:
        n: Number of registers
    
    Returns:
        RegisterAutomata with n registers
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    
    if n == 0:
        # Accept any sequence
        curr = z3.Real("curr")
        register_vars = [z3.Real("r0")]
        transitions = [
            Transition(curr, 0, True, {}, 0)  # Self-loop accepting everything
        ]
        return RegisterAutomata(1, transitions, register_vars, final_states=[0])

    curr = z3.Real("curr")
    
    # Create n registers
    register_vars = [z3.Real(f"r{i}") for i in range(n)]
    
    sink = n + 1
    states = n + 2
    final_states = [n]  # State n is accepting
    
    transitions = []
    
    # Sink self-loop
    transitions.append(
        Transition(curr, sink, True, {}, sink)
    )
    
    # Phase 1: Store first n symbols (states 0 to n-1)
    for i in range(n):
        # Store current symbol in register i, move to next state
        transitions.append(
            Transition(
                curr,
                start_state=i,
                guard=True,
                assignment={f"r{i}": "c"},
                end_state=i + 1
            )
        )
    
    # Phase 2: In state n (accepting), any new symbol must differ from ALL registers
    # Build guard: curr != r0 AND curr != r1 AND ... AND curr != r(n-1)
    if n == 1:
        differ_guard = (curr != register_vars[0])
    else:
        differ_guard = z3.And(*[curr != register_vars[i] for i in range(n)])
    
    # Stay in accepting state if all different
    transitions.append(
        Transition(
            curr,
            start_state=n,
            guard=differ_guard,
            assignment={},
            end_state=n
        )
    )
    
    # Go to sink if matches any register
    if n == 1:
        match_guard = (curr == register_vars[0])
    else:
        match_guard = z3.Or(*[curr == register_vars[i] for i in range(n)])
    
    transitions.append(
        Transition(
            curr,
            start_state=n,
            guard=match_guard,
            assignment={},
            end_state=sink
        )
    )
    
    return RegisterAutomata(states, transitions, register_vars, final_states)


def generate_test_vectors(n: int, num_vectors: int = 100, include_accepted: bool = True, include_rejected: bool = True) -> List[List[float]]:
    """
    Generate test vectors for an automaton that accepts exactly n same symbols.
    
    Args:
        n: Number of states (accepts sequences of length n with same symbols)
        num_vectors: Total number of vectors to generate
        include_accepted: Include vectors that should be accepted
        include_rejected: Include vectors that should be rejected
    
    Returns:
        List of test vectors
    """
    vectors = []
    
    if n == 0:
        # Only empty word is accepted
        if include_accepted:
            vectors.append([])
        if include_rejected:
            for _ in range(num_vectors - 1):
                length = random.randint(1, 10)
                vectors.append([random.randint(-100, 100) for _ in range(length)])
        return vectors
    
    num_each = num_vectors // 2 if (include_accepted and include_rejected) else num_vectors
    
    # Generate accepted vectors: length n with all same symbols
    if include_accepted:
        for _ in range(num_each):
            val = random.randint(-100, 100)
            vectors.append([val] * n)
    
    # Generate rejected vectors
    if include_rejected:
        for _ in range(num_each):
            rejection_type = random.choice(['wrong_length', 'different_symbols'])
            
            if rejection_type == 'wrong_length':
                # Wrong length (but possibly same symbols)
                length = random.choice([l for l in range(1, n + 5) if l != n])
                if length == 0:
                    length = 1
                val = random.randint(-100, 100)
                vectors.append([val] * length)
            else:
                # Correct length but different symbols
                if n >= 2:
                    vec = [random.randint(-100, 100)]
                    for _ in range(n - 1):
                        # Randomly either repeat or change
                        if random.random() < 0.3:
                            vec.append(random.randint(-100, 100))  # Different
                        else:
                            vec.append(vec[0])
                    # Ensure at least one is different
                    if len(set(vec)) == 1:
                        vec[random.randint(1, n - 1)] = vec[0] + random.randint(1, 10)
                    vectors.append(vec)
                else:
                    # For n=1, can only reject by wrong length
                    length = random.choice([0, 2, 3])
                    if length == 0:
                        vectors.append([])
                    else:
                        vectors.append([random.randint(-100, 100) for _ in range(length)])
    
    random.shuffle(vectors)
    return vectors


def generate_test_vectors_for_n_register(n: int, num_vectors: int = 100) -> List[List[float]]:
    """
    Generate test vectors for the n-register automaton (create_n_register_ra).
    
    The automaton accepts sequences of length n where all symbols are equal.
    
    Args:
        n: Number of registers (and sequence length)
        num_vectors: Total number of vectors to generate
    
    Returns:
        List of test vectors (mix of accepted and rejected)
    """
    vectors = []
    
    if n == 0:
        # Only empty word accepted
        vectors.append([])
        for _ in range(num_vectors - 1):
            length = random.randint(1, 5)
            vectors.append([random.randint(-100, 100) for _ in range(length)])
        return vectors
    
    num_each = num_vectors // 2
    
    # Accepted: length n, all same value
    for _ in range(num_each):
        val = random.randint(-100, 100)
        vectors.append([val] * n)
    
    # Rejected: wrong length or different values
    for _ in range(num_each):
        if random.random() < 0.5:
            # Wrong length
            length = random.choice([l for l in range(1, n + 5) if l != n])
            if length == 0:
                length = 1
            val = random.randint(-100, 100)
            vectors.append([val] * length)
        else:
            # Correct length, different values
            if n >= 2:
                vec = [random.randint(-100, 100) for _ in range(n)]
                # Ensure at least one differs
                if len(set(vec)) == 1:
                    vec[1] = vec[0] + random.randint(1, 10)
                vectors.append(vec)
            else:
                vectors.append([random.randint(-100, 100), random.randint(-100, 100)])
    
    random.shuffle(vectors)
    return vectors


def generate_test_vectors_for_distinct(n: int, num_vectors: int = 100) -> List[List[float]]:
    """
    Generate test vectors for the n-register distinct automaton.
    
    The automaton stores first n symbols, then accepts if subsequent 
    symbols are all distinct from the stored values.
    
    Args:
        n: Number of registers
        num_vectors: Total number of vectors
    
    Returns:
        List of test vectors
    """
    vectors = []
    num_each = num_vectors // 2
    
    if n == 0:
        # Accepts anything
        for _ in range(num_vectors):
            length = random.randint(1, 10)
            vectors.append([random.randint(-100, 100) for _ in range(length)])
        return vectors
    
    # Accepted: first n symbols stored, rest all different from those
    for _ in range(num_each):
        stored = [random.randint(-100, 100) for _ in range(n)]
        # Add some symbols that are all different from stored
        extra_len = random.randint(0, 5)
        extra = []
        for _ in range(extra_len):
            # Generate value different from all stored
            val = max(stored) + random.randint(1, 100)
            extra.append(val)
        vectors.append(stored + extra)
    
    # Rejected: first n symbols stored, then at least one matches a stored value
    for _ in range(num_each):
        stored = [random.randint(-100, 100) for _ in range(n)]
        extra_len = random.randint(1, 5)
        extra = []
        for i in range(extra_len):
            if i == 0:
                # First extra symbol matches one of the stored
                extra.append(random.choice(stored))
            else:
                extra.append(random.randint(-100, 100))
        vectors.append(stored + extra)
    
    random.shuffle(vectors)
    return vectors


def run_single_robustness_test(
    n_states: int,
    num_vectors: int = 100,
    delta: float = 0.1,
    metric_name: str = "linf",
    timeout: float = 60,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run robustness check stress test for an automaton with n_states.
    
    Args:
        n_states: Number in the 'exactly n same symbols' automaton (actual states = n+2)
        num_vectors: Number of test vectors
        delta: Robustness radius
        metric_name: Metric for robustness ("linf", "l2", etc.)
        timeout: Timeout in seconds
        verbose: Print progress
    
    Returns:
        Dictionary with results
    """
    result = {
        'n_value': n_states,
        'actual_states': n_states + 2,  # n+2 states in the automaton
        'num_vectors': num_vectors,
        'delta': delta,
        'metric': metric_name,
        'total_time': None,
        'avg_time_per_vector': None,
        'robust_count': 0,
        'non_robust_count': 0,
        'not_accepted_count': 0,
        'timeout_occurred': False,
        'status': 'pending',
        'error': None,
    }
    
    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing n={n_states} (actual states={n_states + 2})")
            print(f"{'='*60}")
        
        # Create automaton
        start_time = time.time()
        ra = create_strictly_increasing_ra(n_states)
        creation_time = time.time() - start_time
        
        if verbose:
            print(f"Automaton created in {creation_time:.4f}s")
            print(f"  States: {ra.states}")
            print(f"  Transitions: {len(ra.transitions)}")
            print(f"  Final states: {ra.final_states}")
        
        # Generate test vectors
        vectors = generate_test_vectors_increasing(n_states, num_vectors)
        
        print(f"Generated {len(vectors)} test vectors")
        
        # Run the exact robustness checker on every accepted vector
        robustness_start = time.time()
        checker = RobustnessChecker(ra, get_metric_raa(metric_name))
        
        robust_count = 0
        non_robust_count = 0
        not_accepted_count = 0
        
        for i, vec in enumerate(vectors):
            if time.time() - robustness_start > timeout:
                result['timeout_occurred'] = True
                if verbose:
                    print(f"Timeout reached after {i} vectors")
                break
            
            # Check if automaton accepts this vector
            if not checker.accepts(vec):
                not_accepted_count += 1
                continue
            
            remaining = timeout - (time.time() - robustness_start)
            try:
                res = checker.find_witness(vec, delta, accepted=True,
                                           timeout=remaining)
            except TimeoutError:
                result['timeout_occurred'] = True
                if verbose:
                    print(f"Timeout reached at vector {i}")
                break
            if res.robust:
                robust_count += 1
            else:
                non_robust_count += 1

        total_time = time.time() - robustness_start
        
        result['total_time'] = total_time
        result['avg_time_per_vector'] = total_time / len(vectors) if vectors else 0
        result['robust_count'] = robust_count
        result['non_robust_count'] = non_robust_count
        result['not_accepted_count'] = not_accepted_count
        result['status'] = 'success'
        
        if verbose:
            print(f"\nResults:")
            print(f"  Total time: {total_time:.4f}s")
            print(f"  Avg per vector: {result['avg_time_per_vector']*1000:.2f}ms")
            print(f"  Robust: {robust_count}, Non-robust: {non_robust_count}, Not accepted: {not_accepted_count}")
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        traceback.print_exc()
    
    return result


def run_stress_test(
    state_values: List[int],
    num_vectors: int = 100,
    delta: float = 0.1,
    metric_name: str = "linf",
    timeout: float = 60,
    output_csv: str = "results/stress_test_robustness_states.csv",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run stress tests across multiple state configurations.
    
    Args:
        state_values: List of n values for create_exactly_n_same_word_ra
        num_vectors: Number of test vectors per automaton
        delta: Robustness radius
        metric_name: Metric for robustness
        timeout: Timeout per test
        output_csv: Output CSV path
        verbose: Print progress
    
    Returns:
        List of result dictionaries
    """
    all_results = []
    
    # Prepare CSV
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    
    header = [
        'timestamp', 'n_value', 'actual_states', 'num_vectors', 'delta', 'metric',
        'total_time', 'avg_time_per_vector', 'robust_count', 'non_robust_count',
        'not_accepted_count', 'timeout_occurred', 'status', 'error'
    ]
    
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
    
    total_tests = len(state_values)
    
    for idx, n in enumerate(state_values, 1):
        if verbose:
            print(f"\n[{idx}/{total_tests}] Testing n={n}...")
        print('metric_name', metric_name)
        result = run_single_robustness_test(
            n_states=n,
            num_vectors=num_vectors,
            delta=delta,
            metric_name=metric_name,
            timeout=timeout,
            verbose=verbose,
        )
        
        result['timestamp'] = datetime.now().isoformat()
        all_results.append(result)
        
        # Write to CSV
        with open(output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                result.get('timestamp', ''),
                result.get('n_value', ''),
                result.get('actual_states', ''),
                result.get('num_vectors', ''),
                result.get('delta', ''),
                result.get('metric', ''),
                f"{result.get('total_time', 0):.4f}" if result.get('total_time') else '',
                f"{result.get('avg_time_per_vector', 0)*1000:.4f}" if result.get('avg_time_per_vector') else '',
                result.get('robust_count', ''),
                result.get('non_robust_count', ''),
                result.get('not_accepted_count', ''),
                result.get('timeout_occurred', ''),
                result.get('status', ''),
                result.get('error', ''),
            ])
    
    return all_results


def print_summary(results: List[Dict[str, Any]]):
    """Print a summary of stress test results."""
    print("\n" + "="*80)
    print("STRESS TEST SUMMARY: Robustness Check Efficiency vs States")
    print("="*80)
    
    if not results:
        print("No results to summarize.")
        return
    
    print(f"\n{'N':<6} {'States':<8} {'Total Time':<12} {'Avg/Vector':<12} {'Robust':<8} {'Non-Rob':<8} {'Status':<10}")
    print(f"{'-'*6} {'-'*8} {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*10}")
    
    for r in sorted(results, key=lambda x: x.get('n_value', 0)):
        n = r.get('n_value', '?')
        states = r.get('actual_states', '?')
        time_str = f"{r['total_time']:.4f}s" if r.get('total_time') else 'N/A'
        avg_str = f"{r['avg_time_per_vector']*1000:.2f}ms" if r.get('avg_time_per_vector') else 'N/A'
        robust = r.get('robust_count', '?')
        non_robust = r.get('non_robust_count', '?')
        status = r.get('status', '?')
        print(f"{n:<6} {states:<8} {time_str:<12} {avg_str:<12} {robust:<8} {non_robust:<8} {status:<10}")
    
    # Time scaling analysis
    successful = [r for r in results if r.get('status') == 'success' and r.get('total_time')]
    if len(successful) >= 2:
        print(f"\n{'='*40}")
        print("Scaling Analysis:")
        times = [(r['n_value'], r['total_time']) for r in successful]
        times.sort()
        
        if len(times) >= 2:
            first_n, first_t = times[0]
            last_n, last_t = times[-1]
            if first_t > 0 and last_t > 0:
                ratio = last_t / first_t
                n_ratio = last_n / first_n if first_n > 0 else 1
                print(f"  Time ratio (n={last_n} vs n={first_n}): {ratio:.2f}x")
                print(f"  N ratio: {n_ratio:.2f}x")
                if n_ratio > 1:
                    approx_complexity = math.log(ratio) / math.log(n_ratio)
                    print(f"  Approximate complexity exponent: O(n^{approx_complexity:.2f})")


def run_single_register_test(
    n_registers: int,
    num_vectors: int = 100,
    delta: float = 0.1,
    metric_name: str = "l1",
    timeout: float = 60,
    verbose: bool = True,
    automaton_type: str = "same",  # "same" or "distinct"
) -> Dict[str, Any]:
    """
    Run robustness check stress test for an automaton with n registers.
    
    Args:
        n_registers: Number of registers in the automaton
        num_vectors: Number of test vectors
        delta: Robustness radius
        metric_name: Metric for robustness
        timeout: Timeout in seconds
        verbose: Print progress
        automaton_type: "same" for create_n_register_ra, "distinct" for create_n_register_distinct_ra
    
    Returns:
        Dictionary with results
    """
    result = {
        'n_registers': n_registers,
        'actual_states': n_registers + 2,
        'automaton_type': automaton_type,
        'num_vectors': num_vectors,
        'delta': delta,
        'metric': metric_name,
        'total_time': None,
        'avg_time_per_vector': None,
        'robust_count': 0,
        'non_robust_count': 0,
        'not_accepted_count': 0,
        'timeout_occurred': False,
        'status': 'pending',
        'error': None,
    }
    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing n_registers={n_registers} (type={automaton_type})")
            print(f"{'='*60}")
        
        # Create automaton
        start_time = time.time()
        if automaton_type == "distinct":
            ra = create_n_register_distinct_ra(n_registers)
            vectors = generate_test_vectors_for_distinct(n_registers, num_vectors)
        else:
            ra = create_n_register_ra(n_registers)
            vectors = generate_test_vectors_for_n_register(n_registers, num_vectors)
        creation_time = time.time() - start_time
        
        if verbose:
            print(f"Automaton created in {creation_time:.4f}s")
            print(f"  States: {ra.states}")
            print(f"  Registers: {len(ra.register_vars)}")
            print(f"  Transitions: {len(ra.transitions)}")
            print(f"  Final states: {ra.final_states}")
            print(f"Generated {len(vectors)} test vectors")
        
        # Run the exact robustness checker on every accepted vector
        robustness_start = time.time()
        checker = RobustnessChecker(ra, get_metric_raa(metric_name))
        
        robust_count = 0
        non_robust_count = 0
        not_accepted_count = 0
        
        for i, vec in enumerate(vectors):
            if time.time() - robustness_start > timeout:
                result['timeout_occurred'] = True
                if verbose:
                    print(f"Timeout reached after {i} vectors")
                break

            # Check if automaton accepts this vector
            if not checker.accepts(vec):
                not_accepted_count += 1
                continue
            
            remaining = timeout - (time.time() - robustness_start)
            try:
                res = checker.find_witness(vec, delta, accepted=True,
                                           timeout=remaining)
            except TimeoutError:
                result['timeout_occurred'] = True
                if verbose:
                    print(f"Timeout reached at vector {i}")
                break
            if res.robust:
                robust_count += 1
            else:
                non_robust_count += 1

        total_time = time.time() - robustness_start
        
        result['total_time'] = total_time
        result['avg_time_per_vector'] = total_time / len(vectors) if vectors else 0
        result['robust_count'] = robust_count
        result['non_robust_count'] = non_robust_count
        result['not_accepted_count'] = not_accepted_count
        result['status'] = 'success'
        
        if verbose:
            print(f"\nResults:")
            print(f"  Total time: {total_time:.4f}s")
            print(f"  Avg per vector: {result['avg_time_per_vector']*1000:.2f}ms")
            print(f"  Robust: {robust_count}, Non-robust: {non_robust_count}, Not accepted: {not_accepted_count}")
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        traceback.print_exc()
    
    return result


def run_register_stress_test(
    register_values: List[int],
    num_vectors: int = 100,
    delta: float = 0.1,
    metric_name: str = "l1",
    timeout: float = 60,
    output_csv: str = "results/stress_test_robustness_registers.csv",
    verbose: bool = True,
    automaton_type: str = "same",
) -> List[Dict[str, Any]]:
    """
    Run stress tests across multiple register configurations.
    
    Args:
        register_values: List of n values for register count
        num_vectors: Number of test vectors per automaton
        delta: Robustness radius
        metric_name: Metric for robustness
        timeout: Timeout per test
        output_csv: Output CSV path
        verbose: Print progress
        automaton_type: "same" or "distinct"
    
    Returns:
        List of result dictionaries
    """
    all_results = []
    
    # Prepare CSV
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    
    header = [
        'timestamp', 'n_registers', 'actual_states', 'automaton_type', 'num_vectors', 
        'delta', 'metric', 'total_time', 'avg_time_per_vector', 'robust_count', 
        'non_robust_count', 'not_accepted_count', 'timeout_occurred', 'status', 'error'
    ]
    
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
    
    total_tests = len(register_values)
    
    for idx, n in enumerate(register_values, 1):
        if verbose:
            print(f"\n[{idx}/{total_tests}] Testing n_registers={n}...")
        
        result = run_single_register_test(
            n_registers=n,
            num_vectors=num_vectors,
            delta=delta,
            metric_name=metric_name,
            timeout=timeout,
            verbose=verbose,
            automaton_type=automaton_type,
        )
        
        result['timestamp'] = datetime.now().isoformat()
        all_results.append(result)
        
        # Write to CSV
        with open(output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                result.get('timestamp', ''),
                result.get('n_registers', ''),
                result.get('actual_states', ''),
                result.get('automaton_type', ''),
                result.get('num_vectors', ''),
                result.get('delta', ''),
                result.get('metric', ''),
                f"{result.get('total_time', 0):.4f}" if result.get('total_time') else '',
                f"{result.get('avg_time_per_vector', 0)*1000:.4f}" if result.get('avg_time_per_vector') else '',
                result.get('robust_count', ''),
                result.get('non_robust_count', ''),
                result.get('not_accepted_count', ''),
                result.get('timeout_occurred', ''),
                result.get('status', ''),
                result.get('error', ''),
            ])
    
    return all_results


def print_register_summary(results: List[Dict[str, Any]]):
    """Print a summary of register stress test results."""
    print("\n" + "="*80)
    print("STRESS TEST SUMMARY: Robustness Check Efficiency vs Registers")
    print("="*80)
    
    if not results:
        print("No results to summarize.")
        return
    
    print(f"\n{'Regs':<6} {'States':<8} {'Total Time':<12} {'Avg/Vector':<12} {'Robust':<8} {'Non-Rob':<8} {'Status':<10}")
    print(f"{'-'*6} {'-'*8} {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*10}")
    
    for r in sorted(results, key=lambda x: x.get('n_registers', 0)):
        n = r.get('n_registers', '?')
        states = r.get('actual_states', '?')
        time_str = f"{r['total_time']:.4f}s" if r.get('total_time') else 'N/A'
        avg_str = f"{r['avg_time_per_vector']*1000:.2f}ms" if r.get('avg_time_per_vector') else 'N/A'
        robust = r.get('robust_count', '?')
        non_robust = r.get('non_robust_count', '?')
        status = r.get('status', '?')
        print(f"{n:<6} {states:<8} {time_str:<12} {avg_str:<12} {robust:<8} {non_robust:<8} {status:<10}")
    
    # Time scaling analysis
    successful = [r for r in results if r.get('status') == 'success' and r.get('total_time')]
    if len(successful) >= 2:
        print(f"\n{'='*40}")
        print("Scaling Analysis (Registers):")
        times = [(r['n_registers'], r['total_time']) for r in successful]
        times.sort()
        
        if len(times) >= 2:
            first_n, first_t = times[0]
            last_n, last_t = times[-1]
            if first_t > 0 and last_t > 0 and first_n > 0:
                ratio = last_t / first_t
                n_ratio = last_n / first_n
                print(f"  Time ratio (n={last_n} vs n={first_n}): {ratio:.2f}x")
                print(f"  Registers ratio: {n_ratio:.2f}x")
                if n_ratio > 1:
                    approx_complexity = math.log(ratio) / math.log(n_ratio)
                    print(f"  Approximate complexity exponent: O(n^{approx_complexity:.2f})")


def main():
    parser = argparse.ArgumentParser(
        description='Stress test robustness check efficiency based on number of states or registers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test states 1 to 10
  python stress_test_robustness_states.py --min_states 1 --max_states 10
  
  # Test specific state values
  python stress_test_robustness_states.py --states 1 2 3 5 8 10
  
  # Test registers instead of states
  python stress_test_robustness_states.py --mode registers --registers 1 2 3 4 5
  
  # Quick test
  python stress_test_robustness_states.py --quick
  
  # With custom parameters
  python stress_test_robustness_states.py --min_states 1 --max_states 15 --delta 0.05 --num_vectors 200
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='states', choices=['states', 'registers'],
                        help='Test by states or registers (default: states)')
    
    # State configuration
    state_group = parser.add_mutually_exclusive_group()
    state_group.add_argument('--states', type=int, nargs='+', 
                             help='Specific state values to test')
    state_group.add_argument('--min_states', type=int, default=1,
                             help='Minimum n value (default: 1)')
    
    parser.add_argument('--max_states', type=int, default=10,
                        help='Maximum n value when using --min_states (default: 10)')
    parser.add_argument('--step', type=int, default=1,
                        help='Step between state values (default: 1)')
    
    # Register configuration
    parser.add_argument('--registers', type=int, nargs='+',
                        help='Specific register values to test (for --mode registers)')
    parser.add_argument('--min_registers', type=int, default=1,
                        help='Minimum register count (default: 1)')
    parser.add_argument('--max_registers', type=int, default=10,
                        help='Maximum register count (default: 10)')
    parser.add_argument('--automaton_type', type=str, default='same', choices=['same', 'distinct'],
                        help='Automaton type for register tests: same (all equal) or distinct (default: same)')
    
    # Test parameters
    parser.add_argument('--num_vectors', type=int, default=100,
                        help='Number of test vectors per automaton (default: 100)')
    parser.add_argument('--delta', type=float, default=0.1,
                        help='Robustness radius delta (default: 0.1)')
    parser.add_argument('--metric', type=str, default='last_letter',
                        choices=['l1', 'hamming', 'edit', 'last_letter', 'dtw'],
                        help='Distance metric (default: last_letter)')
    parser.add_argument('--timeout', type=float, default=60,
                        help='Timeout per test in seconds (default: 60)')
    
    # Output configuration
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path (auto-generated if not specified)')
    parser.add_argument('--quiet', action='store_true', 
                        help='Suppress verbose output')
    
    # Presets
    parser.add_argument('--quick', action='store_true',
                        help='Quick test (1-5 values, fewer vectors)')
    parser.add_argument('--exhaustive', action='store_true',
                        help='Exhaustive test (1-20 values, more vectors)')
    
    args = parser.parse_args()
    
    if args.mode == 'registers':
        # Register-based testing
        if args.registers:
            register_values = args.registers
        else:
            register_values = list(range(args.min_registers, args.max_registers + 1, args.step))
        
        # Apply presets
        if args.quick:
            register_values = [1, 2, 3, 4, 5]
            args.num_vectors = 50
            args.timeout = 30
        elif args.exhaustive:
            register_values = list(range(1, 21))
            args.num_vectors = 100
            args.timeout = 120
        
        output_csv = args.output or 'results/stress_test_robustness_registers.csv'
        
        print("="*80)
        print("STRESS TEST: Robustness Check Efficiency vs Number of Registers")
        print("="*80)
        print(f"Register values: {register_values}")
        print(f"Automaton type: {args.automaton_type}")
        print(f"Vectors per test: {args.num_vectors}")
        print(f"Delta: {args.delta}")
        print(f"Metric: {args.metric}")
        print(f"Timeout: {args.timeout}s")
        print(f"Output: {output_csv}")
        print("="*80)
        
        # Run register stress tests
        results = run_register_stress_test(
            register_values=register_values,
            num_vectors=args.num_vectors,
            delta=args.delta,
            metric_name=args.metric,
            timeout=args.timeout,
            output_csv=output_csv,
            verbose=not args.quiet,
            automaton_type=args.automaton_type,
        )
        
        # Print summary
        print_register_summary(results)
        
    else:
        # State-based testing (original behavior)
        if args.states:
            state_values = args.states
        else:
            state_values = list(range(args.min_states, args.max_states + 1, args.step))
        
        # Apply presets
        if args.quick:
            state_values = [1, 2, 3, 4, 5]
            args.num_vectors = 50
            args.timeout = 30
        elif args.exhaustive:
            state_values = list(range(0, 21, 2))
            args.num_vectors = 50
            args.timeout = 120
        
        output_csv = args.output or 'results/stress_test_robustness_states.csv'
        
        print("="*80)
        print("STRESS TEST: Robustness Check Efficiency vs Number of States")
        print("="*80)
        print(f"State values (n): {state_values}")
        print(f"Actual states: {[n+2 for n in state_values]}")
        print(f"Vectors per test: {args.num_vectors}")
        print(f"Delta: {args.delta}")
        print(f"Metric: {args.metric}")
        print(f"Timeout: {args.timeout}s")
        print(f"Output: {output_csv}")
        print("="*80)
        
        # Run stress tests
        results = run_stress_test(
            state_values=state_values,
            num_vectors=args.num_vectors,
            delta=args.delta,
            metric_name=args.metric,
            timeout=args.timeout,
            output_csv=output_csv,
            verbose=not args.quiet,
        )
        
        # Print summary
        print_summary(results)
    
    print(f"\nResults saved to: {output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
