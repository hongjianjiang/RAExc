#!/usr/bin/env python3
"""
Stress Test: Learning Performance over Sequence Length

This script evaluates how different learners perform as 
sequence lengths increase. It measures:
- Learning time
- Learning accuracy/rate
- Memory usage (optional)
- Scalability characteristics

Usage:
    python stress_test_sequence_length.py --language L1 --learner ls
    python stress_test_sequence_length.py --language S1 --learner smt --lengths 5 10 20 50
    python stress_test_sequence_length.py --all_languages --learner ls
"""

import argparse
import csv
import os
import sys
import time
import traceback
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import copy
import math
import random

sys.path.append(os.path.dirname(__file__))

from benchmark_gen import new_generate_language, random_sample_generation, get_available_languages
from local_serach_synthesis import dra_learner
from register_automata.ra import RegisterAutomata, load_double_ls_dra, load_integer_ls_dra, load_smt_dra, Transition
import smt_based_synthesis.smtlearner as smtlearner
import smt_based_synthesis.eqsmtlearner as eqsmtlearner

import z3

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


def get_theory_from_language(language: str) -> str:
    """Determine theory type based on language prefix."""
    return 'Double' if language.startswith('S') else 'Integer'


def sample_with_max_len(
    language: str, 
    num_pos: int, 
    num_neg: int, 
    max_len: int
) -> Tuple[List, List]:
    """
    Generate positive and negative samples up to a given max sequence length.
    
    Returns:
        Tuple of (positive_samples, negative_samples)
    """
    pos, neg = [], []
    attempts = 0
    max_attempts = num_pos * 10 + num_neg * 10  # Prevent infinite loops
    
    # Use the language's generate_samples method with the specified max_len
    try:
        language_classes = {}
        from benchmark_gen import Language
        for cls in Language.__subclasses__():
            language_classes[cls.__name__] = cls
        
        if language not in language_classes:
            raise ValueError(f"Unknown language: {language}")
        
        lang_class = language_classes[language]
        lang_instance = lang_class(num_pos, num_neg, max_len)
        pos, neg = lang_instance.generate_samples(num_pos, num_neg, max_len)
        
        # Filter to ensure max length constraint
        pos = [p for p in pos if len(p) <= max_len and len(p) > 0]
        neg = [n for n in neg if len(n) <= max_len and len(n) > 0]
        
    except Exception as e:
        print(f"Warning: Error generating samples for {language}: {e}")
        pos, neg = [], []
    
    return pos, neg


def run_ls_stress_test(
    language: str,
    max_len: int,
    states: int,
    regs: int,
    edge_type: bool,
    theory: str,
    algo: str = 'sa',
    timeout: float = 120,
    gamma: float = 0.05,
    epsilon: float = 0.05,
    num_samples: int = 500,
) -> Dict[str, Any]:
    """
    Run local search learner stress test for a given sequence length.
    
    Returns:
        Dictionary with results including time, accuracy, etc.
    """
    result = {
        'language': language,
        'max_len': max_len,
        'learner': 'ls',
        'algo': algo,
        'states': states,
        'registers': regs,
        'learning_time': None,
        'learning_rate': None,
        'status': 'pending',
        'error': None,
        'num_pos_samples': 0,
        'num_neg_samples': 0,
        'avg_sample_len': 0,
    }
    
    try:
        start_time = time.time()
        
        # Generate samples with the specified max length
        pos, neg = sample_with_max_len(language, num_samples, num_samples, max_len)
        
        if not pos or not neg:
            result['status'] = 'no_samples'
            result['error'] = 'Failed to generate samples'
            return result
        
        result['num_pos_samples'] = len(pos)
        result['num_neg_samples'] = len(neg)
        all_samples = pos + neg
        result['avg_sample_len'] = sum(len(s) for s in all_samples) / len(all_samples) if all_samples else 0
        
        learner = dra_learner.DRAlearner(
            num_states=states,
            num_registers=regs,
            constants=[],
            pos_samples=pos,
            neg_samples=neg,
            golden_pos_samples=[],
            auto=None,
            debug=False,
            exactly_one=edge_type,
            theory=theory
        )
        
        # Run the selected algorithm
        num_iterations = 100000
        if algo == 'sa':
            learner.simulated_annealing(
                initial_temperature=1.0,
                cooling_rate=0.995,
                max_iterations=num_iterations,
                timeout=timeout,
            )
        elif algo == 'hc':
            learner.hill_climbing(timeout=timeout, restart_iterations=num_iterations)
        elif algo == 'ant':
            learner.ant_colony_optimization(
                n_ants=10, n_iterations=1000, alpha=1.0, beta=2.0,
                evaporation_rate=0.5, timeout=timeout
            )
        elif algo == 'lns':
            learner.large_neighborhood_search(n_iterations=10000, destruction_size=3, timeout=timeout)
        elif algo == 'pso':
            learner.particle_swarm_optimization(swarm_size=10, n_iterations=1000, timeout=timeout)
        elif algo == 'genetic':
            learner.genetic_algorithm(population_size=10, generations=1000, mutation_rate=0.3, timeout=timeout)
        elif algo == 'tabu':
            learner.tabu_search(n_iterations=1000, tabu_size=50, timeout=timeout)
        
        elapsed_time = time.time() - start_time
        
        result['learning_time'] = elapsed_time
        result['learning_rate'] = learner.best_score
        result['status'] = 'success' if learner.best_score >= (1 - epsilon) else 'partial'
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        result['learning_time'] = time.time() - start_time if 'start_time' in dir() else None
        traceback.print_exc()
    
    return result


def run_smt_stress_test(
    language: str,
    max_len: int,
    states: int,
    regs: List[str],
    ic: List,
    uc: List,
    edge_limit_type: str,
    edge_limit_num: int,
    theory: str,
    gamma: float = 0.05,
    epsilon: float = 0.05,
    num_samples: int = 500,
) -> Dict[str, Any]:
    """
    Run SMT learner stress test for a given sequence length.
    
    Returns:
        Dictionary with results including time, accuracy, etc.
    """
    result = {
        'language': language,
        'max_len': max_len,
        'learner': 'smt',
        'algo': 'smt',
        'states': states,
        'registers': len(regs),
        'learning_time': None,
        'learning_rate': None,
        'status': 'pending',
        'error': None,
        'num_pos_samples': 0,
        'num_neg_samples': 0,
        'avg_sample_len': 0,
    }
    
    try:
        start_time = time.time()
        
        # Generate samples with the specified max length
        pos, neg = sample_with_max_len(language, num_samples, num_samples, max_len)
        
        if not pos or not neg:
            result['status'] = 'no_samples'
            result['error'] = 'Failed to generate samples'
            return result
        
        result['num_pos_samples'] = len(pos)
        result['num_neg_samples'] = len(neg)
        all_samples = pos + neg
        result['avg_sample_len'] = sum(len(s) for s in all_samples) / len(all_samples) if all_samples else 0
        
        # Select appropriate SMT learner based on theory
        if theory == "Integer":
            learner = eqsmtlearner.Register_Learner(
                n=states,
                regs=regs,
                ic=ic,
                uc=uc,
                pos=pos,
                neg=neg,
                edge_limit_type=edge_limit_type,
                edge_limit_num=edge_limit_num,
                debug=False,
                batch_size=8,
                target=1 - epsilon,
            )
        else:
            learner = smtlearner.Register_Learner(
                n=states,
                regs=regs,
                ic=ic,
                uc=uc,
                pos=pos,
                neg=neg,
                edge_limit_type=edge_limit_type,
                edge_limit_num=edge_limit_num,
                debug=False,
                batch_size=5,
                strategy="edge",
                target=1 - epsilon,
            )
        
        auto, epos = learner.get_model()
        elapsed_time = time.time() - start_time
        
        result['learning_time'] = elapsed_time
        
        if auto == "unsat" or auto == 'timeout':
            result['status'] = 'unsat' if auto == 'unsat' else 'timeout'
            result['learning_rate'] = 0
        else:
            # Save and reload to compute learning rate
            os.makedirs("stress_test_output", exist_ok=True)
            json_file = f"stress_test_output/smt_{language}_len{max_len}.json"
            auto.save_to_json(json_file)
            loaded_auto = load_smt_dra(json_file)
            learning_rate = learner.consistence_check(loaded_auto, pos, neg)
            result['learning_rate'] = learning_rate
            result['status'] = 'success' if learning_rate >= (1 - epsilon) else 'partial'
            
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        result['learning_time'] = time.time() - start_time if 'start_time' in dir() else None
        traceback.print_exc()
    
    return result


def run_stress_test(
    languages: List[str],
    learner_type: str,
    sequence_lengths: List[int],
    output_csv: str,
    algo: str = 'sa',
    timeout: float = 120,
    gamma: float = 0.05,
    epsilon: float = 0.05,
    num_samples: int = 500,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run comprehensive stress tests across multiple languages and sequence lengths.
    
    Args:
        languages: List of language names to test
        learner_type: Type of learner ('ls', 'smt', 'al')
        sequence_lengths: List of max sequence lengths to test
        output_csv: Path to output CSV file
        algo: Algorithm for local search ('sa', 'hc', 'ant', etc.)
        timeout: Timeout per learning attempt
        gamma: PAC parameter
        epsilon: PAC parameter
        num_samples: Number of positive/negative samples to generate
        verbose: Print progress information
    
    Returns:
        List of result dictionaries
    """
    all_results = []
    
    # Prepare CSV
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    
    header = [
        'timestamp', 'language', 'learner', 'algo', 'max_len', 
        'states', 'registers', 'num_pos_samples', 'num_neg_samples',
        'avg_sample_len', 'learning_time', 'learning_rate', 'status', 'error'
    ]
    
    # Write header if file doesn't exist
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
    
    total_tests = len(languages) * len(sequence_lengths)
    test_count = 0
    
    for language in languages:
        try:
            # Get language configuration
            lang = new_generate_language(language, 0, 0, 0)
            print(f"\nProcessing language: {language} (States: {lang.states}, Regs: {len(lang.regs)})")
            theory = get_theory_from_language(language)
            
            for max_len in sequence_lengths:
                test_count += 1
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"[{test_count}/{total_tests}] Testing {language} with max_len={max_len}")
                    print(f"Learner: {learner_type}, Theory: {theory}")
                    print(f"{'='*60}")
                
                if learner_type == 'ls':
                    print('algo:', algo)
                    result = run_ls_stress_test(
                        language=language,
                        max_len=max_len,
                        states=lang.states,
                        regs=len(lang.regs),
                        edge_type=True if lang.edge_limit_type == 'exactly' else False,
                        theory=theory,
                        algo=algo,
                        timeout=timeout,
                        gamma=gamma,
                        epsilon=epsilon,
                        num_samples=num_samples,
                    )
                elif learner_type == 'smt':
                    result = run_smt_stress_test(
                        language=language,
                        max_len=max_len,
                        states=lang.states,
                        regs=lang.regs,
                        ic=lang.ic,
                        uc=lang.uc,
                        edge_limit_type=lang.edge_limit_type,
                        edge_limit_num=lang.edge_limit_num,
                        theory=theory,
                        gamma=gamma,
                        epsilon=epsilon,
                        num_samples=num_samples,
                    )
                else:
                    print(f"Unsupported learner type: {learner_type}")
                    continue
                
                result['timestamp'] = datetime.now().isoformat()
                all_results.append(result)
                
                # Write result to CSV
                with open(output_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        result.get('timestamp', ''),
                        result.get('language', ''),
                        result.get('learner', ''),
                        result.get('algo', ''),
                        result.get('max_len', ''),
                        result.get('states', ''),
                        result.get('registers', ''),
                        result.get('num_pos_samples', ''),
                        result.get('num_neg_samples', ''),
                        f"{result.get('avg_sample_len', 0):.2f}",
                        f"{result.get('learning_time', 0):.2f}" if result.get('learning_time') else '',
                        f"{result.get('learning_rate', 0):.4f}" if result.get('learning_rate') else '',
                        result.get('status', ''),
                        result.get('error', ''),
                    ])
                
                if verbose:
                    print(f"\nResult: {result['status']}")
                    if result['learning_time']:
                        print(f"  Time: {result['learning_time']:.2f}s")
                    if result['learning_rate'] is not None:
                        print(f"  Learning Rate: {result['learning_rate']:.4f}")
                    print(f"  Samples: {result['num_pos_samples']} pos, {result['num_neg_samples']} neg")
                    print(f"  Avg Sample Length: {result['avg_sample_len']:.2f}")
                    
        except Exception as e:
            print(f"Error processing language {language}: {e}")
            traceback.print_exc()
    
    return all_results


def print_summary(results: List[Dict[str, Any]]):
    """Print a summary of stress test results."""
    print("\n" + "="*80)
    print("STRESS TEST SUMMARY")
    print("="*80)
    
    if not results:
        print("No results to summarize.")
        return
    
    # Group by language
    by_language = {}
    for r in results:
        lang = r.get('language', 'unknown')
        if lang not in by_language:
            by_language[lang] = []
        by_language[lang].append(r)
    
    for lang, lang_results in sorted(by_language.items()):
        print(f"\n{lang}:")
        print(f"  {'Max Len':<10} {'Time (s)':<12} {'Rate':<10} {'Status':<10}")
        print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
        
        for r in sorted(lang_results, key=lambda x: x.get('max_len', 0)):
            max_len = r.get('max_len', '?')
            time_str = f"{r['learning_time']:.2f}" if r.get('learning_time') else 'N/A'
            rate_str = f"{r['learning_rate']:.4f}" if r.get('learning_rate') is not None else 'N/A'
            status = r.get('status', '?')
            print(f"  {max_len:<10} {time_str:<12} {rate_str:<10} {status:<10}")
    
    # Overall statistics
    successful = [r for r in results if r.get('status') == 'success']
    partial = [r for r in results if r.get('status') == 'partial']
    failed = [r for r in results if r.get('status') in ('error', 'timeout', 'unsat', 'no_samples')]
    
    print(f"\n{'='*40}")
    print(f"Total Tests: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Partial: {len(partial)}")
    print(f"  Failed: {len(failed)}")
    
    if successful:
        avg_time = sum(r['learning_time'] for r in successful if r.get('learning_time')) / len(successful)
        avg_rate = sum(r['learning_rate'] for r in successful if r.get('learning_rate') is not None) / len(successful)
        print(f"\nSuccessful Tests Statistics:")
        print(f"  Avg Learning Time: {avg_time:.2f}s")
        print(f"  Avg Learning Rate: {avg_rate:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Stress test learning performance over sequence lengths',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single language with default settings
  python stress_test_sequence_length.py --language L1 --learner ls
  
  # Test with specific sequence lengths
  python stress_test_sequence_length.py --language S1 --learner smt --lengths 5 10 20 50 100
  
  # Test multiple languages
  python stress_test_sequence_length.py --languages L1 L2 L3 --learner ls --algo sa
  
  # Test all available languages
  python stress_test_sequence_length.py --all_languages --learner ls
  
  # Quick sanity test
  python stress_test_sequence_length.py --language L1 --learner ls --lengths 5 10 --quick
        """
    )
    
    # Language selection
    lang_group = parser.add_mutually_exclusive_group(required=True)
    lang_group.add_argument('--language', type=str, help='Single language to test')
    lang_group.add_argument('--languages', type=str, nargs='+', help='Multiple languages to test')
    lang_group.add_argument('--all_languages', action='store_true', help='Test all available languages')
    
    # Learner configuration
    parser.add_argument('--learner', type=str, default='ls', choices=['ls', 'smt'],
                        help='Learner type (default: ls)')
    parser.add_argument('--algo', type=str, default='sa',
                        choices=['sa', 'hc', 'ant', 'lns', 'pso', 'genetic', 'tabu'],
                        help='Local search algorithm (default: sa)')
    
    # Sequence length configuration
    parser.add_argument('--lengths', type=int, nargs='+', default=[5, 10, 20, 30, 50, 75, 100],
                        help='Sequence lengths to test (default: 5 10 20 30 50 75 100)')
    parser.add_argument('--min_len', type=int, default=None, help='Minimum sequence length')
    parser.add_argument('--max_len', type=int, default=None, help='Maximum sequence length')
    parser.add_argument('--len_step', type=int, default=None, help='Step between sequence lengths')
    
    # Test parameters
    parser.add_argument('--timeout', type=float, default=120, help='Timeout per test in seconds (default: 120)')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples per class (default: 500)')
    parser.add_argument('--gamma', type=float, default=0.05, help='PAC gamma parameter (default: 0.05)')
    parser.add_argument('--epsilon', type=float, default=0.05, help='PAC epsilon parameter (default: 0.05)')
    
    # Output configuration
    parser.add_argument('--output', type=str, default='results/stress_test_sequence_length.csv',
                        help='Output CSV file path')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    # Preset configurations
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer lengths and samples')
    parser.add_argument('--exhaustive', action='store_true', help='Exhaustive test with many lengths')
    
    args = parser.parse_args()
    
    # Determine languages to test
    if args.language:
        languages = [args.language]
    elif args.languages:
        languages = args.languages
    elif args.all_languages:
        try:
            languages = get_available_languages()
            # Filter to only include L* and S* languages that have models
            languages = [l for l in languages if l.startswith('L') or l.startswith('S')]
        except Exception as e:
            print(f"Error getting available languages: {e}")
            return 1
    else:
        print("Error: No language specified")
        return 1
    
    # Determine sequence lengths
    if args.min_len and args.max_len and args.len_step:
        sequence_lengths = list(range(args.min_len, args.max_len + 1, args.len_step))
    else:
        sequence_lengths = args.lengths
    
    # Apply presets
    if args.quick:
        sequence_lengths = [5, 10, 20]
        args.num_samples = 100
        args.timeout = 60
    elif args.exhaustive:
        sequence_lengths = list(range(20, 201, 20))
        args.num_samples = 1000
    
    print("="*80)
    print("STRESS TEST: Learning Performance over Sequence Length")
    print("="*80)
    print(f"Languages: {languages}")
    print(f"Learner: {args.learner}")
    if args.learner == 'ls':
        print(f"Algorithm: {args.algo}")
    print(f"Sequence Lengths: {sequence_lengths}")
    print(f"Samples per class: {args.num_samples}")
    print(f"Timeout: {args.timeout}s")
    print(f"Output: {args.output}")
    print("="*80)
    
    # Run stress tests
    results = run_stress_test(
        languages=languages,
        learner_type=args.learner,
        sequence_lengths=sequence_lengths,
        output_csv=args.output,
        algo=args.algo,
        timeout=args.timeout,
        gamma=args.gamma,
        epsilon=args.epsilon,
        num_samples=args.num_samples,
        verbose=not args.quiet,
    )
    
    # Print summary
    print_summary(results)
    
    print(f"\nResults saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
 