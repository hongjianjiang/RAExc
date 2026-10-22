#!/usr/bin/env python3
"""
Random Testing Experiment: L1 Metric Perturbation

This script tests robustness by randomly perturbing all values in the sequence within [-delta, delta] (L1 metric).
"""

import argparse
import csv
import os
import sys
import time
from tqdm import tqdm
import random
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# Handle path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)
os.chdir(parent_dir)

from benchmark_gen import random_sample_generation, get_available_languages
from robustness_check import predict_with_model

def perturb_l1(sequence: List[float], delta: float) -> List[float]:
    """Perturb all values in the sequence by adding random deltas in [-delta, delta]."""
    if not sequence:
        return sequence
    return [x + random.randint(-delta, delta) for x in sequence]

def generate_l1_perturbations(
    sequence: List[float],
    delta: int = 1,
    num_trials: int = 100
) -> List[List[float]]:
    """
    Generate random L1 metric perturbations of a sequence.
    """
    perturbations = []
    for _ in range(num_trials):
        perturbed = perturb_l1(sequence, delta)
        perturbations.append(perturbed)
    return perturbations

def run_single_sample_test(
    original_seq: List[float],
    language: str,
    model_type: str = "lstm",
    delta: int = 1,
    num_trials: int = 1000,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run L1-perturbation test on a single sample using multiple random trials."""
    result = {
        'original_seq': original_seq,
        'original_accepted': None,
        'num_perturbations': 0,
        'num_rejected': 0,
        'num_accepted': 0,
        'rejection_rate': 0.0,
        'found_rejection': False,
    }
    try:
        orig_pred = predict_with_model([original_seq], language, model_type)
        if not orig_pred or not orig_pred[0][-1]:
            result['original_accepted'] = False
            return result
        result['original_accepted'] = True
    except Exception as e:
        if verbose: print(f"Error: {e}")
        return result
    perturbations = generate_l1_perturbations(original_seq, delta, num_trials)
    result['num_perturbations'] = len(perturbations)
    for perturbed_seq in perturbations:
        try:
            pred = predict_with_model([perturbed_seq], language, model_type)
            is_accepted = pred[0][-1]
            if is_accepted:
                result['num_accepted'] += 1
            else:
                result['num_rejected'] += 1
                result['found_rejection'] = True
                result['rejection_rate'] = result['num_rejected'] / (result['num_accepted'] + result['num_rejected'])
        except Exception:
            continue
    if result['num_perturbations'] > 0:
        result['rejection_rate'] = result['num_rejected'] / result['num_perturbations']
    return result

def run_random_testing_experiment(
    language: str,
    num_samples: int = 100,
    delta: int = 1,
    num_trials: int = 10,
    model_type: str = "lstm",
    max_seq_len: int = 10,
    output_csv: str = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Main experiment loop for L1 metric."""
    experiment_result = {
        'language': language,
        'model_type': model_type,
        'num_samples': num_samples,
        'delta': delta,
        'num_trials': num_trials,
        'total_samples_tested': 0,
        'samples_with_any_rejection': 0,
        'timestamp': datetime.now().isoformat(),
    }
    pos_samples, _ = random_sample_generation(language, num_samples * 2, 0, max_seq_len)
    positive_samples = [s for s in pos_samples if s][:num_samples]
    start_time = time.time()
    for i, sample in enumerate(tqdm(positive_samples, desc="Samples")):
        res = run_single_sample_test(sample, language, model_type, delta, num_trials)
        if res['original_accepted']:
            experiment_result['total_samples_tested'] += 1
            if res['found_rejection']:
                experiment_result['samples_with_any_rejection'] += 1
                if verbose:
                    print(f"Not robust under L1 for {language}")
                    break
    if not res['found_rejection']:
        total_time = time.time() - start_time
        print(f"robust under L1 for {language}")

    return experiment_result

def main():
    parser = argparse.ArgumentParser(description='Robustness testing with random L1 metric perturbations.')
    parser.add_argument('--language', type=str, default='L1')
    parser.add_argument('--num_samples', type=int, default=369)
    parser.add_argument('--num_trials', type=int, default=10, help='Random points to sample in [-delta, delta]')
    parser.add_argument('--delta', type=int, default=1)
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'transformer'])
    args = parser.parse_args()
    run_random_testing_experiment(
        language=args.language,
        num_samples=args.num_samples,
        delta=args.delta,
        num_trials=args.num_trials,
        model_type=args.model
    )

if __name__ == "__main__":
    main()
