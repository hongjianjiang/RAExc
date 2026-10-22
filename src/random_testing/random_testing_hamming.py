#!/usr/bin/env python3
"""
Random Testing Experiment: Hamming Metric Perturbation

This script tests robustness by randomly flipping a single value in the sequence (Hamming distance 1).
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

def perturb_hamming(sequence: List[float], delta: float = None) -> Tuple[List[float], int, float]:
    """Flip a random value in the sequence to a random new value (Hamming distance 1)."""
    if not sequence:
        return sequence, -1, 0.0
    new_seq = sequence.copy()
    idx = random.randint(0, len(sequence) - 1)
    old_val = new_seq[idx]
    # If delta is given, restrict new value to old_val ± delta
    if delta is not None:
        min_val = max(0, int(old_val - delta))
        max_val = min(255, int(old_val + delta))
        possible_vals = [v for v in range(min_val, max_val + 1) if v != int(old_val)]
        if not possible_vals:
            possible_vals = [v for v in range(256) if v != int(old_val)]
    else:
        possible_vals = [v for v in range(256) if v != int(old_val)]
    new_val = random.choice(possible_vals)
    new_seq[idx] = float(new_val)
    return new_seq, idx, new_val

def generate_hamming_perturbations(
    sequence: List[float],
    num_trials: int = 100,
    delta: float = None
) -> List[Tuple[List[float], int, float]]:
    """
    Generate random Hamming distance 1 perturbations of a sequence.
    """
    perturbations = []
    for _ in range(num_trials):
        perturbed, idx, new_val = perturb_hamming(sequence, delta)
        perturbations.append((perturbed, idx, new_val))
    return perturbations

def run_single_sample_test(
    original_seq: List[float],
    language: str,
    model_type: str = "lstm",
    num_trials: int = 1000,
    verbose: bool = False,
    delta: float = None,
) -> Dict[str, Any]:
    """Run Hamming-perturbation test on a single sample using multiple random trials."""
    result = {
        'original_seq': original_seq,
        'original_accepted': None,
        'num_perturbations': 0,
        'num_rejected': 0,
        'num_accepted': 0,
        'rejection_rate': 0.0,
        'found_rejection': False,
        'rejection_idx': None,
        'rejection_val': None,
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
    perturbations = generate_hamming_perturbations(original_seq, num_trials, delta)
    result['num_perturbations'] = len(perturbations)
    for perturbed_seq, idx, new_val in perturbations:
        try:
            pred = predict_with_model([perturbed_seq], language, model_type)
            is_accepted = pred[0][-1]
            if is_accepted:
                result['num_accepted'] += 1
            else:
                result['num_rejected'] += 1
                result['found_rejection'] = True
                result['rejection_idx'] = idx
                result['rejection_val'] = new_val
                result['rejection_rate'] = result['num_rejected'] / (result['num_accepted'] + result['num_rejected'])
        except Exception:
            continue
    if result['num_perturbations'] > 0:
        result['rejection_rate'] = result['num_rejected'] / result['num_perturbations']
    return result

def run_random_testing_experiment(
    language: str,
    num_samples: int = 100,
    num_trials: int = 10,
    model_type: str = "lstm",
    max_seq_len: int = 10,
    output_csv: str = None,
    verbose: bool = True,
    delta: float = None,
) -> Dict[str, Any]:
    """Main experiment loop for Hamming metric."""
    experiment_result = {
        'language': language,
        'model_type': model_type,
        'num_samples': num_samples,
        'num_trials': num_trials,
        'total_samples_tested': 0,
        'samples_with_any_rejection': 0,
        'timestamp': datetime.now().isoformat(),
    }
    pos_samples, _ = random_sample_generation(language, num_samples * 2, 0, max_seq_len)
    positive_samples = [s for s in pos_samples if s][:num_samples]
    start_time = time.time()
    for i, sample in enumerate(tqdm(positive_samples, desc="Samples")):
        res = run_single_sample_test(sample, language, model_type, num_trials, delta=delta)
        if res['original_accepted']:
            experiment_result['total_samples_tested'] += 1
            if res['found_rejection']:
                experiment_result['samples_with_any_rejection'] += 1
                if verbose:
                    print(f"Not robust under HAMMING for {language}")
                    break
    total_time = time.time() - start_time
    if not res['found_rejection']:
        print(f"robust under HAMMING for {language}")

    return experiment_result

def main():
    parser = argparse.ArgumentParser(description='Robustness testing with random Hamming distance 1 perturbations.')
    parser.add_argument('--language', type=str, default='L1')
    parser.add_argument('--num_samples', type=int, default=369)
    parser.add_argument('--num_trials', type=int, default=10, help='Random points to sample')
    parser.add_argument('--delta', type=int, default=1, help='Restrict new value to old_val ± delta')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'transformer'])
    args = parser.parse_args()
    run_random_testing_experiment(
        language=args.language,
        num_samples=args.num_samples,
        num_trials=args.num_trials,
        model_type=args.model,
        delta=args.delta
    )

if __name__ == "__main__":
    main()
