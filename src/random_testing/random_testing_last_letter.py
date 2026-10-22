#!/usr/bin/env python3
"""
Random Testing Comparison Experiment: Continuous Range Perturbation

This script compares the robustness of neural networks to last-letter perturbations
by sampling values uniformly from the range [-delta, delta].
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

def perturb_last_letter(sequence: List[float], delta: float) -> List[float]:
    """Perturb the last letter of a sequence by adding delta."""
    if not sequence:
        return sequence
    new_seq = sequence.copy()
    new_seq[-1] = new_seq[-1] + delta
    return new_seq

def generate_perturbations(
    sequence: List[float],
    delta: float = 0.5,
    num_trials: int = 100
) -> List[Tuple[List[float], float]]:
    """
    Generate random perturbations of a sequence's last letter 
    by sampling uniformly from the continuous range [-delta, delta].
    """
    perturbations = []
    for _ in range(num_trials):
        # Sample a random value from the range [-delta, delta]
        random_delta = random.randint(-delta, delta)
        perturbed = perturb_last_letter(sequence, random_delta)
        perturbations.append((perturbed, random_delta))
    return perturbations

def run_single_sample_test(
    original_seq: List[float],
    language: str,
    model_type: str = "lstm",
    delta: float = 0.5,
    num_trials: int = 1000,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run perturbation test on a single sample using multiple random trials."""
    result = {
        'original_seq': original_seq,
        'original_accepted': None,
        'num_perturbations': 0,
        'num_rejected': 0,
        'num_accepted': 0,
        'rejection_rate': 0.0,
        'found_rejection': False,
        'rejection_delta': None,
    }
    
    # Verify original is accepted
    try:
        orig_pred = predict_with_model([original_seq], language, model_type)
        if not orig_pred or not orig_pred[0][-1]:
            result['original_accepted'] = False
            return result
        result['original_accepted'] = True
    except Exception as e:
        if verbose: print(f"Error: {e}")
        return result
    
    # Generate random perturbations in the range [-delta, delta]
    perturbations = generate_perturbations(original_seq, delta, num_trials)
    result['num_perturbations'] = len(perturbations)
    
    for perturbed_seq, d in perturbations:
        try:
            pred = predict_with_model([perturbed_seq], language, model_type)
            is_accepted = pred[0][-1]
            
            if is_accepted:
                result['num_accepted'] += 1
            else:
                result['num_rejected'] += 1
                result['found_rejection'] = True
                result['rejection_delta'] = d
                # Optimization: In some robustness contexts, 1 failure = non-robust.
                # If you want to see the full distribution, remove the 'return' here.
                result['rejection_rate'] = result['num_rejected'] / (result['num_accepted'] + result['num_rejected'])
        except Exception:
            continue
    
    if result['num_perturbations'] > 0:
        result['rejection_rate'] = result['num_rejected'] / result['num_perturbations']
        
    return result

def run_random_testing_experiment(
    language: str,
    num_samples: int = 100,
    delta: float = 0.5,
    num_trials: int = 10,
    model_type: str = "lstm",
    max_seq_len: int = 10,
    output_csv: str = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Main experiment loop."""
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
    
    # Generate positive samples
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
                    print(f"Not robust under Last Letter for {language}")
                    # Stop if looking for any counter-example
                    break
    if not res['found_rejection']:
        print(f"robust under Last Letter for {language}")
               
    total_time = time.time() - start_time

    return experiment_result

def main():
    parser = argparse.ArgumentParser(description='Robustness testing with range-based random perturbations.')
    parser.add_argument('--language', type=str, default='L1')
    parser.add_argument('--num_samples', type=int, default=369)
    parser.add_argument('--num_trials', type=int, default=1, help='Random points to sample in [-delta, delta]')
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