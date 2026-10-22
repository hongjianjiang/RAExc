#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import signal
import warnings
import os
import argparse
import csv
import subprocess
import random
from time import time
import numpy as np
import torch
import math 
from register_automata.metric_automata import *   
from register_automata.robustness import RobustnessChecker
from register_automata.raa import *
from register_automata.ra import load_smt_dra
from register_automata.ra import load_double_ls_dra, load_integer_ls_dra
from neural_networks.lstm import LSTMClassifier
from benchmark_gen import random_sample_generation, membership_query


def sample_robustness_with_gamma_epsilon(delta, epsilon, language, max_len=10, n=None):
    numerator = math.log(2 / epsilon)
    denominator = 2 * delta ** 2
    num = math.ceil(numerator / denominator)
    if (language == "LnA" or language == 'Sn') and n is not None:
        pos, neg = random_sample_generation(language, num, num, max_len, n=n)
    else:
        pos, neg = random_sample_generation(language, num, num, max_len)    # samples = pos + neg
    random.shuffle(pos)
    random.shuffle(neg)
    return pos, neg

def sample_robustness_with_gamma_epsilon_1(delta, epsilon, language, max_len=10, n=None, sample_count=None):
    import math
    numerator = math.log(2 / epsilon)
    denominator = 2 * delta ** 2
    num = sample_count if sample_count is not None else math.ceil(numerator / denominator) // 2
    # Pass n if language is LnA
    if (language == "LnA" or language == 'Sn') and n is not None:
        return random_sample_generation(language, num, num, max_len, n=n)
    else:
        return random_sample_generation(language, num, num, max_len)

def pac_sampling(delta, epsilon, language, max_len=10, model_type="transformer"):
    """
    PAC-style sampling via local search on the target language.
    Returns a list of sequences, each with a final True/False label.
    """
    pos, neg = sample_robustness_with_gamma_epsilon_1(delta, epsilon, language, max_len)
    samples = pos + neg
    # samples.sort(key=len, reverse=True)
    # random.shuffle(samples)
    # check whether there is empty sequence in samples 
    return predict_with_model(samples, language, model_type)

def active_learning_rate(auto, delta, epsilon, language, max_len=10):
    """
    PAC-style sampling via local search on the target language.
    Returns a list of sequences, each with a final True/False label.
    """
    pos, neg = sample_robustness_with_gamma_epsilon_1(delta, epsilon, language, max_len)
    correct = 0
    for i in range(len(pos)):
        if auto.run(pos[i]):
            correct += 1
        # else:
            # print(pos[i], "(pos is misclassified by the automaton")
        # correct += auto.run(pos[i])
    for i in range(len(neg)):
        if not auto.run(neg[i]):
            correct += 1
        # else:
            # print(neg[i], " (neg is misclassified by the automaton")

    return correct / (len(pos) + len(neg))
 
def predict_with_model(samples, language, model_type="lstm"):
    """
    Predict using either LSTM or transformer model based on model_type.
    
    Args:
        samples: List of sequences to predict
        language: Language name for model loading
        model_type: Either "lstm" or "transformer"
    
    Returns:
        List of [seq..., bool] predictions
    """
    if language == "mnist" and model_type in ("lstm", "transformer"):
        return predict_mnist(samples, model_type)
    if model_type == "lstm":
        return predict_with_lstm(samples, language)
    elif model_type == "transformer":
        return predict_with_transformer(samples, language)
    elif model_type == 'language':
        return predict_with_language(samples, language) 
    else:
        raise ValueError(f"Unknown model type: {model_type}")




def predict_mnist(samples, model_type):
    """MNIST row-mean networks take raw values shaped (1, length, 1), unpadded
    and unnormalised, as in neural_networks/train_mnist_{rnn,transformer}.py."""
    model = torch.jit.load(f"nn_models/{model_type}/mnist.pt", map_location=torch.device("cpu"))
    model.eval()
    res = []
    with torch.no_grad():
        for seq in samples:
            x = torch.tensor([[[float(v)] for v in seq]], dtype=torch.float)
            res.append(list(seq) + [bool(model(x).item() >= 0.5)])
    return res

def predict_with_lstm(samples, language):
    """
    Given raw sequences and a language name, load/train the LSTM
    and predict each sequence's label. Returns list of [seq..., bool].
    """

    try:
        device = torch.device('cpu')  # JIT tracing should happen on CPU for portability
        model = torch.torch.jit.load(f"nn_models/lstm/{language}.pt", map_location=device)
        model.eval()
        res = []
        for seq in samples:
            if len(seq) == 0:
                print("Empty sequence detected, using default normalization????.")
                norm_seq = [0.0]
            else:
                mean = np.mean(seq)
                std = np.std(seq)
                if std == 0:
                    norm_seq = [0.0 for _ in seq]
                else:
                    norm_seq = [(x - mean) / std for x in seq]
            padded_seq = norm_seq + [-999.0] * (51 - len(norm_seq))
            x = torch.tensor([padded_seq], dtype=torch.float).to("cpu")
            with torch.no_grad():
                prob = model(x).item()  # Model already outputs probabilities with sigmoid
            pred =  prob >= 0.5 
            new_seq = seq.copy()
            new_seq.append(pred)
            res.append(new_seq)
        return res
    except Exception as e:
        print(f"Warning: LSTM prediction failed: {e}") 

def predict_with_transformer(samples, language):
    """
    Predict using TorchScript transformer model with optimized approach.
    
    Args:
        sequences: List of sequences to predict
        language: Language name for model loading
        device: Device to run on ('cpu' or 'cuda')
        threshold: Probability threshold for classification
    
    Returns:
        List of [seq..., bool] predictions
    """
    try:
        device = torch.device('cpu')  # JIT tracing should happen on CPU for portability
        model = torch.torch.jit.load(f"nn_models/transformer/{language}.pt", map_location=device)
        model.eval()
        res = []
        for seq in samples:
            if len(seq) == 0:
                norm_seq = [0.0]
            else:
                mean = np.mean(seq)
                std = np.std(seq)
                if std == 0:
                    norm_seq = [0.0 for _ in seq]
                else:
                    norm_seq = [(x - mean) / std for x in seq]
            padded_seq = norm_seq + [-999.0] * (51 - len(norm_seq))
            x = torch.tensor([padded_seq], dtype=torch.float).to("cpu")
            with torch.no_grad():
                prob = model(x).item()  # Model already outputs probabilities with sigmoid
            pred =  prob >= 0.5 
            # print(pred, prob)
            new_seq = seq.copy()
            new_seq.append(pred)
            res.append(new_seq)
        return res
    except Exception as e:
        print(f"Warning: Transformer prediction failed: {e}")
        # res = []
        # for seq in samples:
        #     new_seq = seq.copy()
        #     new_seq.append(False)  # Default to False prediction
        #     res.append(new_seq)
        # return res

def predict_with_language(samples, language):
    """
    using benchmark.membership to predict whether the sequence is in the language or not.
    """
    return [[*seq, membership_query(seq, language)] for seq in samples]
         
            
def run_rnn_with_java(config_path, max_retries = 5, adv = None):
    """
    Invoke the Java RNN tool with the given config file.
    
    Returns:
        tuple: (success: bool, is_robust: bool or None, p: float or None)
               success: True if Java command succeeded
               is_robust: True if robust, False if not robust, None if couldn't parse
               p: The P value from the last round, None if not found
    """
    jar = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     ".", "active_learning_synthesis", "target",
                     "ralib-0.1-SNAPSHOT-jar-with-dependencies.jar")
    )
    if not adv:
        cmd = ["java", "-ea", "-jar", "--enable-native-access=ALL-UNNAMED", jar, "rnn", "-f", config_path]
    else:
        str_adv = ",".join(map(str, adv))
        cmd = ["java", "-ea", "-jar", "--enable-native-access=ALL-UNNAMED", jar, "rnn", "-f", config_path, f"adv={str_adv}"]
    for attempt in range(max_retries):
        # print(f"Exact learning No: {attempt + 1}/{max_retries}")
        try:
            r = subprocess.run(cmd, check=True, capture_output=True, text=True)       
            # Extract P value from the last round
            p = 0
            for line in r.stdout.split('\n'):
                if line.startswith('Round:') and ', P: ' in line:
                    p_part = line.split(', P: ')[1]
                    p = float(p_part)
            
            if adv:
                is_robust = None
                for line in r.stdout.split('\n'):
                    if line.startswith('Robust:'):
                        robust_value = line.split(':')[1].strip().lower()
                        is_robust = robust_value == 'true'
                        break
                return is_robust, adv, p   
            else:
                return True, None, p
        except subprocess.CalledProcessError as e:
            # Extract Robust status
            last_sequence = None
            for line in e.stderr.split('\n'):
                # Extract last sequence from counterexample logs
                if "adding counterexample:" in line or "word is not a counterexample:" in line:
                    # Find the sequence part (between ": " and " - ")
                    try:
                        if "adding counterexample:" in line:
                            sequence_part = line.split("adding counterexample: ")[1].split(" - ")[0]
                        else:  # "word is not a counterexample:"
                            sequence_part = line.split("word is not a counterexample: ")[1]
                            # Handle the case where there's " - false" at the end
                            if " - " in sequence_part:
                                sequence_part = sequence_part.split(" - ")[0]
                        
                        # Parse the comma-separated sequence
                        sequence_values = [float(x.strip()) for x in sequence_part.split(',')]
                        last_sequence = sequence_values
                    except (IndexError, ValueError) as e:
                        print(f"Failed to parse sequence from line: {line}")
                        # print(f"Error: {e}")


            if attempt == max_retries - 1:
                return False, last_sequence, 0  # Failed after all retries
 
 
def evaluate_robustness(
    dra,
    delta,
    inc,
    metric_name,
    language,
    vectors,
    model_type="lstm",
    search_adversarial=False,
    marked_adversarial=[],
    local_search=False,
    timeout=7200,
    start_index=0,
    reverse=False,
    silent=False,
    _deadline=None,  # internal: absolute monotonic deadline shared across recursion
):
    """Compatibility entry point for exact DRA robustness on accepted vectors.

    Returns ``(robust, witness, timed_out, delta, next_index)``.  A witness
    always flips this DRA at cost strictly below delta, including zero-cost
    flips and the empty word.  No accepted samples retains the legacy False
    flag.  Neural confirmation belongs to ``pac_pipeline.validate``.

    ``inc``, model/refinement flags and ``marked_adversarial`` remain accepted
    for older callers; this exact check neither shrinks delta nor suppresses
    a verdict because a witness was already seen.
    """
    import time as _t

    if _deadline is None and timeout is not None:
        _deadline = _t.monotonic() + float(timeout)

    def remaining():
        return None if _deadline is None else _deadline - _t.monotonic()

    def timed_out():
        return _deadline is not None and remaining() <= 0

    if timed_out():
        return False, None, True, delta, start_index
    dra = load_smt_dra(dra) if isinstance(dra, str) else dra
    checker = RobustnessChecker(dra, get_metric_raa(metric_name))
    accepted_count = 0
    for i in range(start_index, len(vectors)):
        if timed_out():
            return False, None, True, delta, i
        v = vectors[i]
        if not checker.accepts(v):
            continue
        accepted_count += 1
        try:
            res = checker.find_witness(v, delta, accepted=True, timeout=remaining())
        except TimeoutError:
            return False, None, True, delta, i
        if not silent:
            print(f"[{i + 1}/{len(vectors)}] {res}")
        if not res.robust:
            return False, res.witness, False, delta, i + 1
    if timed_out():
        return False, None, True, delta, len(vectors)
    return accepted_count > 0, None, False, delta, len(vectors)


def read_vector_from_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    vs = []
    with open(path) as f:
        for i,row in enumerate(csv.reader(f),1):
            try:
                nums = [float(x) for x in row if x.strip()]
                if nums: vs.append(nums)
            except ValueError:
                print(f"Skipping non-numeric row {i}: {row}")
    if not vs:
        raise ValueError("No valid numeric rows in CSV")
    return vs

def get_metric_raa(name):
    M = {
        "hamming":    create_hamming_raa,
        "l1":         create_l1_raa,
        "edit":       create_edit_raa,
        "dtw":        create_dtw_raa,
        "last_letter":create_last_letter_raa
    }
    if name not in M:
        raise ValueError(f"Unknown metric {name}")
    return M[name]()
 
def normalize_sequence(seq):
    if not seq:
        return []
    max_val = max(seq) if max(seq) > 0 else 1
    return [x / max_val for x in seq]


if __name__ == "__main__":
    print(predict_with_lstm([[1,0,-1,2]], "S6"))
