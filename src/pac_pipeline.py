"""PAC-validated robust DRA extraction (Section 5 of the paper).

One configuration (language, network, metric, delta) runs the loop:

1. Learn a candidate DRA with local search from samples labelled by the
   network N.
2. Validation: draw n = ceil(ln(1/eps) / (2 gamma^2)) fresh samples from the
   reference distribution D and label them with N.  If more than
   d_max = floor(n (1 - p)) of them disagree with the DRA, refine.
3. On every agreeing sample v, run the exact checker
   (register_automata.robustness).  A witness w that also flips N is a
   genuine non-robustness witness and ends the run.  A witness on which N does
   not flip is spurious and becomes a refinement example labelled by N.
4. Accept the DRA iff fidelity holds and no flip was found.  Otherwise add the
   refinement examples to the sample set and go back to 1, up to max_loops
   rounds.

The reference distribution D is the benchmark's sampler
(benchmark_gen.random_sample_generation), mixing positive and negative
sequences of the language in equal proportion.
"""

import csv
import math
import os
import random
import time
from typing import List, Optional

from benchmark_gen import random_sample_generation
from local_serach_synthesis import dra_learner
from register_automata.ra import load_double_ls_dra, load_integer_ls_dra
from register_automata.robustness import RobustnessChecker
from robustness_check import get_metric_raa, predict_with_model

# The LSTM and transformer classifiers pad inputs to this length.
MODEL_MAX_LEN = 51

CSV_FIELDS = [
    "language", "model", "metric", "delta", "states", "registers",
    "outcome", "rounds", "n_validation", "d_max", "disagreements_last",
    "spurious_total", "unqueryable_total", "checked_last",
    "witness_v", "witness_w", "witness_cost", "train_score",
    "learning_time", "check_time", "query_time", "total_time", "seed",
]


def _label(samples: List[List[float]], language: str, model_type: str) -> List[bool]:
    if not samples:
        return []
    too_long = [s for s in samples if len(s) > MODEL_MAX_LEN]
    if too_long:
        raise ValueError(f"sequence of length {len(too_long[0])} exceeds the "
                         f"network input length {MODEL_MAX_LEN}")
    out = predict_with_model(samples, language, model_type)
    if out is None:
        raise RuntimeError(f"no {model_type} model available for {language} "
                           f"(expected nn_models/{model_type}/{language}.pt)")
    return [bool(row[-1]) for row in out]


def _draw(language, count, max_len, n_param):
    """Draw `count` sequences from the reference distribution D."""
    half = (count + 1) // 2
    kwargs = {"n": n_param} if n_param is not None else {}
    pos, neg = random_sample_generation(language, half, half, max_len, **kwargs)
    samples = list(pos) + list(neg)
    random.shuffle(samples)
    return samples[:count]


class ValidationOutcome:
    def __init__(self):
        self.status = None          # "accepted" | "genuine" | "refine" | "timeout"
        self.refine_pos: List[list] = []
        self.refine_neg: List[list] = []
        self.disagreements = 0
        self.spurious = 0
        self.unqueryable = 0
        self.checked = 0
        self.witness = None         # (v, w, cost) for a genuine witness
        self.check_time = 0.0
        self.query_time = 0.0


def validate(dra, language, model_type, metric, delta, p, gamma, epsilon,
             max_len, n_param=None, deadline=None) -> ValidationOutcome:
    """One validation phase for a candidate DRA (Lemmas 5 and 6)."""
    out = ValidationOutcome()
    n = math.ceil(math.log(1 / epsilon) / (2 * gamma ** 2))
    d_max = math.floor(n * (1 - p))
    checker = RobustnessChecker(dra, get_metric_raa(metric))
    samples = _draw(language, n, max_len, n_param)
    t = time.monotonic()
    labels = _label(samples, language, model_type)
    out.query_time += time.monotonic() - t

    agreeing = []
    for s, y in zip(samples, labels):
        if checker.accepts(s) == y:
            agreeing.append((s, y))
        else:
            out.disagreements += 1
            (out.refine_pos if y else out.refine_neg).append(s)
    if out.disagreements > d_max:
        out.status = "refine"
        return out

    for v, y in agreeing:
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            out.status = "timeout"
            return out
        t = time.monotonic()
        try:
            res = checker.find_witness(v, delta, accepted=y, timeout=remaining)
        except TimeoutError:
            out.check_time += time.monotonic() - t
            out.status = "timeout"
            return out
        out.check_time += time.monotonic() - t
        out.checked += 1
        if res.robust:
            continue
        w = res.witness
        if len(w) > MODEL_MAX_LEN:
            # The DRA flips, so it cannot be accepted, but N cannot be queried.
            out.unqueryable += 1
            continue
        t = time.monotonic()
        yw = _label([w], language, model_type)[0]
        out.query_time += time.monotonic() - t
        if yw != y:
            out.status = "genuine"
            out.witness = (v, w, float(res.cost))
            return out
        out.spurious += 1
        (out.refine_pos if yw else out.refine_neg).append(w)

    if out.spurious == 0 and out.unqueryable == 0:
        out.status = "accepted"
    else:
        out.status = "refine"
    return out


def run_pac_experiment(language, metric, delta, model_type, states, regs,
                       edge_type, theory, csv_path, max_loops=10, constants=(),
                       learn_timeout=1800.0, check_timeout=3600.0,  # per configuration
                       p=0.95, gamma=0.05, epsilon=0.05, max_len=10,
                       n_param=None, seed=0, silent=False):
    """Run the PAC-validated extraction loop for one configuration."""
    random.seed(seed)
    start = time.monotonic()
    row = {k: "" for k in CSV_FIELDS}
    row.update(language=language, model=model_type, metric=metric, delta=delta,
               states=states, registers=regs, seed=seed,
               n_validation=math.ceil(math.log(1 / epsilon) / (2 * gamma ** 2)))
    row["d_max"] = math.floor(row["n_validation"] * (1 - p))

    # Initial training sample from D, labelled by the network (not by the language).
    init = _draw(language, 2 * 369, max_len, n_param)
    t = time.monotonic()
    labels = _label(init, language, model_type)
    query_time = time.monotonic() - t
    pos = [s for s, y in zip(init, labels) if y]
    neg = [s for s, y in zip(init, labels) if not y]
    golden: List[list] = []

    learning_time = check_time = 0.0
    spurious_total = unqueryable_total = 0
    best_auto = None
    outcome = "inconclusive"
    os.makedirs("ls_dra_json", exist_ok=True)
    check_deadline = time.monotonic() + check_timeout
    learn_deadline = time.monotonic() + learn_timeout
    rounds = 0
    for rounds in range(1, max_loops + 1):
        t = time.monotonic()
        # the learning budget is per configuration, shared evenly by the
        # remaining rounds
        round_budget = max(1.0, (learn_deadline - t) / (max_loops - rounds + 1))
        learner = dra_learner.DRAlearner(
            num_states=states, num_registers=regs, constants=list(constants),
            pos_samples=pos, neg_samples=neg, golden_pos_samples=golden,
            auto=best_auto, debug=False, exactly_one=edge_type, theory=theory)
        learner.simulated_annealing(initial_temperature=1.0, cooling_rate=0.995,
                                    max_iterations=50000, timeout=round_budget)
        best_auto = learner.best_dra.translate_to_initial_zero()
        path = f"ls_dra_json/pac_{language}_{model_type}_{metric}_r{rounds}.json"
        best_auto.to_json(path, theory)
        dra = load_double_ls_dra(path) if theory == "Double" else load_integer_ls_dra(path)
        learning_time += time.monotonic() - t
        row["train_score"] = learner.best_score

        res = validate(dra, language, model_type, metric, delta, p, gamma, epsilon,
                       max_len, n_param, deadline=check_deadline)
        check_time += res.check_time
        query_time += res.query_time
        spurious_total += res.spurious
        unqueryable_total += res.unqueryable
        row.update(disagreements_last=res.disagreements, checked_last=res.checked)
        if not silent:
            print(f"[{language}/{model_type}/{metric}] round {rounds}: {res.status} "
                  f"(disagreements {res.disagreements}/{row['d_max']}, "
                  f"checked {res.checked}, spurious {res.spurious})", flush=True)

        if res.status == "accepted":
            outcome = "robust"
            break
        if res.status == "genuine":
            outcome = "non-robust"
            v, w, cost = res.witness
            row.update(witness_v=v, witness_w=w, witness_cost=cost)
            break
        if res.status == "timeout":
            outcome = "timeout"
            break
        pos += res.refine_pos
        neg += res.refine_neg
        golden += res.refine_pos

    row.update(outcome=outcome, rounds=rounds, spurious_total=spurious_total,
               unqueryable_total=unqueryable_total,
               learning_time=f"{learning_time:.2f}", check_time=f"{check_time:.2f}",
               query_time=f"{query_time:.2f}", total_time=f"{time.monotonic() - start:.2f}")
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new_file = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return row
