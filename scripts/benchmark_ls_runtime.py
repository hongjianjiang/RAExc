#!/usr/bin/env python3
import argparse
import contextlib
import io
import json
import math
import platform
import random
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark_gen import new_generate_language, random_sample_generation
from local_serach_synthesis.dra_learner import DRAlearner


def sample_budget(delta: float, epsilon: float) -> int:
    numerator = math.log(2 / epsilon)
    denominator = 2 * (delta ** 2)
    return math.ceil(numerator / denominator) // 2


def build_dataset(language: str, max_len: int, n: int | None, gamma: float, epsilon: float, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    num = sample_budget(gamma, epsilon)
    if language == "LnA":
        if n is None:
            raise ValueError("--n is required for LnA")
        lang = new_generate_language(language, 0, 0, max_len, n=n)
        pos, neg = random_sample_generation(language, num, num, max_len, n=n)
    else:
        lang = new_generate_language(language, 0, 0, max_len)
        pos, neg = random_sample_generation(language, num, num, max_len)
    return lang, pos, neg


def dataset_payload(language: str, max_len: int, n: int | None, gamma: float, epsilon: float, seed: int) -> dict:
    lang, pos, neg = build_dataset(language, max_len, n, gamma, epsilon, seed)
    return {
        "language": language,
        "n": n,
        "max_len": max_len,
        "gamma": gamma,
        "epsilon": epsilon,
        "seed": seed,
        "states_default": lang.states,
        "registers": len(lang.regs),
        "edge_limit_type": lang.edge_limit_type,
        "constants": [8],
        "pos": pos,
        "neg": neg,
    }


def load_or_generate_dataset(args) -> tuple[dict, list, list]:
    if args.dataset_json is not None:
        with open(args.dataset_json) as f:
            payload = json.load(f)
        return payload, payload["pos"], payload["neg"]

    payload = dataset_payload(
        language=args.language,
        max_len=args.max_len,
        n=args.n,
        gamma=args.gamma,
        epsilon=args.epsilon,
        seed=args.seed,
    )
    if args.write_dataset is not None:
        with open(args.write_dataset, "w") as f:
            json.dump(payload, f)
    return payload, payload["pos"], payload["neg"]


def fresh_learner(language: str, max_len: int, n: int | None, gamma: float, epsilon: float, seed: int, states_override: int | None):
    payload, pos, neg = load_or_generate_dataset(
        argparse.Namespace(
            language=language,
            max_len=max_len,
            n=n,
            gamma=gamma,
            epsilon=epsilon,
            seed=seed,
            dataset_json=None,
            write_dataset=None,
        )
    )
    states = states_override if states_override is not None else payload["states_default"]
    theory = "Double" if ("S" in language or "mnist" in language) else "Integer"

    random.seed(seed)
    np.random.seed(seed)
    learner = DRAlearner(
        num_states=states,
        num_registers=payload["registers"],
        constants=[8],
        pos_samples=pos,
        neg_samples=neg,
        golden_pos_samples=[],
        auto=None,
        debug=False,
        exactly_one=(payload["edge_limit_type"] == "exactly"),
        theory=theory,
    )
    return learner, payload, pos, neg


def benchmark_compute_score(learner: DRAlearner, reps: int) -> dict:
    scores = []
    learner.compute_score()
    for _ in range(reps):
        t0 = time.perf_counter()
        learner.compute_score()
        scores.append((time.perf_counter() - t0) * 1000)
    return {
        "reps": reps,
        "mean_ms": statistics.mean(scores),
        "median_ms": statistics.median(scores),
        "min_ms": min(scores),
        "max_ms": max(scores),
    }


def benchmark_transition_candidate(learner: DRAlearner, reps: int, warmup: int) -> dict:
    learner._build_trace_score_cache()

    for _ in range(warmup):
        learner.mutate_transition()
        learner._score_candidate_after_last_mutation()
        learner.revert_last_mutation()

    durations = []
    rescored = []
    deltas = []
    base_score = learner._score_candidate_after_last_mutation()

    for _ in range(reps):
        learner.mutate_transition()
        t0 = time.perf_counter()
        score = learner._score_candidate_after_last_mutation()
        durations.append((time.perf_counter() - t0) * 1000)
        rescored.append(
            learner._pending_score_update.get("rescored_samples", 0)
            if learner._pending_score_update
            else 0
        )
        deltas.append(score - base_score)
        learner.revert_last_mutation()

    return {
        "reps": reps,
        "warmup": warmup,
        "mean_ms": statistics.mean(durations),
        "median_ms": statistics.median(durations),
        "avg_rescored_samples": statistics.mean(rescored),
        "best_delta": max(deltas),
        "avg_delta": statistics.mean(deltas),
    }


def run_search(
    learner: DRAlearner,
    algo: str,
    timeout: float,
    restart_iterations: int,
    sa_max_iterations: int,
) -> dict:
    buf = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        if algo == "hc":
            learner.hill_climbing(timeout=timeout, restart_iterations=restart_iterations)
        elif algo == "sa":
            learner.simulated_annealing(
                initial_temperature=1.0,
                cooling_rate=0.995,
                max_iterations=sa_max_iterations,
                timeout=timeout,
            )
        else:
            raise ValueError(f"Unsupported algo: {algo}")
    elapsed = time.perf_counter() - start
    captured = buf.getvalue()

    result = {
        "algo": algo,
        "elapsed_s": elapsed,
        "best_score": learner.best_score,
    }

    if algo == "hc":
        stats = learner.get_search_stats()
        result["stats"] = stats
    else:
        match = re.search(
            r"Stopped after (\d+) iterations, ([0-9]+(?:\.[0-9]+)?)s elapsed\. Best Score=([0-9]+(?:\.[0-9]+)?)\.?",
            captured,
        )
        if match:
            result["iterations"] = int(match.group(1))
            result["reported_elapsed_s"] = float(match.group(2))
            result["reported_best_score"] = float(match.group(3))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local-search runtime under the current interpreter.")
    parser.add_argument("--language", required=True)
    parser.add_argument("--algo", choices=["hc", "sa"], default="sa")
    parser.add_argument("--n", type=int, default=None, help="LnA parameter")
    parser.add_argument("--states", type=int, default=None, help="Override automaton state count")
    parser.add_argument("--max_len", type=int, default=110)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--score_reps", type=int, default=30)
    parser.add_argument("--transition_reps", type=int, default=1000)
    parser.add_argument("--transition_warmup", type=int, default=200)
    parser.add_argument("--restart_iterations", type=int, default=100000)
    parser.add_argument("--sa_max_iterations", type=int, default=100000)
    parser.add_argument("--dataset_json", type=Path, default=None, help="Load a pre-generated dataset from JSON")
    parser.add_argument("--write_dataset", type=Path, default=None, help="Write the generated dataset to JSON before benchmarking")
    args = parser.parse_args()

    payload, pos, neg = load_or_generate_dataset(args)
    states = args.states if args.states is not None else payload["states_default"]
    theory = "Double" if ("S" in args.language or "mnist" in args.language) else "Integer"

    def build_learner():
        random.seed(args.seed)
        np.random.seed(args.seed)
        return DRAlearner(
            num_states=states,
            num_registers=payload["registers"],
            constants=payload.get("constants", [8]),
            pos_samples=pos,
            neg_samples=neg,
            golden_pos_samples=[],
            auto=None,
            debug=False,
            exactly_one=(payload["edge_limit_type"] == "exactly"),
            theory=theory,
        )

    learner = build_learner()

    compute_stats = benchmark_compute_score(learner, args.score_reps)

    learner = build_learner()
    transition_stats = benchmark_transition_candidate(
        learner,
        reps=args.transition_reps,
        warmup=args.transition_warmup,
    )

    learner = build_learner()
    search_stats = run_search(
        learner,
        algo=args.algo,
        timeout=args.timeout,
        restart_iterations=args.restart_iterations,
        sa_max_iterations=args.sa_max_iterations,
    )

    payload = {
        "language": args.language,
        "algo": args.algo,
        "n": args.n,
        "states": states,
        "max_len": args.max_len,
        "seed": args.seed,
        "interpreter": {
            "implementation": platform.python_implementation(),
            "version": sys.version,
            "executable": sys.executable,
        },
        "dataset": {
            "pos_samples": len(pos),
            "neg_samples": len(neg),
            "registers": payload["registers"],
            "edge_limit_type": payload["edge_limit_type"],
            "source": str(args.dataset_json) if args.dataset_json is not None else "generated",
        },
        "benchmarks": {
            "compute_score": compute_stats,
            "transition_candidate": transition_stats,
            "search": search_stats,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
