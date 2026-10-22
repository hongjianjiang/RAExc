#!/usr/bin/env python3
import argparse
import copy
import contextlib
import io
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from local_serach_synthesis.dra_learner import DRAlearner


def load_dataset(path: Path):
    with open(path) as f:
        payload = json.load(f)

    train = payload.get("train", {"pos": payload["pos"], "neg": payload["neg"]})
    eval_split = payload.get("eval", train)
    return payload, train, eval_split


def build_learner(payload, train_split, states, seed):
    random.seed(seed)
    np.random.seed(seed)
    return DRAlearner(
        num_states=states,
        num_registers=payload["registers"],
        constants=payload.get("constants", [0]),
        pos_samples=train_split["pos"],
        neg_samples=train_split["neg"],
        golden_pos_samples=[],
        auto=None,
        debug=False,
        exactly_one=(payload.get("edge_limit_type") == "exactly"),
        theory=payload.get("theory", "Integer"),
    )


def capture_dra_state(dra):
    return {
        "initial_state": dra.initial_state,
        "final_states": set(dra.final_states),
        "transitions": {state: edges.copy() for state, edges in dra.transitions.items()},
        "register_transitions": {
            state: edges.copy()
            for state, edges in dra.register_transitions.items()
        },
    }


def warm_start_learner(learner, dra_state):
    if dra_state is None:
        return

    learner.mydra.initial_state = dra_state["initial_state"]
    learner.mydra.final_states = set(dra_state["final_states"])
    for state, edges in dra_state["transitions"].items():
        learner.mydra.transitions[state] = edges.copy()
    for state, edges in dra_state["register_transitions"].items():
        learner.mydra.register_transitions[state] = edges.copy()

    learner.best_dra = copy.deepcopy(learner.mydra)
    learner._invalidate_score_trace_cache()
    learner.best_score = learner.compute_score()


def evaluate_active_states(dra, pos, neg):
    visited = set()
    for sample in pos + neg:
        trace = dra.evaluate_input_trace(sample)
        visited.update(trace.source_states)
        if trace.completed and trace.final_state is not None:
            visited.add(trace.final_state)
    return len(visited)


def pick_states(args, payload):
    if args.states:
        return args.states
    recommended = payload.get("recommended_state_sweep")
    if recommended:
        return recommended
    claim = payload.get("min_active_states_claim", 100)
    return [10, 20, 40, 60, 80, claim, claim + 20]


def run_search(learner, algo, timeout, sa_initial_temperature, sa_cooling_rate, sa_max_iterations, hc_restart_iterations):
    capture = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(capture):
        if algo == "sa":
            learner.simulated_annealing(
                initial_temperature=sa_initial_temperature,
                cooling_rate=sa_cooling_rate,
                max_iterations=sa_max_iterations,
                timeout=timeout,
            )
        elif algo == "hc":
            learner.hill_climbing(timeout=timeout, restart_iterations=hc_restart_iterations)
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")
    elapsed = time.perf_counter() - start
    return elapsed, capture.getvalue(), learner.get_search_stats()


def run_one(payload, train_split, eval_split, states, algo, timeout, seed, args, warm_start_state=None, warm_started_from=None):
    learner = build_learner(payload, train_split, states, seed)
    warm_start_learner(learner, warm_start_state)
    elapsed, captured_output, search_stats = run_search(
        learner,
        algo=algo,
        timeout=timeout,
        sa_initial_temperature=args.sa_initial_temperature,
        sa_cooling_rate=args.sa_cooling_rate,
        sa_max_iterations=args.sa_max_iterations,
        hc_restart_iterations=args.hc_restart_iterations,
    )
    best_dra = learner.best_dra
    eval_accuracy = best_dra.correctness_check(eval_split["pos"], eval_split["neg"])
    train_accuracy = best_dra.correctness_check(train_split["pos"], train_split["neg"])
    active_states_eval = evaluate_active_states(best_dra, eval_split["pos"], eval_split["neg"])
    active_states_train = evaluate_active_states(best_dra, train_split["pos"], train_split["neg"])
    num_transitions = sum(len(v) for v in best_dra.transitions.values())

    result = {
        "states": states,
        "seed": seed,
        "algo": algo,
        "wall_time_s": elapsed,
        "train_accuracy": train_accuracy,
        "eval_accuracy": eval_accuracy,
        "active_states_train": active_states_train,
        "active_states_eval": active_states_eval,
        "num_transitions": num_transitions,
        "warm_started": warm_start_state is not None,
        "warm_started_from": warm_started_from,
        "search_output": captured_output.strip(),
        "search_stats": search_stats,
    }

    if search_stats:
        proposals = search_stats.get("proposals", 0)
        total_time = search_stats.get("timing", {}).get("total", 0.0)
        result["proposals_per_second"] = proposals / total_time if total_time > 0 else 0.0
        result["avg_scoring_ms"] = (
            1000.0 * search_stats.get("timing", {}).get("scoring", 0.0) / proposals
            if proposals > 0
            else 0.0
        )
    else:
        result["proposals_per_second"] = 0.0
        result["avg_scoring_ms"] = 0.0

    return result, capture_dra_state(best_dra)


def summarize_runs(runs):
    if not runs:
        return {}

    summary = {}
    numeric_fields = [
        "train_accuracy",
        "eval_accuracy",
        "active_states_train",
        "active_states_eval",
        "num_transitions",
        "wall_time_s",
        "proposals_per_second",
        "avg_scoring_ms",
    ]

    for field in numeric_fields:
        values = [run[field] for run in runs]
        summary[field] = {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }

    summary["runs"] = len(runs)
    summary["perfect_train_fraction"] = statistics.mean(
        1.0 if abs(run["train_accuracy"] - 1.0) < 1e-12 else 0.0 for run in runs
    )
    summary["perfect_eval_fraction"] = statistics.mean(
        1.0 if abs(run["eval_accuracy"] - 1.0) < 1e-12 else 0.0 for run in runs
    )
    summary["eval_at_least_95_fraction"] = statistics.mean(
        1.0 if run["eval_accuracy"] >= 0.95 else 0.0 for run in runs
    )

    if runs[0]["search_stats"]:
        proposals = [run["search_stats"].get("proposals", 0) for run in runs]
        acceptance = [run["search_stats"].get("acceptance_rate", 0.0) for run in runs]
        improving = [run["search_stats"].get("improving_acceptance_rate", 0.0) for run in runs]
        avg_rescored = [run["search_stats"].get("avg_rescored_samples", 0.0) for run in runs]
        summary["search"] = {
            "proposals_mean": statistics.mean(proposals),
            "acceptance_rate_mean": statistics.mean(acceptance),
            "improving_acceptance_rate_mean": statistics.mean(improving),
            "avg_rescored_samples_mean": statistics.mean(avg_rescored),
        }

        operators = {}
        for operator_name in runs[0]["search_stats"]["operators"]:
            operator_runs = [run["search_stats"]["operators"][operator_name] for run in runs]
            operators[operator_name] = {
                "acceptance_rate_mean": statistics.mean(
                    item.get("acceptance_rate", 0.0) for item in operator_runs
                ),
                "improving_acceptance_rate_mean": statistics.mean(
                    item.get("improving_acceptance_rate", 0.0) for item in operator_runs
                ),
                "avg_delta_mean": statistics.mean(item.get("avg_delta", 0.0) for item in operator_runs),
                "avg_scoring_time_ms_mean": 1000.0
                * statistics.mean(item.get("avg_scoring_time", 0.0) for item in operator_runs),
                "avg_rescored_samples_mean": statistics.mean(
                    item.get("avg_rescored_samples", 0.0) for item in operator_runs
                ),
            }
        summary["operators"] = operators

    return summary


def write_csv(path: Path, results):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "states",
                "seed",
                "algo",
                "warm_started",
                "warm_started_from",
                "train_accuracy",
                "eval_accuracy",
                "active_states_train",
                "active_states_eval",
                "num_transitions",
                "wall_time_s",
                "proposals",
                "acceptance_rate",
                "improving_acceptance_rate",
                "avg_rescored_samples",
                "proposals_per_second",
                "avg_scoring_ms",
            ]
        )
        for run in results:
            stats = run.get("search_stats") or {}
            writer.writerow(
                [
                    run["states"],
                    run["seed"],
                    run["algo"],
                    run["warm_started"],
                    run["warm_started_from"],
                    run["train_accuracy"],
                    run["eval_accuracy"],
                    run["active_states_train"],
                    run["active_states_eval"],
                    run["num_transitions"],
                    run["wall_time_s"],
                    stats.get("proposals", 0),
                    stats.get("acceptance_rate", 0.0),
                    stats.get("improving_acceptance_rate", 0.0),
                    stats.get("avg_rescored_samples", 0.0),
                    run["proposals_per_second"],
                    run["avg_scoring_ms"],
                ]
            )


def main():
    parser = argparse.ArgumentParser(
        description="Run local-search state sweeps on a frozen dataset without regenerating samples."
    )
    parser.add_argument("--dataset_json", type=Path, required=True)
    parser.add_argument("--algo", choices=["sa", "hc"], default="sa")
    parser.add_argument("--states", type=int, nargs="*", default=None)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sa_initial_temperature", type=float, default=1.0)
    parser.add_argument("--sa_cooling_rate", type=float, default=0.995)
    parser.add_argument("--sa_max_iterations", type=int, default=100000)
    parser.add_argument("--hc_restart_iterations", type=int, default=100000)
    parser.add_argument("--continuation", action="store_true")
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_csv", type=Path, default=None)
    args = parser.parse_args()

    payload, train_split, eval_split = load_dataset(args.dataset_json)
    states_to_run = pick_states(args, payload)

    all_results = []
    grouped_results = {}

    if args.continuation:
        state_to_runs = {str(states): [] for states in states_to_run}
        for repeat in range(args.repeats):
            run_seed = args.seed + repeat
            warm_start_state = None
            previous_states = None
            for states in states_to_run:
                result, warm_start_state = run_one(
                    payload=payload,
                    train_split=train_split,
                    eval_split=eval_split,
                    states=states,
                    algo=args.algo,
                    timeout=args.timeout,
                    seed=run_seed,
                    args=args,
                    warm_start_state=warm_start_state,
                    warm_started_from=previous_states,
                )
                state_to_runs[str(states)].append(result)
                all_results.append(result)
                previous_states = states
        grouped_results = {
            state: summarize_runs(runs)
            for state, runs in state_to_runs.items()
        }
    else:
        for states in states_to_run:
            state_runs = []
            for repeat in range(args.repeats):
                run_seed = args.seed + repeat
                result, _ = run_one(
                    payload=payload,
                    train_split=train_split,
                    eval_split=eval_split,
                    states=states,
                    algo=args.algo,
                    timeout=args.timeout,
                    seed=run_seed,
                    args=args,
                )
                state_runs.append(result)
                all_results.append(result)
            grouped_results[str(states)] = summarize_runs(state_runs)

    report = {
        "dataset": {
            "path": str(args.dataset_json),
            "task": payload.get("task"),
            "prefix_length": payload.get("prefix_length"),
            "max_len": payload.get("max_len"),
            "registers": payload.get("registers"),
            "constants": payload.get("constants", [0]),
            "train_pos": len(train_split["pos"]),
            "train_neg": len(train_split["neg"]),
            "eval_pos": len(eval_split["pos"]),
            "eval_neg": len(eval_split["neg"]),
            "min_active_states_claim": payload.get("min_active_states_claim"),
        },
        "run_config": {
            "algo": args.algo,
            "timeout": args.timeout,
            "repeats": args.repeats,
            "base_seed": args.seed,
            "continuation": args.continuation,
            "states": states_to_run,
        },
        "summary_by_state": grouped_results,
        "runs": all_results,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)

    if args.output_csv is not None:
        write_csv(args.output_csv, all_results)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
