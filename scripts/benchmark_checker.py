#!/usr/bin/env python3
"""Compare one-shot queries, compiled reuse, and register liveness.

Run from the repository root:
  python3 scripts/benchmark_checker.py --repeats 3 --samples 5

Optionally compare one-shot queries with an earlier robustness.py snapshot:
  python3 scripts/benchmark_checker.py --reference /tmp/robustness_before.py

CSV timings include compilation once per batch for the reused modes, or once
per sample for one-shot mode.  Every mode must agree on verdict and infimum.
These synthetic cases exercise both witness search and proofs of robustness.
"""
import argparse
import csv
import importlib.util
from pathlib import Path
import statistics
import sys
import time
from fractions import Fraction

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import z3
from register_automata.metric_automata import (
    create_dtw_raa, create_edit_raa, create_hamming_raa,
    create_l1_raa, create_last_letter_raa,
)
from register_automata.ra import RegisterAutomata, Transition
from register_automata.robustness import RobustnessChecker, find_witness

METRICS = {
    "hamming": create_hamming_raa, "l1": create_l1_raa,
    "edit": create_edit_raa, "dtw": create_dtw_raa,
    "last_letter": create_last_letter_raa,
}


def cases(length):
    c, r0, r1 = z3.Reals("curr r0 r1")
    increasing = RegisterAutomata(2, [
        Transition(c, 0, True, {"r0": "curr"}, 1),
        Transition(c, 1, c >= r0, {"r0": "curr"}, 1),
    ], [r0], [1])
    dead_suffix = RegisterAutomata(3, [
        Transition(c, 0, True, {"r0": "curr"}, 1),
        Transition(c, 1, c < r0, {}, 2),
        Transition(c, 2, True, {"r0": "curr", "r1": "r0"}, 2),
    ], [r0, r1], [2])
    return [("increasing", increasing, list(range(1, length + 1))),
            ("dead_suffix", dead_suffix, [10, 5] + list(range(1, length - 1)))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--length", type=int, default=6)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.samples < 1 or args.length < 3:
        parser.error("repeats and samples must be positive, length must be >= 3")
    one_shot = find_witness
    if args.reference:
        spec = importlib.util.spec_from_file_location("checker_reference", args.reference)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        one_shot = reference.find_witness
    writer = csv.writer(sys.stdout)
    writer.writerow(["case", "metric", "mode", "samples", "median_batch_seconds",
                     "expanded_per_query", "speedup_vs_one_shot"])
    for name, dra, v in cases(args.length):
        for metric in args.metrics:
            raa = METRICS[metric]()
            timings = {mode: [] for mode in ("one_shot", "reused", "pruned")}
            expansions = {}
            expected = None
            for repeat in range(args.repeats):
                # Rotate execution order to reduce warmup/order effects.
                modes = list(timings)
                modes = modes[repeat % 3:] + modes[:repeat % 3]
                for mode in modes:
                    start = time.perf_counter()
                    checker = (None if mode == "one_shot" else
                               RobustnessChecker(dra, raa, prune_registers=mode == "pruned"))
                    expanded = 0
                    for _ in range(args.samples):
                        if checker is None:
                            result = one_shot(dra, raa, v, Fraction(1, 2),
                                              accepted=True, timeout=30)
                        else:
                            result = checker.find_witness(v, Fraction(1, 2),
                                                          accepted=True, timeout=30)
                        signature = result.robust, result.infimum
                        if expected is None:
                            expected = signature
                        if signature != expected:
                            raise AssertionError((name, metric, mode, expected, signature))
                        expanded += result.expanded
                    timings[mode].append(time.perf_counter() - start)
                    expansions[mode] = expanded / args.samples
            baseline = statistics.median(timings["one_shot"])
            for mode, samples in timings.items():
                elapsed = statistics.median(samples)
                writer.writerow([name, metric, mode, args.samples, f"{elapsed:.6f}",
                                 f"{expansions[mode]:g}", f"{baseline / elapsed:.2f}"])
            sys.stdout.flush()


if __name__ == "__main__":
    main()
