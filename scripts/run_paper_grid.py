#!/usr/bin/env python3
"""Run the paper's RQ2 grid with the PAC-validated pipeline.

Each configuration (language, network, metric) runs
src/robustness_evaluation.py --learner ls in its own process and writes its
own CSV and log under --out, so the grid can be interrupted and resumed:
configurations whose CSV already exists are skipped.  Merge the per-config
CSVs with --merge.

Example (from the repository root):
    python3 scripts/run_paper_grid.py --out results/paper_grid \
        --models_dir /path/to/nn_models --workers 6
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANGUAGES = [f"L{i}" for i in range(1, 8)] + [f"S{i}" for i in range(1, 12)]
MODELS = ["lstm", "transformer"]
METRICS = ["last_letter", "hamming", "l1", "edit", "dtw"]
# Hamming and edit distance are integer-valued; delta = 1.5 with the strict
# "< delta" neighbourhood means "at most one edit".
DELTA = {"hamming": 1.5, "edit": 1.5}


def job_name(lang, model, metric):
    return f"{lang}_{model}_{metric}"


def run_job(args):
    lang, model, metric, opts = args
    name = job_name(lang, model, metric)
    csv_path = os.path.join(opts["out"], "configs", f"{name}.csv")
    log_path = os.path.join(opts["out"], "logs", f"{name}.log")
    if os.path.exists(csv_path):
        return name, "skipped"
    cmd = [sys.executable, os.path.join(ROOT, "src", "robustness_evaluation.py"),
           "--language", lang, "--metric", metric, "--model", model,
           "--learner", "ls", "--distance", str(DELTA.get(metric, 1.0)),
           "--timeout", str(opts["learn_timeout"]),
           "--check_timeout", str(opts["check_timeout"]),
           "--max_loops", str(opts["max_loops"]),
           "--seed", str(opts["seed"]), "--csv", csv_path]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONWARNINGS="ignore")
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, cwd=opts["out"], env=env, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    return name, "ok" if rc == 0 and os.path.exists(csv_path) else f"failed (rc={rc})"


def merge(out):
    rows, fields = [], None
    for path in sorted(glob.glob(os.path.join(out, "configs", "*.csv"))):
        with open(path) as f:
            reader = csv.DictReader(f)
            fields = fields or reader.fieldnames
            rows.extend(reader)
    merged = os.path.join(out, "all_results.csv")
    with open(merged, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"merged {len(rows)} rows into {merged}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models_dir", required=True,
                    help="directory with lstm/<L>.pt and transformer/<L>.pt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--learn_timeout", type=float, default=1800.0)
    ap.add_argument("--check_timeout", type=float, default=3600.0)
    ap.add_argument("--max_loops", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--languages", nargs="+", default=LANGUAGES)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--metrics", nargs="+", default=METRICS)
    ap.add_argument("--merge", action="store_true", help="only merge existing CSVs")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "configs"), exist_ok=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    if a.merge:
        merge(out)
        return
    link = os.path.join(out, "nn_models")
    if not os.path.exists(link):
        os.symlink(os.path.abspath(a.models_dir), link)

    opts = dict(out=out, learn_timeout=a.learn_timeout, check_timeout=a.check_timeout,
                max_loops=a.max_loops, seed=a.seed)
    jobs, missing = [], []
    for lang in a.languages:
        for model in a.models:
            if not os.path.exists(os.path.join(link, model, f"{lang}.pt")):
                missing.append(f"{model}/{lang}")
                continue
            for metric in a.metrics:
                jobs.append((lang, model, metric, opts))
    if missing:
        print("skipping, no network:", ", ".join(missing))
    print(f"{len(jobs)} configurations, {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        for name, status in pool.imap_unordered(run_job, jobs):
            print(f"{name}: {status}", flush=True)
    merge(out)


if __name__ == "__main__":
    main()
