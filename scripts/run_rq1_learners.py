"""RQ1: learn each benchmark language with the SMT, local-search and active learners.

Both learners get the same 369 positive and 369 negative ground-truth
samples (lengths 1..10, same seed), the language's number of states and
registers, its edge limit and its constants. Each run is a subprocess that is
killed after --timeout seconds. Reported per run: balanced accuracy on the
training samples, balanced accuracy on fresh held-out samples, learning time
and the size of the learned DRA. Learned DRAs go to runs/<lang>_<learner>_dra.json;
for the SMT learner, the formula of its last solver call (cumulative over all
sample batches) goes to runs/<lang>_smt.smt2.

    python3 scripts/run_rq1_learners.py --out results/rq1_learners --workers 6
"""
import argparse
import csv
import glob
import json
import os
import random
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

LANGUAGES = [f"L{i}" for i in range(1, 8)] + [f"S{i}" for i in range(1, 12)] + ["mnist"]
LEARNERS = ["smt", "ls", "al"]
N_TRAIN, N_TEST, MAX_LEN = 369, 1000, 10
FIELDS = ["language", "learner", "status", "train_acc", "test_acc", "n_test",
          "learning_time", "states", "registers", "transitions", "seed"]


def samples(language, seed):
    """Training and held-out samples; deterministic in (language, seed)."""
    from benchmark_gen import Language, random_sample_generation
    random.seed(seed)
    np.random.seed(seed)
    if language == "mnist":
        lang = {c.__name__: c for c in Language.__subclasses__()}["mnist"](0, 0, 0)
        pos, neg = list(lang.pos), list(lang.neg)
        random.shuffle(pos)
        random.shuffle(neg)
        return (pos[:N_TRAIN], neg[:N_TRAIN]), (pos[N_TRAIN:], neg[N_TRAIN:])
    train = random_sample_generation(language, N_TRAIN, N_TRAIN, MAX_LEN)
    from net_accuracy import sample
    test = sample(language, 2 * N_TEST, MAX_LEN, seed + 1000)
    return train, test


def balanced_acc(dra, pos, neg):
    """Balanced accuracy under the paper's DRA semantics.

    Uses the robustness checker's compiled DRA, whose register assignments
    are parallel (RegisterAutomata.run applies them one after another).
    """
    from fractions import Fraction
    from register_automata.robustness import _CompiledDRA
    c = _CompiledDRA(dra, set())

    def accepts(s):
        return c.run([Fraction(x) for x in s])
    tp = sum(bool(accepts(s)) for s in pos)
    tn = sum(not accepts(s) for s in neg)
    return (tp / len(pos) + tn / len(neg)) / 2


def load_learned(learner, language, path):
    from register_automata.ra import load_double_ls_dra, load_integer_ls_dra, load_smt_dra
    if learner in ("smt", "al"):
        return load_smt_dra(path)
    theory = "Double" if ("S" in language or language == "mnist") else "Integer"
    return load_double_ls_dra(path) if theory == "Double" else load_integer_ls_dra(path)


def rescore(out):
    """Recompute train/test accuracy of every saved DRA without relearning."""
    for p in sorted(glob.glob(os.path.join(out, "runs", "*.json"))):
        if p.endswith("_dra.json"):
            continue
        res = json.load(open(p))
        dra_path = p.replace(".json", "_dra.json")
        if not os.path.exists(dra_path):
            continue
        dra = load_learned(res["learner"], res["language"], dra_path)
        (pos, neg), (tpos, tneg) = samples(res["language"], res["seed"])
        res.update(train_acc=round(100 * balanced_acc(dra, pos, neg), 2),
                   test_acc=round(100 * balanced_acc(dra, tpos, tneg), 2))
        json.dump(res, open(p, "w"))


def learn_one(learner, language, seed, timeout, out_json):
    """Child process: learn, evaluate, write one JSON result."""
    from benchmark_gen import new_generate_language
    from register_automata.ra import load_double_ls_dra, load_integer_ls_dra
    lang = new_generate_language(language, 0, 0, 0)
    theory = "Double" if ("S" in language or language == "mnist") else "Integer"
    (pos, neg), (tpos, tneg) = samples(language, seed)
    res = dict(language=language, learner=learner, seed=seed, n_test=len(tpos) + len(tneg))
    t0 = time.monotonic()
    if learner == "smt":
        if theory == "Integer":
            from smt_based_synthesis import eqsmtlearner as m
            lr = m.Register_Learner(n=lang.states, regs=lang.regs, ic=lang.ic, uc=lang.uc,
                                    pos=pos, neg=neg, edge_limit_type=lang.edge_limit_type,
                                    edge_limit_num=lang.edge_limit_num, debug=False,
                                    batch_size=15, target=0.95)
        else:
            from smt_based_synthesis import smtlearner as m
            lr = m.Register_Learner(n=lang.states, regs=lang.regs, ic=lang.ic, uc=lang.uc,
                                    pos=pos, neg=neg, edge_limit_type=lang.edge_limit_type,
                                    edge_limit_num=lang.edge_limit_num, debug=False, auto=None,
                                    batch_size=5, strategy="edge", target=0.95)
        # Keep the formula: before every solver call, write the current
        # assertions as SMT-LIB 2, so a run killed at the timeout still
        # leaves the formula it was solving.
        solver, smt2 = lr._solver, out_json.replace(".json", ".smt2")
        z3_check = solver.check

        def check(*args):
            with open(smt2, "w") as f:
                f.write(solver.to_smt2())
            return z3_check(*args)
        solver.check = check
        dra, _ = lr.get_model(timeout=timeout)
        if isinstance(dra, str):  # "unsat" or "timeout"
            res.update(status=dra, learning_time=round(time.monotonic() - t0, 2))
            json.dump(res, open(out_json, "w"))
            return
        dra.save_to_json(out_json.replace(".json", "_dra.json"))
    elif learner == "al":
        # RALib (SL-lambda) with exact membership queries to the language and
        # sampling-based equivalence checks (eps = gamma = 0.05). It runs from
        # the repository root, which its sample generator and model export
        # paths are relative to, and saves its hypothesis after every round.
        from register_automata.ra import load_smt_dra
        if language == "mnist":  # no membership oracle for MNIST
            res.update(status="n/a", learning_time=0)
            json.dump(res, open(out_json, "w"))
            return
        teachers = ("double:de.learnlib.ralib.tools.theories.DoubleInequalityTheory"
                    if theory == "Double" else
                    "int:de.learnlib.ralib.tools.theories.IntegerEqualityTheory")
        cfg = out_json.replace(".json", ".config")
        with open(cfg, "w") as f:
            f.write(f"delta=0.05\nepsilon=0.05\np=0.01\nfixTraceLen=false\n"
                    f"datatype={'double' if theory == 'Double' else 'integer'}\n"
                    f"modelFileName=unused\nteachers={teachers}\nrnn=language\n"
                    f"learner=sllambda\nqueryRNNFileName={out_json.replace('.json', '.queries')}\n"
                    f"timeout={int(timeout)}\nlanguage={language}\nmaxLen={MAX_LEN}\nseed={seed}\n")
        model = os.path.join(ROOT, "active_models", f"{language}.xml")
        if os.path.exists(model):
            os.remove(model)
        jar = os.path.join(ROOT, "src", "active_learning_synthesis", "target",
                           "ralib-0.1-SNAPSHOT-jar-with-dependencies.jar")
        r = subprocess.run(["java", "-ea", "--enable-native-access=ALL-UNNAMED", "-jar", jar,
                            "rnn", "-f", cfg], cwd=ROOT, capture_output=True, text=True)
        with open(out_json.replace(".json", ".stdout"), "w") as f:
            f.write(r.stdout + r.stderr)
        if not os.path.exists(model):
            res.update(status=f"error (rc={r.returncode})",
                       learning_time=round(time.monotonic() - t0, 2))
            json.dump(res, open(out_json, "w"))
            return
        path = out_json.replace(".json", "_dra.json")
        with open(model) as src, open(path, "w") as dst:
            dst.write(src.read())
        dra = load_smt_dra(path)
        log = r.stderr + r.stdout
        # On a timeout or crash, the model is the last intermediate hypothesis.
        if "TIMEOUT:" in log:
            res.update(status="timeout")
        elif "terminated abnormally" in log:
            res.update(status="crash")
        else:
            res.update(status="ok")
        res.update(learning_time=round(time.monotonic() - t0, 2),
                   states=dra.states, registers=len(dra.register_vars),
                   transitions=len(dra.transitions),
                   train_acc=round(100 * balanced_acc(dra, pos, neg), 2),
                   test_acc=round(100 * balanced_acc(dra, tpos, tneg), 2))
        json.dump(res, open(out_json, "w"))
        return
    else:
        from local_serach_synthesis import dra_learner
        lr = dra_learner.DRAlearner(num_states=lang.states, num_registers=len(lang.regs),
                                    constants=list(lang.ic), pos_samples=pos, neg_samples=neg,
                                    golden_pos_samples=[], auto=None, debug=False,
                                    exactly_one=(lang.edge_limit_type == "exactly"), theory=theory)
        lr.simulated_annealing(initial_temperature=1.0, cooling_rate=0.995,
                               max_iterations=50000, timeout=timeout)
        path = out_json.replace(".json", "_dra.json")
        lr.best_dra.translate_to_initial_zero().to_json(path, theory)
        dra = load_double_ls_dra(path) if theory == "Double" else load_integer_ls_dra(path)
    res.update(status="ok", learning_time=round(time.monotonic() - t0, 2),
               states=dra.states, registers=len(dra.register_vars),
               transitions=len(dra.transitions),
               train_acc=round(100 * balanced_acc(dra, pos, neg), 2),
               test_acc=round(100 * balanced_acc(dra, tpos, tneg), 2))
    json.dump(res, open(out_json, "w"))


def run_job(args):
    learner, language, seed, timeout, out = args
    name = f"{language}_{learner}"
    out_json = os.path.join(out, "runs", f"{name}.json")
    if os.path.exists(out_json):
        return name, "skipped"
    cmd = [sys.executable, os.path.abspath(__file__), "--one", learner, language,
           str(seed), str(timeout), out_json]
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONWARNINGS="ignore")
    t0 = time.monotonic()
    with open(os.path.join(out, "logs", f"{name}.log"), "w") as log:
        try:
            # Grace period on top of the learner's own budget, whose checks
            # only happen between solver calls.
            rc = subprocess.run(cmd, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout + 60).returncode
        except subprocess.TimeoutExpired:
            rc = "killed"
    if not os.path.exists(out_json):
        json.dump(dict(language=language, learner=learner, seed=seed,
                       status="timeout" if rc == "killed" else f"error (rc={rc})",
                       learning_time=round(time.monotonic() - t0, 2)), open(out_json, "w"))
    return name, json.load(open(out_json))["status"]


def merge(out):
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, "runs", "*.json")))
            if not p.endswith("_dra.json")]
    with open(os.path.join(out, "rq1_learners.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"merged {len(rows)} rows")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        _, _, learner, language, seed, timeout, out_json = sys.argv
        learn_one(learner, language, int(seed), float(timeout), out_json)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--languages", nargs="+", default=LANGUAGES)
    ap.add_argument("--learners", nargs="+", default=LEARNERS)
    ap.add_argument("--rescore", action="store_true",
                    help="only recompute accuracies of the saved DRAs, then merge")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if a.rescore:
        rescore(out)
        merge(out)
        return
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    jobs = [(l, lang, a.seed, a.timeout, out) for l in a.learners for lang in a.languages]
    with Pool(a.workers) as pool:
        for name, status in pool.imap_unordered(run_job, jobs):
            print(f"{name}: {status}", flush=True)
    merge(out)


if __name__ == "__main__":
    main()
