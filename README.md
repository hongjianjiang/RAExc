# Synthesis and Verification of Register Automata over Data Sequences

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

InterpreteRA (called **RAExc** in the paper) is the implementation of *Synthesis and Verification of Register Automata over Data Sequences*. It **learns** and **verifies** **Deterministic Register Automata (DRAs)** over ordered data sequences, i.e. over $(\mathbb{Q}; <, =)$ with rational guard constants:

- **Learning:** a local-search (simulated annealing) learner synthesises compact DRAs from labelled samples or from a queried classifier. Exact active learning and SMT-based learners are included as baselines.
- **Verification:** an exact robustness checker decides whether a DRA's classification is locally stable around a sequence $v$, i.e. whether $B_\delta(v) \cap L(\mathcal{A}^c) = \emptyset$. Perturbation costs (edit, Hamming, L1, DTW, ...) are given as register automata with accumulators. The check reduces to coverability and then to a shortest-path problem. It ranges over all of $\mathbb{Q}^+$, so counterexamples can have a different length from $v$.
- **Neural surrogates:** for **LSTM** and **Transformer** classifiers, a PAC validation loop learns a DRA surrogate, refines rejected candidates, and either returns a genuine non-robustness witness for the network or accepts the DRA with statistical guarantees for fidelity and local stability.

---

## Installation

```bash
pip3 install -r requirements.txt
```
 
You will also need to have Java (it was tested on Java version 24) and Maven installed.

To build the Java-based exact extractor:

```bash
cd src/active_learning_synthesis
mvn package assembly:single
```


## Learning & Extraction Evaluation

### 1. Single Experiment via `run_evaluation.py`

You can run a single evaluation using the command-line interface.
Run from root directory:
```bash
pypy3 src/learning_evaluation.py --language Sn --learner ls --n 10 --max_len 15
```

```bash
python3 src/robustness_evaluation.py   --language S1 --model lstm  --learner al --distance 1 --metric edit 
```

### CLI Options
| Option        | Description                                 | Example                   |
| ------------- | ------------------------------------------- | ------------------------- |
| `--language`  | Language name (e.g., S1, L3, S2)            | `--language S2`           |
| `--metric`    | Distance metric (last\_letter, edit, etc.)  | `--metric edit`           |
| `--distance`  | Radius δ; a witness needs cost **strictly below** δ (use 1.5 for "at most one edit" under Hamming/edit) | `--distance 1.0` |
| `--model`     | Type of neural model (transformer or lstm)  | `--model transformer`     |
| `--learner`   | Learner                                     | `--learner ls`            |
| `--timeout`   | Total local-search budget per configuration (s), shared by the rounds, default 1800 | `--timeout 1800` |
| `--check_timeout` | Total robustness-checking budget per configuration (s), default 3600 | `--check_timeout 3600` |
| `--max_loops` | Maximum learn/validate rounds, default 10 | `--max_loops 10` |
| `--p`, `--gamma`, `--epsilon` | PAC parameters (agreement threshold, tolerance, error), defaults 0.95/0.05/0.05 | `--p 0.95` |
| `--seed`      | Random seed | `--seed 0` |

With `--learner ls`, `src/pac_pipeline.py` runs the PAC-validated extraction
loop: the DRA is learned from samples labelled by the network (`--model`;
`language` uses the ground-truth language as oracle), validated on fresh
samples, and every agreeing sample is checked by the exact checker in
`src/register_automata/robustness.py`. Each configuration appends one row to
the CSV with its outcome (`robust` = accepted surrogate, `non-robust` = genuine
witness confirmed on the network, `inconclusive`, `timeout`), the number of
rounds, disagreements, spurious counterexamples, the witness and its cost, and
timings. Networks are loaded from `nn_models/{lstm,transformer}/<language>.pt`
relative to the working directory.

Tests for the checker: `python3 -m pytest src/tests -q`.

For multiple sequences, compile the DRA and metric once:

```python
from register_automata.metric_automata import create_edit_raa
from register_automata.robustness import RobustnessChecker

checker = RobustnessChecker(dra, create_edit_raa())
for v in vectors:
    result = checker.find_witness(v, delta=1.5, timeout=60)
```

The checker explores the product on demand, prunes paths at the strict radius,
and uses backward register liveness to merge configurations whose differences
cannot affect future guards. Returned witnesses are verified against the full
original automata. Rebuild the checker after changing either automaton.
`prune_registers=False` disables liveness for comparison. The PAC pipeline,
stress tests, and legacy `evaluate_robustness` entry point reuse compilation;
the legacy entry point now checks the DRA at the requested radius without
constructing a full graph or reducing the radius. Neural witness confirmation
is performed by the PAC pipeline.

Run `python3 scripts/benchmark_checker.py` to compare one-shot queries,
compiled reuse, and liveness on synthetic cases. Its CSV output includes
median batch time and expanded-state counts; `--reference /path/to/robustness.py`
uses an earlier checker snapshot as the one-shot baseline.


Full Available List:

languages = ["S1", "S2", "S3", "S4", "S5", "S6","S7","S8","S9","S10","S11","L1","L2","L3", "L4", "L5", "L6", "L7"]

metrics = ["last_letter", "edit", "hamming", "l1", "dtw"]

model = ["lstm", "transformer"]

learners = ["ls", "al", "smt"]

### Run in parallel

Use this to automatically run all combinations of languages and distance metrics in parallel.

```bash
python3 src/run_parallel.py --type check/learn
```


The script:
- Iterates over predefined languages and metrics
- Calls run_evaluation.py in a subprocess for each combination
- Saves results to separate CSV files under results/batch/

#### To Modify Behavior

- Edit the language and metric lists in run_parallel.py
- Set NUM_WORKERS to control parallelism
- Change default distance or CSV behavior directly in the script

## Learning LnA with Local Search

To learn the LnA language using Local Search, you must specify the number of states/registers with the `--n` argument. You can use the main evaluation script as follows:

```bash
python3 src/learning_evaluation.py --language LnA --learner ls --n <num_states> [--max_len <max_sequence_length>]
```

- `--n` (required for LnA): Number of states/registers for LnA language generation (e.g., 4)
- `--max_len` (optional): Maximum sequence length for sample generation (default: 10)

**Example:**

```bash
python3 src/learning_evaluation.py --language LnA --learner ls --n 4 --max_len 15
```

This will run the local search learner for LnA and save the results to the appropriate CSV file in the `results/` directory.

> **Note:** For other languages, the `--n` argument is not required.
