import sys
import os
import argparse
from learning_evaluation import run_ls_experiment
from benchmark_gen import new_generate_language

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Local Search learning for LnA.")
    parser.add_argument("--n", type=int, required=True, help="Number of states/registers for LnA language generation")
    parser.add_argument("--max_len", type=int, default=10, help="Maximum sequence length for LnA language generation")
    args = parser.parse_args()

    language = "LnA"
    # Generate language object for LnA with user-specified n
    lang = new_generate_language(language, args.n, 0, args.max_len)
    theory = "Integer"  # Change to 'Double' if needed for your use case
    csv_name = f"results/ls_learning_results_LnA.csv"

    run_ls_experiment(
        language=language,
        states=lang.states,
        edge_type=True if getattr(lang, 'edge_limit_type', 'exactly') == 'exactly' else False,
        regs=len(lang.regs),
        csv_path=csv_name,
        max_loops=10,  # You can adjust this
        timeout=120,   # You can adjust this
        silent=False,
        theory=theory,
        algo='sa',     # Local search algorithm: 'hc' (hill climbing), 'sa', etc.
        use_portfolio=False,
        n=args.n,
        max_len=args.max_len
    )
    print(f"Local search learning for LnA completed. Results saved to {csv_name}")
