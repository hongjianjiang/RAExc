import argparse
import subprocess
from multiprocessing import Pool
import os
import csv

# List of languages and metrics
languages = ['Sn'] 
metrics   =  ['last_letter']  
models    = ['language']  
learners  = ['ls']

OUTPUT_DIR = "results/run_parallel"
os.makedirs(OUTPUT_DIR, exist_ok=True)
NUM_WORKERS = 1 # Reduced from 8 to prevent memory exhaustion

def run_experiment_job(args):
    language, metric, model, lr = args
    cmd = [
        "python3", "src/robustness_evaluation.py",
        "--language", language,
        "--metric", metric,
        "--distance", "1",
        "--model", model,
        '--learner', lr,
        '--silent', "True",
        '--ls_alg', 'sa',
        '--n', "100", 
        '--max_len', "105"
    ]
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ Completed {language}-{metric} with {lr}")
    except subprocess.CalledProcessError as e:
        if e.returncode == -9:  # SIGKILL
            print(f"✗ Error processing {language}-{metric}: Process killed (likely out of memory)")
        else:
            print(f"✗ Error processing {language}-{metric}: {e}")
    except Exception as e:
        print(f"✗ Unexpected error processing {language}-{metric}: {e}")

def run_experiment_learn_job(args):
    language, model, lr = args
    out_csv = f"{OUTPUT_DIR}/total_{lr}_learn_record.csv"
    if language.startswith('S') or 'LStock' in language or 'mnist' in language:
        print(f"Using Double theory for {language}")
        theory = 'Double'
    else:
        print(f"Using Integer theory for {language}")
        theory = 'Integer'
        
    cmd = [
        "python3", "src/learning_evaluation.py",
        "--language", language,
        "--csv", out_csv,
        '--learner', lr,
        '--silent', "True",
        '--ls_alg', 'sa',
        '--theory', theory
    ]
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ Completed {language} with {lr}")
    except subprocess.CalledProcessError as e:
        if e.returncode == -9:  # SIGKILL
            print(f"✗ Error processing {language}: Process killed (likely out of memory)")
        else:
            print(f"✗ Error processing {language}: {e}")
    except Exception as e:
        print(f"✗ Unexpected error processing {language}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DRA robustness evaluation experiments.")
    parser.add_argument("--type", type=str, required=True, help="Learn or robustness evaluation type")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help=f"Number of parallel workers (default: {NUM_WORKERS})")
    args = parser.parse_args()
    
    # Override NUM_WORKERS if specified
    if args.workers:
        NUM_WORKERS = args.workers
        print(f"Using {NUM_WORKERS} parallel workers")
    if args.type == "check":
        for lr in learners:
            out_csv = f"results/run_parallel/total_{lr}_robust_record.csv"
            file_exists = os.path.isfile(out_csv)
            is_empty = not file_exists or os.path.getsize(out_csv) == 0
            with open(out_csv, "a", newline="") as f:
                writer = csv.writer(f)
                if is_empty:
                    if lr == 'ls':
                        writer.writerow([
                           "language", "automaton_size", "registers", "metric",
                            "distance", "epsilon", "gamma", "model", "learning_time", "interpret_time", "time_taken", 'learning_rate', "outcome"
                    ])
                    elif lr =='al':
                        writer.writerow([
                           "language", "automaton_size", "registers", "metric",
                            "distance", "epsilon", "gamma", "model", "learning_time", "interpret_time", "time_taken", 'learning_rate', "found_robust_dra",' olg_result'
                        ])
                    else:
                        writer.writerow([
                           "language", "automaton_size", "registers", "metric",
                            "distance", "epsilon", "gamma", "model", "learning_time", "interpret_time", "time_taken", 'learning_rate', "outcome"
                        ])
                        
        jobs = [(lang, metric, model, lr) for lang in languages for metric in metrics for model in models for lr in learners]
        with Pool(NUM_WORKERS) as pool:
            pool.map(run_experiment_job, jobs)
    elif args.type == "learn":
        for lr in learners:
            out_csv = f"results/run_parallel/total_{lr}_learn_record.csv"
            file_exists = os.path.isfile(out_csv)
            is_empty = not file_exists or os.path.getsize(out_csv) == 0
            with open(out_csv, "a", newline="") as f:
                writer = csv.writer(f)
                if is_empty:
                    if lr == 'ls':
                        writer.writerow([
                        "language", "automaton_size", "registers","epsilon", "gamma", "time_taken", "learning_rate", 'method'
                    ])
                    else:
                        writer.writerow([
                        "language", "automaton_size", "registers","epsilon", "gamma", "time_taken", "learning_rate"
                    ])
    
    
        jobs = [(lang, model, lr) for lang in languages for model in models for lr in learners]
        with Pool(NUM_WORKERS) as pool:
            pool.map(run_experiment_learn_job, jobs)
    else:
        raise ValueError("Invalid type. Use 'learn' or 'robustness'.")
    # jobs = [(lang, metric, model, lr) for lang in languages for metric in metrics for model in models for lr in learners]
    # with Pool(NUM_WORKERS) as pool:
    #     pool.map(run_experiment_job, jobs)
    print("\nAll experiments complete.")