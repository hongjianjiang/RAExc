from local_serach_synthesis import dra_learner
from register_automata.ra import load_double_ls_dra, load_integer_ls_dra, load_smt_dra
from time import time
import argparse
from benchmark_gen import new_generate_language
import sys
import os
import tempfile
import csv
from typing import Dict, Set, Tuple, List, Union
from local_serach_synthesis.portfolio_runner import run_portfolio_parallel, run_portfolio_racing
import copy
from benchmark_gen import random_sample_generation, membership_query
import subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


            
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


def get_config_content(gamma, epsilon, language, model, theory):
    model_name = f"nn_models/{model}/{language}.pt"
    if theory == 'Double':
        config_content = f"""delta = {gamma}
        epsilon = {epsilon}
        p = 0.01
        fixTraceLen = false
        datatype = double
        modelFileName = {model_name}
        teachers=double:de.learnlib.ralib.tools.theories.DoubleInequalityTheory
        rnn=language
        learner=sllambda
        queryRNNFileName= nn_models/{model}/query_rnn_{language}.log
        timeout=60
        language={language}
        maxLen=15
        """
    else:
        config_content = f"""delta = {gamma}
        epsilon = {epsilon}
        p = 0.01
        fixTraceLen = false
        datatype = integer
        modelFileName = {model_name}
        teachers=int:de.learnlib.ralib.tools.theories.IntegerEqualityTheory
        rnn=language
        learner=sllambda
        queryRNNFileName= nn_models/{model}/query_rnn_{language}.log
        timeout=60
        language={language}
        maxLen=15
        """
    return config_content

def run_ls_experiment(
    language, states, regs, edge_type,
    csv_path="results/local_search_robustness_results.csv", max_loops=10, timeout=120,
    gamma=0.05, epsilon=0.05, silent=False, theory='Integer', algo='hc',
    # --- NEW: portfolio options ---
    use_portfolio: bool = False,
    portfolio_mode: str = "parallel",  # "parallel" or "racing"
    portfolio_algos: List[str] = None,
    per_algo_timeout: float = 120,   # used by parallel
    racing_total_budget: float = 120,# used by racing
    racing_time_slice: float = 5.0,    # used by racing
    portfolio_algo_args: Dict[str, dict] = None,
    portfolio_seed: int = 123,
    n: int = None,
    max_len: int = 10
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    automata_learning_time = 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "language", "automaton_size", "registers", "num_registers", "num_transitions", "learning_time", "learning_rate"
            ])

        print(f"=== Local search for Language: {language}===") 
        # Use n for LnA if provided
        if (language == "LnA" or language == "Sn") and n is not None:
            pos, neg = sample_robustness_with_gamma_epsilon_1(gamma, epsilon, language, n=n, max_len=max_len, sample_count=369)
        else:
            print("Max:", max_len)
            pos, neg = sample_robustness_with_gamma_epsilon_1(gamma, epsilon, language, max_len=max_len, sample_count=369)
        print(pos[:5])
        golden_pos = []
        best_auto = None
        loop_counter = 0
        start_time = time()
        # samples = sample_robustness_with_gamma_epsilon(gamma, epsilon, language)
        while loop_counter < max_loops:
            print(f"--- Loop {loop_counter+1}/{max_loops} ---")
            # Reset budgets for each round to avoid stale timeouts (fixes ACO/Tabu/LNS bug)
            num_iterations = 10000
            time_out = timeout
            learn_start_time = time()
            learner = dra_learner.DRAlearner(
                num_states=states,
                num_registers=regs,
                constants=[],
                pos_samples=pos,
                neg_samples=neg,
                golden_pos_samples=golden_pos,
                auto=best_auto,
                debug=False,
                exactly_one=edge_type,
                theory=theory
            )

            # --- NEW: portfolio switch ---
            if use_portfolio:
                # sensible defaults if not provided
                if portfolio_algos is None:
                    portfolio_algos = ["hill_climbing", "sim_anneal", "aco", "pso", "ga", "lns", "tabu"]
                if portfolio_algo_args is None:
                    portfolio_algo_args = {
                        "hill_climbing": {"restart_iterations": 5000},
                        "sim_anneal": {"initial_temperature": 1.0, "cooling_rate": 0.995, "max_iterations": 20000},
                        "aco": {"n_ants": 12, "n_iterations": 3000, "evaporation_rate": 0.6, "alpha": 1.0, "beta": 2.0},
                        "pso": {"swarm_size": 16, "n_iterations": 4000},
                        "ga": {"population_size": 32, "generations": 4000, "mutation_rate": 0.35},
                        "lns": {"n_iterations": 4000, "destruction_size": 3},
                        "tabu": {"n_iterations": 4000, "tabu_size": 100},
                    }
                if portfolio_mode == "parallel":
                    summary = run_portfolio_parallel(
                        learner,
                        per_algo_timeout=min(per_algo_timeout, timeout),  # respect outer budget
                        algos=portfolio_algos,
                        algo_args=portfolio_algo_args,
                        base_seed=portfolio_seed,
                        start_method="spawn",
                    )
                else:
                    summary = run_portfolio_racing(
                        learner,
                        total_budget=min(racing_total_budget, timeout),
                        time_slice=racing_time_slice,
                        algos=portfolio_algos,
                        algo_args=portfolio_algo_args,
                        base_seed=portfolio_seed,
                    ) 
                learner.load_dra_state(summary["best_state"])
                learner.best_score = summary["best_score"]
                learner.best_dra = copy.deepcopy(learner.mydra)
                print(f"[Portfolio] best={summary['best_score']:.6f} via {summary['best_algo']}")
                best_auto = learner.best_dra.translate_to_initial_zero()
                os.makedirs("ls_dra_json", exist_ok=True)
                json_file = f"ls_dra_json/ra_{language}_loop{loop_counter+1}.json"
                best_auto.to_json(json_file, theory)
                if theory == "Double":
                    auto = load_double_ls_dra(json_file)
                else:
                    auto = load_integer_ls_dra(json_file)
                best_auto = learner.best_dra.translate_to_initial_zero() 
            else:
                # print('using single ls algorithm:', algo)
                # --- ORIGINAL single-algo path (kept unchanged) ---
                num_iterations = 50000
                time_out = timeout
                print(f"Running {algo} for up to {num_iterations} iterations and {time_out} seconds...")
                if algo == 'sa':
                    learner.simulated_annealing(
                        initial_temperature=1.0,
                        cooling_rate=0.995,
                        max_iterations=num_iterations,
                        timeout=time_out,
                    )
                elif algo == 'hc':
                    learner.hill_climbing(timeout=time_out, restart_iterations=num_iterations)
                elif algo == 'ant':
                    learner.ant_colony_optimization(
                        n_ants=10, n_iterations=1000, alpha=1.0, beta=2.0,
                        evaporation_rate=0.5, timeout=time_out
                    )
                elif algo == "lns":
                    learner.large_neighborhood_search(n_iterations=10000, destruction_size=3, timeout=time_out)
                elif algo == 'pso':
                    learner.particle_swarm_optimization(swarm_size=10, n_iterations=1000, timeout=time_out)
                elif algo == 'genetic':
                    learner.genetic_algorithm(population_size=10, generations=1000, mutation_rate=0.3, timeout=time_out)
                elif algo == 'tabu':
                    learner.tabu_search(n_iterations=1000, tabu_size=50, timeout=time_out)

                if not silent and algo == 'hc':
                    stats = learner.get_search_stats()
                    if stats:
                        print(
                            "[hc] "
                            f"proposals={stats['proposals']} "
                            f"accepted={stats['accepted']} "
                            f"restarts={stats['restarts']} "
                            f"rescored_samples={stats['rescored_samples']} "
                            f"avg_rescored={stats['avg_rescored_samples']:.1f} "
                            f"active_states={stats.get('active_states', 'n/a')} "
                            f"scoring={stats['timing']['scoring']:.2f}s "
                            f"mutation={stats['timing']['mutation']:.2f}s"
                        )

                # === Continue exactly as you already do ===
                best_auto = learner.best_dra.translate_to_initial_zero()
                os.makedirs("ls_dra_json", exist_ok=True)
                json_file = f"ls_dra_json/LS_{language}_loop{loop_counter+1}_learning.json"
                best_auto.to_json(json_file, theory)
                if theory == "Double":
                    auto = load_double_ls_dra(json_file)
                    auto.show("Sn")
                else:
                    auto = load_integer_ls_dra(json_file)

            if learner.best_score < 0.95:
                print(f'learner fails to learn the {language} upto {1-epsilon}, now is {learner.best_score}, restart')
                loop_counter += 1
                # You may want to bump budgets only when NOT in portfolio mode:
                if not use_portfolio:
                    num_iterations += 1000
                    time_out += 60
                continue
            else:
                used_learning_time = time() - learn_start_time
                automata_learning_time  += used_learning_time
                num_transitions = len(auto.transitions)
                num_registers = len(auto.register_vars)
                writer.writerow([
                    language,
                    auto.states,
                    regs,
                    num_registers,
                    num_transitions,
                    f"{time() - start_time:.2f}",
                    learner.best_score,
                ])
                print(f"Learned automaton for {language}: {auto.states} states, {num_registers} registers, {num_transitions} transitions. accuracy: {learner.best_score:.4f}, learning_time: {used_learning_time:.2f}s")
                break
        else:
            num_transitions = len(auto.transitions)
            num_registers = len(auto.register_vars)
            writer.writerow([
                language,
                auto.states,
                regs,
                num_registers,
                num_transitions,
                epsilon,
                gamma, 
                f"{automata_learning_time:.2f}",
                learner.best_score,
            ])
            print(f"Learned automaton for {language}: {auto.states} states, {num_registers} registers, {num_transitions} transitions.")
            print(f"Reached max loops ({max_loops}) without adversarial sample.")
        print(f'===      End LS Learning      {language} ===')

def run_smt_experiment(language, states, regs, ic, uc, edge_limit_type, edge_limit_num, csv_path="results/local_search_robustness_results.csv", gamma=0.05, epsilon=0.05, silent=False, theory='Integer'):
    import smt_based_synthesis.smtlearner as smtlearner
    import smt_based_synthesis.eqsmtlearner as eqsmtlearner

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    is_empty = not file_exists or os.path.getsize(csv_path) == 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_empty:
            writer.writerow([
                "language", "automaton_size", "registers", "num_registers", "num_transitions", "learning_time", "learning_rate"
            ])
        print(f"=== SMT learner for Language: {language} ===")
        start_time = time() 
        auto = None
        # while loop_counter < max_loops:
        pos, neg = sample_robustness_with_gamma_epsilon_1(gamma, epsilon, language)
        golden_pos = []
        pos = golden_pos + pos
        auto_arg = auto if not isinstance(auto, str) else None
        if theory == "Integer":
            print("Using Integer SMT Learner")
            learner = eqsmtlearner.Register_Learner(
            n=states,
            regs=regs,
            ic=ic,
            uc=uc,
            pos=pos,
            neg=neg,
            edge_limit_type=edge_limit_type,
            edge_limit_num=edge_limit_num,
            debug=False,
            batch_size=15,
            target = 1 - epsilon,
        )
        else:
            learner = smtlearner.Register_Learner(
                n=states,
                regs=regs,
                ic=ic,
                uc=uc,
                pos=pos,
                neg=neg,
                edge_limit_type=edge_limit_type,
                edge_limit_num=edge_limit_num,
                debug=False,
                auto=auto_arg,
                batch_size=5,
                strategy="edge",  # state or edge
                target = 1 - epsilon,
            )
        auto, epos = learner.get_model()
        if "unsat" == auto or 'timeout' == auto:
            auto = None
            learning_rate = None
            writer.writerow([
                language,
                'None',
                'None',
                'None',
                'None',
                f"{time() - start_time:.2f}",
                learning_rate,
            ])
        else:
            os.makedirs("smt_dra_json", exist_ok=True)
            json_file = f"smt_dra_json/smt_{language}_learning.json"
            auto.save_to_json(json_file)
            auto = load_smt_dra(json_file)
            learning_rate = learner.consistence_check(auto, pos, neg)
            
            num_transitions = len(auto.transitions)
            num_registers = len(auto.register_vars)
            if learning_rate >= 1 - epsilon:
                writer.writerow([
                    language,
                    auto.states,
                    len(regs),
                    num_registers,
                    num_transitions,
                    f"{time() - start_time:.2f}",
                    learning_rate,
                ])
                print(f"Learned automaton for {language}: {auto.states} states, {num_registers} registers, {num_transitions} transitions.")
            else:
                print(f"SMT {language}:{learning_rate}, and restart with more test samples")
                writer.writerow([
                    language,
                    auto.states if auto else "unknown",
                    len(regs) if auto else 'unknown',
                    num_registers if auto else 'unknown',
                    num_transitions if auto else 'unknown',
                    f"{time() - start_time:.2f}",
                    0,
                ])
                if auto:
                    print(f"Learned automaton for {language}: {auto.states} states, {num_registers} registers, {num_transitions} transitions.")
        print(f'===      End SMT Learning     {language} ===')

def run_al_experiment(language, silent=False, csv_path = "results/active_learning_robustness_results.csv", max_loop = 10, theory = 'Integer'):
    total_learning_time = 0
    flag = False
    loop = 0
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "language", "automaton_size", "registers", "num_registers", "num_transitions", "learning_time", "learning_rate"
            ])
        language_learning_time = 0
        language_robustness_time = 0

        config_content = get_config_content(0.05, 0.05, language, 'lstm', theory)
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.config', delete=False) as temp_config:
                temp_config.write(config_content)
                temp_config_path = temp_config.name
            
            learning_time = time()
            while loop < max_loop:
                print(f"=== Active Learning for Language: {language} ===")
                flag, ogc, p_value = run_rnn_with_java(temp_config_path, max_retries=5)
                language_learning_time += time() - learning_time
                total_learning_time += time() - learning_time
                if not flag:
                    print("Found oligomorphic sample:", ogc)
                    break
                    # found_delta = None
                else:
                    # samples = sample_robustness_with_gamma_epsilon(gamma, epsilon=epsilon, language=language)
                    auto = load_smt_dra(f"active_models/{language}.xml")
                    if p_value is None or p_value < 1-0.05:
                        # print(f'Active learner fails to learn a {language} up to {1-0.05}, current: {p_value}, restart....')
                        result = "unknown"
                        loop += 1
                        continue
                    else:
                        # print(f'Active learner learns a DRA up to {p_value}')
                        break
                 
            learning_rate_percent = f"{p_value * 100:.2f}%" if p_value is not None else "N/A"
            total_time = language_learning_time + language_robustness_time
            num_transitions = len(auto.transitions)
            num_registers = len(auto.register_vars)
            writer.writerow([language, auto.states, len(auto.register_vars), num_registers, num_transitions, f"{total_time:.2f}", learning_rate_percent])
            print(f"Learned automaton for {language}: {auto.states} states, {num_registers} registers, {num_transitions} transitions.")
            print(f'===        End AL Learning        {language} ===')
        except Exception as e:
            print(f"✗ Error processing {language}: {e}")
        finally:
            try:
                os.unlink(temp_config_path)
            except:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DRA robustness evaluation experiments.")
    parser.add_argument("--language", type=str, required=True, help="Language to evaluate (e.g., S1)")
    parser.add_argument("--csv", type=str, default="results/local_search_learning_results.csv", help="Path to output CSV file")
    parser.add_argument("--max_loops", type=int, default=2, help="Maximum number of refinement loops")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")
    parser.add_argument("--learner", type=str, default="ls", help="Learner type (e.g., ls, al)")
    parser.add_argument("--silent", type=bool, default=False, help="Silence output")
    parser.add_argument("--theory", type=str, default="Integer", help="Arithmetic theory (e.g., Integer Equality, Double Equality)")
    parser.add_argument("--ls_alg", type=str, default="sa", help="Local Search algorithm (e.g., sa, hc)")
    parser.add_argument("--states", type=int, default=None, help="Override the default number of automaton states for the chosen language")
    parser.add_argument("--use_portfolio", action="store_true")
    parser.add_argument("--portfolio_mode", type=str, default="parallel", choices=["parallel", "racing"])
    parser.add_argument("--per_algo_timeout", type=float, default=120)
    parser.add_argument("--racing_total_budget", type=float, default=120)
    parser.add_argument("--racing_time_slice", type=float, default=5.0)
    parser.add_argument("--n", type=int, default=None, help="Number of states/registers for LnA (required for LnA)")
    parser.add_argument("--max_len", type=int, default=10, help="Maximum sequence length for LnA (optional, default 10)")

    args = parser.parse_args()
    # Handle LnA special case
    if args.language == "LnA" or args.language =='Sn':
        if args.n is None:
            parser.error("--n is required when --language LnA")
        lang = new_generate_language(args.language, 0, 0, args.max_len, n=args.n)
    else:
        lang = new_generate_language(args.language, 0, 0, args.max_len)

    if 'S' in args.language or 'mnist' in args.language:
        theory = 'Double'
    else:
        theory = "Integer"

    csv_name = f'results/{args.learner}_learning_results.csv'
    if args.learner == "ls":
        run_ls_experiment(
            language=args.language,
            states=lang.states,
            edge_type=True if lang.edge_limit_type=='exactly' else False,
            regs=len(lang.regs),
            csv_path=csv_name,
            max_loops=args.max_loops,
            timeout=args.timeout,
            silent=args.silent,
            theory=theory,
            algo=args.ls_alg,
            use_portfolio=args.use_portfolio,
            portfolio_mode=args.portfolio_mode,
            per_algo_timeout=args.per_algo_timeout,
            racing_total_budget=args.racing_total_budget,
            racing_time_slice=args.racing_time_slice,
            n=args.n if (args.language == "LnA" or args.language == "Sn") else None,
            max_len=args.max_len
        )
    elif args.learner == "smt":
        run_smt_experiment(
            language=args.language,
            states=lang.states,
            ic=lang.ic, 
            uc=lang.uc,
            edge_limit_num=lang.edge_limit_num,
            edge_limit_type=lang.edge_limit_type,
            regs=lang.regs,
            csv_path=csv_name,
            silent=args.silent,
            theory=theory
        )
    else:
        run_al_experiment(
            language=args.language,
            silent=args.silent,
            csv_path=csv_name,
            max_loop=args.max_loops,
            theory=theory
        )
