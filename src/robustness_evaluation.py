from robustness_check import pac_sampling, evaluate_robustness, sample_robustness_with_gamma_epsilon
from local_serach_synthesis import dra_learner
from register_automata.ra import load_double_ls_dra, load_integer_ls_dra
from time import time
import argparse
from benchmark_gen import new_generate_language
import sys
import os
import tempfile
import csv
import smt_based_synthesis.smtlearner as smtlearner
import smt_based_synthesis.eqsmtlearner as eqsmtlearner
from typing import Dict, Set, Tuple, List, Union
from local_serach_synthesis.portfolio_runner import run_portfolio_parallel, run_portfolio_racing
import copy
from pac_pipeline import run_pac_experiment

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from robustness_check import *

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
        rnn={model}
        learner=sllambda
        queryRNNFileName= nn_models/{model}/query_rnn_{language}.log
        timeout=120
        language={language}
        """
    else:
        config_content = f"""delta = {gamma}
        epsilon = {epsilon}
        p = 0.01
        fixTraceLen = false
        datatype = integer
        modelFileName = {model_name}
        teachers=int:de.learnlib.ralib.tools.theories.IntegerEqualityTheory
        rnn={model}
        learner=sllambda
        queryRNNFileName= nn_models/{model}/query_rnn_{language}.log
        timeout=120
        language={language}
        """
    return config_content

def run_smt_experiment(
    language, metric, distance, states, regs, ic, uc, edge_limit_num, edge_limit_type,
    csv_path="results/ls_smt_robustness_results.csv",
    transformer_type="transformer", max_loops=10, timeout=300,
    gamma=0.05, epsilon=0.05, silent=False, theory='Integer'
):
    # Ensure CSV dir exists and header is present
    csv_dir = os.path.dirname(csv_path) or "."
    os.makedirs(csv_dir, exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    automata_learning_time = 0.0
    interpret_time = 0.0
    learning_rate = float("nan")

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "language","automaton_size","registers","metric","distance",
                "epsilon","gamma","model","learning_time","interpret_time","time_taken",
                "learning_rate","found_robust_dra"
            ])

        # Initial PAC sampling (kept from your code)
        samples = pac_sampling(gamma, epsilon, language, 15, model_type=transformer_type)
        pos = [s[:-1] for s in samples if s[-1]]
        neg = [s[:-1] for s in samples if not s[-1]]
        random.shuffle(pos)
        random.shuffle(neg)

        golden_pos = []
        loop_counter = 0
        start_time = time()

        auto = None
        last_score = float("nan")

        while loop_counter < max_loops:
            # Respect outer timeout budget for each loop
            remaining = timeout - (time() - start_time)
            if remaining <= 0:
                total_elapsed = time() - start_time
                regs_count = regs if isinstance(regs, int) else len(regs)
                writer.writerow([
                    language, getattr(auto, "states", 0), regs_count, f"{last_score:.6f}",
                    metric, distance, epsilon, gamma, transformer_type,
                    f"{automata_learning_time:.2f}", f"{interpret_time:.2f}", f"{total_elapsed:.2f}",
                    loop_counter, f"{learning_rate:.6f}" if isinstance(learning_rate, float) else learning_rate, "timeout"
                ])
                break

            # --- SMT learning phase ---
            t_learn = time()
            # Prepend any golden CE to positives
            pos_for_learn = (golden_pos + pos) if golden_pos else pos
            auto_arg = auto  # optional warm start if implemented

            if theory == "Integer":
                learner = eqsmtlearner.Register_Learner(
                    n=states, regs=regs, ic=ic, uc=uc,
                    pos=pos_for_learn, neg=neg,
                    edge_limit_type=edge_limit_type, edge_limit_num=edge_limit_num,
                    debug=False, batch_size=10, target=1 - epsilon,
                )
            else:
                learner = smtlearner.Register_Learner(
                    n=states, regs=regs, ic=ic, uc=uc,
                    pos=pos_for_learn, neg=neg,
                    edge_limit_type=edge_limit_type, edge_limit_num=edge_limit_num,
                    debug=False, auto=auto_arg, batch_size=8, strategy="edge", target=1 - epsilon,
                )
            print(f'=== SMT-based robustness check {language} with model: {transformer_type}, Starting ===')
            auto, epos = learner.get_model()  # auto may be "unsat" / "timeout" / model
            automata_learning_time += (time() - t_learn)
            os.makedirs("smt_dra_json", exist_ok=True)
            json_file = f"smt_dra_json/ra_{language}_{metric}_loop{loop_counter+1}.json"

            # Handle outcomes of learning
            if auto == "unsat":
                auto = None
                print(f"Unsat and restarting learner of {language} with {metric}.")
                loop_counter += 1
                continue
            elif auto == "timeout":
                # treat as timeout for this loop, write and exit
                total_elapsed = time() - start_time
                regs_count = regs if isinstance(regs, int) else len(regs)
                writer.writerow([
                    language, 0, regs_count,
                    metric, distance, epsilon, gamma, transformer_type,
                    f"{automata_learning_time:.2f}", f"{interpret_time:.2f}", f"{total_elapsed:.2f}",
                     f"{float('nan'):.6f}", "timeout"
                ])
                break
            else:
                auto.save_to_json(json_file)
                auto = load_smt_dra(json_file)

            # Optional consistency/score
            try:
                learning_rate = learner.consistence_check(auto, pos_for_learn, neg)
            except Exception:
                learning_rate = float("nan")

            # If you maintain a score on the learner, put it here; else infer from learning_rate
            last_score = 1.0 if (isinstance(learning_rate, float) and learning_rate >= (1 - epsilon)) else learning_rate

            if isinstance(learning_rate, float) and learning_rate < (1 - epsilon):
                # print(f"SMT learner learns {language} with acc {learning_rate:.4f} < {1-epsilon:.4f}, restarting with more samples.")
                loop_counter += 1
                continue
             

            # --- Robustness sampling ---
            pos_eval, neg_eval = sample_robustness_with_gamma_epsilon(gamma, epsilon, language)
            numerator = math.log(2 / epsilon)
            denominator = 2 * (gamma ** 2)
            num_needed = math.ceil(numerator / denominator)
            half = 2

            # Pre-filter by auto/complement acceptance
            comp_auto = auto.complement()
            new_pos_samples, new_neg_samples = [], []
            for i in pos_eval:
                if len(new_pos_samples) >= half: break
                if auto.run(i): new_pos_samples.append(i)
            for i in neg_eval:
                if len(new_neg_samples) >= half: break
                if comp_auto.run(i): new_neg_samples.append(i)

            # print(f'----------------------SMT Robustness Check {language} {loop_counter + 1}----------------------')

            # === EVALUATE 1: auto on positives ===
            t_pos = time()
            flag_pos, cex_pos, tout_pos, delta_pos, _ = evaluate_robustness(
                auto, distance, 1, metric, language, new_pos_samples,
                model_type=transformer_type, search_adversarial=True, local_search=True, silent=silent
            )
            interpret_time += (time() - t_pos)

            total_elapsed = time() - start_time
            # if tout_pos or total_elapsed >= timeout:
            #     regs_count = regs if isinstance(regs, int) else len(regs)
            #     writer.writerow([
            #         language, getattr(auto, "states", 0), regs_count, 
            #         metric, distance, epsilon, gamma, transformer_type,
            #         f"{automata_learning_time:.2f}", f"{interpret_time:.2f}", f"{total_elapsed:.2f}",
            #          f"{(learning_rate if isinstance(learning_rate, float) else float('nan')):.6f}", "timeout"
            #     ])
            #     break

            # Early fail on positives
            if (delta_pos <= 0) or (not flag_pos):
                regs_count = regs if isinstance(regs, int) else len(regs)
                writer.writerow([
                    language, getattr(auto, "states", 0), regs_count, 
                    metric, distance, epsilon, gamma, transformer_type,
                    f"{automata_learning_time:.2f}", f"{interpret_time:.2f}", f"{total_elapsed:.2f}",
                     f"{(learning_rate if isinstance(learning_rate, float) else float('nan')):.6f}", "not robust"
                ])
                break

            # Restart if we found a counterexample on positives
            if cex_pos:
                golden_pos = [cex_pos]
                loop_counter += 1
                print(f"Restarting learner with golden_pos for {language} in {metric}.")
                continue

            # === EVALUATE 2: complement on negatives ===
            t_neg = time()
            flag_neg, cex_neg, tout_neg, delta_neg, _ = evaluate_robustness(
                comp_auto, distance, 1, metric, language, new_neg_samples,
                model_type=transformer_type, search_adversarial=True, local_search=True, silent=silent
            )
            interpret_time += (time() - t_neg)

            total_elapsed = time() - start_time 
            # Early fail on negatives
            if (delta_neg <= 0) or (not flag_neg):
                regs_count = regs if isinstance(regs, int) else len(regs)
                writer.writerow([
                    language, getattr(auto, "states", 0), regs_count, 
                    metric, distance, epsilon, gamma, transformer_type,
                    f"{automata_learning_time:.2f}", f"{interpret_time:.2f}", f"{total_elapsed:.2f}",
                     f"{(learning_rate if isinstance(learning_rate, float) else float('nan')):.6f}", "not robust"
                ])
                break

            # Restart if CE found on negatives
            if cex_neg:
                golden_pos = [cex_neg]
                loop_counter += 1
                print(f"Restarting learner with golden_pos (from negatives) for {language} in {metric}.")
                continue

            print(f'--------------------End SMT Robustness Check {language} --------------------')

            # If we got here: passed both evaluations → robust
            regs_count = regs if isinstance(regs, int) else len(regs)
            writer.writerow([
                language, getattr(auto, "states", 0), regs_count, 
                metric, distance, epsilon, gamma, transformer_type,
                f"{automata_learning_time:.2f}", f"{interpret_time:.2f}", f"{(time() - start_time):.2f}",
                 f"{(learning_rate if isinstance(learning_rate, float) else float('nan')):.6f}", "robust"
            ])
            break

        else:
            # max_loops reached without decisive result
            total_elapsed = time() - start_time
            regs_count = regs if isinstance(regs, int) else len(regs)
            writer.writerow([
                language, getattr(auto, "states", 0), regs_count, 
                metric, distance, epsilon, gamma, transformer_type,
                f"{automata_learning_time:.2f}", f"{interpret_time:.2f}", f"{total_elapsed:.2f}",
                f"{(learning_rate if isinstance(learning_rate, float) else float('nan')):.6f}", "inconclusive"
            ])
            print(f"Reached max loops ({max_loops}) without adversarial sample.")

        print(f'=== End SMT-based robustness check {language} with model: {transformer_type} ===')

def run_al_experiment(language, distance, inc, metric, model, gamma, epsilon,
                      silent=False, csv_path="results/active_learning_robustness_results.csv",
                      max_loop=10, theory='Integer'):
    import os, csv, math, tempfile
    from time import time

    # Per-language accumulators (what you actually write to CSV)
    language_learning_time = 0.0
    language_robustness_time = 0.0
 
    loop = 0
    result = "unknown"
    p_value = None
    auto_states = "N/A"
    auto_registers = "N/A"

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "language","automaton_size","registers","metric","distance",
                "epsilon","gamma","model","learning_time","interpret_time","time_taken",
                "learning_rate","found_robust_dra"
            ])

        config_content = get_config_content(gamma, epsilon, language, model, theory)
        temp_config_path = None

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.config', delete=False) as temp_config:
                temp_config.write(config_content)
                temp_config_path = temp_config.name

            while loop < max_loop:
                print(f"=== Active Learning-based robustness check {language}, Metric: {metric}, Distance: {distance}, Model: {model} ===")

                # ---- LEARNING (measure every run_rnn_with_java) ----
                t0 = time()
                flag, ogc, p_value = run_rnn_with_java(temp_config_path, max_retries=5)
                language_learning_time += (time() - t0)

                if not flag:
                    print("Found oligomorphic sample:", ogc)
                    result = "unknown"
                    break

                if p_value < 1 - epsilon:
                    print(f'Active learner fails to learn {language} up to {1-epsilon}, current: {p_value}, restart....')
                    result = "unknown"
                    loop += 1
                    continue

                # Load current automata once learning succeeded
                auto = load_smt_dra(f"active_models/{language}.xml")
                comp_auto = auto.complement()
                auto_states = getattr(auto, "states", "N/A")
                auto_registers = len(getattr(auto, "register_vars", []))

                # Prepare samples
                pos, neg = sample_robustness_with_gamma_epsilon(gamma, epsilon, language)
                numerator = math.log(2 / epsilon)
                denominator = 2 * gamma ** 2
                num = math.ceil(numerator / denominator)
                new_pos_samples, new_neg_samples = [], []
                count = 0
                for i in pos:
                    if count >= num // 2: break
                    if auto.run(i) is True:
                        new_pos_samples.append(i); count += 1

                count = 0
                for i in neg:
                    if count >= num // 2: break
                    if comp_auto.run(i) is True:
                        new_neg_samples.append(i); count += 1

                # ---- ROBUSTNESS (measure every evaluate_robustness) ----
                t0 = time()
                # print('----------------starting robustness check comp auto------------------')

                flag, adv, timeout, found_delta, inx = evaluate_robustness(
                    auto, distance, inc, metric, language, new_pos_samples,
                    model_type=model, search_adversarial=True, local_search=True, silent=silent
                )
                # include time for the first robustness call
                language_robustness_time += (time() - t0)

                if not flag and ogc:
                    print(f"Failed distance to robustness: {found_delta}")
                    global_res = 'Yes'
                    result = 'unknown'
                    break
                elif not flag:
                    result = 'not robust'
                    break
                elif (found_delta <= 0):
                    result = 'non robust'
                    break

                t0 = time()
                flag1, adv1, timeout1, found_delta1, inx1 = evaluate_robustness(
                    comp_auto, distance, inc, metric, language, new_neg_samples,
                    model_type=model, search_adversarial=True, local_search=True, silent=silent, reverse=True
                )
                language_robustness_time += (time() - t0)

                print('flag', flag, adv)
                if timeout or timeout1:
                    result = 'timeout'
                    break 

                if flag or flag1:
                    found_robust_dra = True
                    result = "robust"
                elif (not flag or not flag1) and ogc:
                    print(f"Failed distance to robustness: {found_delta if 'found_delta' in locals() else 'N/A'}")
                    global_res = 'Yes'
                    result = 'unknown'
                elif not (flag or flag1):
                    result = 'not robust'
                elif ('found_delta' in locals() and found_delta <= 0) or ('found_delta1' in locals() and found_delta1 <= 0):
                    result = 'non robust'
                else:
                    result = 'robust'
                break  # end while loop

        except Exception as e:
            print(f"✗ Error processing {language}: {e}")
        finally:
            try:
                if temp_config_path: os.unlink(temp_config_path)
            except:
                pass

        learning_rate_percent = f"{p_value * 100:.2f}%" if p_value is not None else "N/A"
        total_time = language_learning_time + language_robustness_time

        writer.writerow([
            language, auto_states, auto_registers, metric, distance, epsilon, gamma, model,
            f"{language_learning_time:.2f}", f"{language_robustness_time:.2f}", f"{total_time:.2f}",
            learning_rate_percent, result
        ])
        print(f'=== End AL Robustness Check {language} with {language}, Metric: {metric}, Distance: {distance}, Model: {model} ===')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DRA robustness evaluation experiments.")
    parser.add_argument("--language", type=str, required=True, help="Language to evaluate (e.g., S1)")
    parser.add_argument("--metric", type=str, required=True, help="Distance metric (e.g., last_letter, edit, dtw)")
    parser.add_argument("--distance", type=float, default=1.0, help="Distance threshold for robustness evaluation")
    parser.add_argument("--model", type=str, default="transformer", help="Model type (transformer or lstm)")
    parser.add_argument("--max_loops", type=int, default=10, help="Maximum number of learn/validate rounds")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Total local-search budget per configuration, in seconds (shared by the rounds)")
    parser.add_argument("--check_timeout", type=float, default=3600.0, help="Total robustness-checking budget per configuration, in seconds")
    parser.add_argument("--p", type=float, default=0.95, help="Target agreement threshold p")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--csv", type=str, default=None, help="CSV path for the selected learner's results")
    parser.add_argument("--learner", type=str, default="ls", help="Learner type (e.g., ls, al)")
    parser.add_argument("--gamma", type=float, default=0.05, help="Gamma parameter for the learner")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Epsilon parameter for the learner")
    parser.add_argument("--silent", type=bool, default=False, help="Silence output")
    parser.add_argument("--theory", type=str, default="Integer", help="Arithmetic theory (e.g., Integer Equality, Double Equality)")
    parser.add_argument("--ls_alg", type=str, default="sa", help="Local Search algorithm (e.g., sa, hc)")
    parser.add_argument("--use_portfolio", action="store_true")
    parser.add_argument("--portfolio_mode", type=str, default="parallel", choices=["parallel", "racing"])
    parser.add_argument("--per_algo_timeout", type=float, default=240.0)
    parser.add_argument("--racing_total_budget", type=float, default=240.0)
    parser.add_argument("--racing_time_slice", type=float, default=5.0)
    parser.add_argument("--n", type=int, default=None, help="Number of states/registers for LnA (required for LnA)")
    parser.add_argument("--max_len", type=int, default=10, help="Maximum sequence length for LnA (optional, default 10)")

    args = parser.parse_args()
    lang = new_generate_language(args.language, 0, 0, 0)
    if 'S' in args.language or args.language == 'mnist':
        theory = 'Double'
    else:
        theory = "Integer"
    if args.language == "LnA" or args.language =='Sn':
        if args.n is None:
            parser.error("--n is required when --language LnA")
        lang = new_generate_language(args.language, 0, 0, args.max_len, n=args.n)
    else:
        lang = new_generate_language(args.language, 0, 0, args.max_len)
    csv_name = args.csv or f'results/{args.learner}_robustness_results.csv'
    if args.learner == "ls":
        run_pac_experiment(
            language=args.language,
            metric=args.metric,
            delta=args.distance,
            model_type=args.model,
            states=lang.states,
            regs=len(lang.regs),
            constants=list(lang.ic),
            edge_type=(lang.edge_limit_type == 'exactly'),
            theory=theory,
            csv_path=csv_name,
            max_loops=args.max_loops,
            learn_timeout=args.timeout,
            check_timeout=args.check_timeout,
            p=args.p,
            gamma=args.gamma,
            epsilon=args.epsilon,
            max_len=args.max_len,
            n_param=args.n if args.language in ("LnA", "Sn") else None,
            seed=args.seed,
            silent=args.silent,
        )
    elif args.learner == "smt":
        run_smt_experiment(
            language=args.language,
            states= lang.states,
            ic = lang.ic, 
            uc = lang.uc,
            metric=args.metric,
            distance=args.distance,
            edge_limit_num = lang.edge_limit_num,
            edge_limit_type=lang.edge_limit_type,
            regs=lang.regs,
            csv_path=csv_name,
            transformer_type=args.model,
            max_loops=args.max_loops,
            timeout=args.timeout,
            gamma=args.gamma,
            epsilon=args.epsilon,
            silent=args.silent,
            theory=theory
        )
    else:
        run_al_experiment(
            language=args.language,
            distance=args.distance,
            inc = 1,
            metric=args.metric,
            model=args.model,
            gamma=args.gamma,
            epsilon=args.epsilon,
            silent=args.silent,
            csv_path=csv_name,
            max_loop=args.max_loops,
            theory = theory,
            
        )
