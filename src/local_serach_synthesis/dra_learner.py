from typing import Dict, Set, Tuple, List, Union
from enum import Enum
from local_serach_synthesis.utils import *
import math
from itertools import chain, combinations
from local_serach_synthesis.register_automata import *
import csv
import ast
import time
import copy
from collections import Counter, defaultdict

class DRAlearner:    
    def __init__(self, 
                 num_states: int, 
                 num_registers: int, 
                 constants: List[float], 
                 pos_samples: List[List[float]], 
                 neg_samples: List[List[float]],
                 golden_pos_samples: List[List[float]],
                 auto : None,
                 debug : bool = False,
                 exactly_one = False,
                 theory = "Double") -> None:
        """
        Initialize a Deterministic Register Automaton (DRA) learner.

        Args:
            num_states (int): Number of states in the DRA.
            num_registers (int): Number of registers in the DRA.
            constants (List[float]): List of constants used in transitions.
            pos_samples (List[List[float]]): List of positive samples.
            neg_samples (List[List[float]]): List of negative samples.
        """
        self.states: Set[int] = set(range(num_states))
        self.num_registers: int = num_registers
        self.constants: List[float] = constants
        self.pos_samples: List[List[float]] = pos_samples
        self.neg_samples: List[List[float]] = neg_samples
        self.debug = debug
        self.exactly_one = exactly_one
        self.max_constants = 3  # Cap on number of constants
        self.golden_pos_samples = golden_pos_samples
        self.theory = theory

        # Generate condition and register assignment maps
        
        # register automaton
        self.condition_map = createMap(num_registers, constants, self.theory)
        # for k, v in self.condition_map.items():
            # print(k, v)
        self.assignment_map = createRegisterAssignmentMap(num_registers, constants)
    
        self.pairwise_det_map = self.compute_pairwise_det_map_dra()
        self._refresh_choice_cache()
        # Randomly generate transitions and register transitions
        self.transitions, self.register_transitions = self.generate_synchronized_transitions()
        # Initialize DRA with random states, transitions, and final states
        initial_state: int = random.choice(list(self.states))
        final_states: Set[int] = set(random.sample(list(self.states), random.randint(0, len(self.states))))


        if auto:
            self.mydra = auto
            # self.condition_map = auto.condition_map
            # self.assignment_map = auto.assignment_map
            
            self.pairwise_det_map = self.compute_pairwise_det_map_dra()
        else:
            self.mydra = DRA(
                states=self.states,
                num_registers=self.num_registers,
                constants=self.constants,
                transitions=self.transitions,
                initial_state=initial_state,
                final_states=final_states,
                condition_map=self.condition_map,
                assignment_map=self.assignment_map,
                register_transitions=self.register_transitions,
                theory=self.theory
            )
        self._always_true_condition_ids_cache = None
        self.best_dra = copy.deepcopy(self.mydra)
        self.best_score = self.compute_score()
        self.search_stats = None
        self._score_trace_cache = None
        self._pending_score_update = None

    def _refresh_choice_cache(self) -> None:
        self._state_choices = tuple(sorted(self.states))
        self._condition_choices = tuple(self.condition_map.keys())
        self._assignment_choices = tuple(self.assignment_map.keys())
        self._deterministic_pair_choices = tuple(self.pairwise_det_map)

    # --- Trap-state penalty -------------------------------------------------
    # Local search repeatedly rediscovers a cheap shortcut for languages that
    # need "accept a structured pattern, reject everything else": a final
    # state with an unconditional (guard = True) self-loop, which turns into
    # an "accept everything from here on" trap. Any sample whose run happens
    # to reach that state scores as correct regardless of what follows, which
    # is a much bigger and easier local reward than actually implementing the
    # target language's real per-symbol structure -- so hill-climbing/SA keep
    # re-finding it. Penalize it directly rather than trying to out-design
    # adversarial data against it.
    TRAP_STATE_PENALTY = 0.85

    def _always_true_condition_ids(self) -> Set[int]:
        if self._always_true_condition_ids_cache is None:
            ids = set()
            for cid, cond in self.condition_map.items():
                if self.theory == "Integer":
                    _, op1 = cond
                    if op1 == Operator.TT:
                        ids.add(cid)
                else:
                    lower, _, upper, _ = cond
                    if lower == "nop" and upper == "nop":
                        ids.add(cid)
            self._always_true_condition_ids_cache = ids
        return self._always_true_condition_ids_cache

    def _has_trap_final_state(self) -> bool:
        always_true_ids = self._always_true_condition_ids()
        if not always_true_ids:
            return False
        for state in self.mydra.final_states:
            for condition_id, next_state in self.mydra.transitions.get(state, []):
                if next_state == state and condition_id in always_true_ids:
                    return True
        return False

    def _apply_trap_penalty(self, raw_score: float) -> float:
        if raw_score <= 0.0:
            return raw_score
        if self._has_trap_final_state():
            return raw_score * self.TRAP_STATE_PENALTY
        return raw_score

    def test_word(self, word: List[float]) -> bool:
        """
        Test if a word (sample) is accepted by the DRA.

        Args:
            word (List[float]): The input word (sequence of floats).

        Returns:
            bool: True if the word is accepted, False otherwise.
        """
        current_state = self.initial_state
        registers = [0.0] * self.num_registers  # Initialize all registers to 0.0

        for symbol in word:
            transition_found = False

            # Check all transitions from the current state
            for condition, next_state in self.transitions.get(current_state, []):
                if self.condition_map[condition](symbol, registers, self.constants):
                    # Update registers if needed
                    assignment = self.register_transitions[current_state][0]
                    if assignment:
                        reg_idx, value_func = self.assignment_map[assignment[0]]
                        registers[reg_idx] = value_func(symbol, registers, self.constants)

                    current_state = next_state
                    transition_found = True
                    break

            if not transition_found:
                return False  # No valid transition means the word is rejected

        return current_state in self.final_states

     

    def introduce_promising_constant(self, epsilon: float = 0.01) -> None:
        """
        Analyze the sample data and introduce a new constant near the most common value seen
        in the samples. Updates all internal maps and transitions accordingly.

        Args:
            epsilon (float): Small perturbation to avoid exact overlap with common values.
        """
        if len(self.constants) >= self.max_constants:
            if self.debug:
                print("Max constant limit reached. Skipping new constant.")
            return
        # 1. Flatten and round sample values
        all_values = [v for seq in (self.pos_samples + self.neg_samples) for v in seq]
        if not all_values:
            if self.debug:
                print("No values in samples to extract a constant from.")
            return

        # 2. Find most frequent value
        most_common_value, _ = Counter(all_values).most_common(1)[0]
        new_constant = most_common_value

        # 3. Add the constant
        self.constants.append(new_constant)

        # 4. Rebuild maps
        self.condition_map = createMap(self.num_registers, self.constants)
        self.assignment_map = createRegisterAssignmentMap(self.num_registers, self.constants)
        self._always_true_condition_ids_cache = None

        self.pairwise_det_map = self.compute_pairwise_det_map_dra()
        self._refresh_choice_cache()

        # 5. Rebuild transitions
        self.transitions, self.register_transitions = self.generate_synchronized_transitions()

        # 6. Push updates to the automaton
        self.mydra.constants = self.constants
        self.mydra.condition_map = self.condition_map
        self.mydra.assignment_map = self.assignment_map
        self.mydra.transitions = self.transitions
        self.mydra.register_transitions = self.register_transitions
        self.mydra.refresh_execution_cache()
        self._invalidate_score_trace_cache()

        if self.debug:
            print(f"Introduced new constant near frequent value: {new_constant}")
            print(f"Constants now: {self.constants}")

    def introduce_new_constant(self, new_value: float) -> None:
        """
        Introduce a new constant into the learner and update all relevant automaton components.
        
        Args:
            new_value (float): The new constant to be added.
        """
        # 1. Add the new constant
        self.constants.append(new_value)

        # 2. Rebuild maps with updated constants
        self.condition_map = createMap(self.num_registers, self.constants)
        self.assignment_map = createRegisterAssignmentMap(self.num_registers, self.constants)
        self._always_true_condition_ids_cache = None

        # 3. Recompute pairwise determinism map
        
        self.pairwise_det_map = self.compute_pairwise_det_map_dra()
        self._refresh_choice_cache()
 
        # 4. Regenerate transitions
        self.transitions, self.register_transitions = self.generate_synchronized_transitions()

        # 5. Update the automaton’s internal data
        self.mydra.constants = self.constants
        self.mydra.condition_map = self.condition_map
        self.mydra.assignment_map = self.assignment_map
        self.mydra.transitions = self.transitions
        self.mydra.register_transitions = self.register_transitions
        self.mydra.refresh_execution_cache()
        self._invalidate_score_trace_cache()

        if self.debug:
            print(f"Introduced new constant: {new_value}")
            print(f"Total constants: {self.constants}")
            print(f"Updated condition map size: {len(self.condition_map)}")


    def mutation(self) -> None:
        """
        Apply mutations to improve the DRA score.
        If no improvement is found, apply a random modification.
        """
        current_score = self.compute_score()
        methods = [
            self.change_initial_state,
            self.modify_final_states,
            self.change_transition_and_register
        ]
        applied = False
        force_methods = [
            lambda score: self.modify_final_states(score, force_apply=True),
            lambda score: self.change_transition_and_register(score, force_apply=True)  # Flip transition inherently applies changes
        ]

        # Try each mutation method in sequence
        for method in methods:
            if method(current_score):
                applied = True
                break

        # If no improvement, randomly apply one mutation
        if not applied:
            random.choice(force_methods)(current_score)
            if self.debug:
                print("No improvement found. Applied a random modification.")

    def compute_score_test(self, dra, pos_sample, neg_sample) -> float:
        """
        Compute the score of the current DRA based on the positive and negative samples.

        Returns:
            float: The computed score.
        """
        correct = 0
        incorrect_pos_samples = []
        incorrect_neg_samples = []
        total = len(pos_sample) + len(neg_sample)
        for sample in pos_sample:
             
            if dra.accepts_input(sample, debug=False):
                correct += 1
            else:
                incorrect_pos_samples.append(sample)

        for sample in neg_sample:
             
            if not dra.accepts_input(sample, debug=False):
                correct += 1
            else:
                incorrect_neg_samples.append(sample)
        return correct / total

    def compute_score_return_inconsistent(self, dra) -> float:
        """
        Compute the score of the current DRA based on the positive and negative samples.

        Returns:
            float: The computed score.
        """
        correct = 0
        incorrect_pos_samples = []
        incorrect_neg_samples = []
        total = len(self.pos_samples) + len(self.neg_samples)
        for sample in self.pos_samples:
             
            if dra.accepts_input(sample, debug=False):
                correct += 1
            else:
                incorrect_pos_samples.append(sample)

        for sample in self.neg_samples:
             
            if not dra.accepts_input(sample, debug=False):
                correct += 1
            else:
                incorrect_neg_samples.append(sample)
        return (incorrect_pos_samples, incorrect_neg_samples)
    
    def compute_score(self) -> float:
        """
        Compute the weighted accuracy score of the DRA.

        Returns:
            float: Normalized score in [0, 1]
        """
        correct = 0
        golden_weight = 0
        if len(self.golden_pos_samples) > 0:
            golden_weight = (len(self.pos_samples) + len(self.neg_samples)) / (len(self.golden_pos_samples) * 10)
        weighted_total = len(self.pos_samples) + len(self.neg_samples) + golden_weight * len(self.golden_pos_samples)

        for sample in self.pos_samples:
            if self.mydra.accepts_input(sample, debug=False):
                correct += 1

        for sample in self.neg_samples:
             
            if not self.mydra.accepts_input(sample, debug=False):
                correct += 1

        for sample in self.golden_pos_samples:
            if self.mydra.accepts_input(sample, debug=False):
                    correct += golden_weight

        raw_score = correct / weighted_total if weighted_total > 0 else 0.0
        return self._apply_trap_penalty(raw_score)

    def _invalidate_score_trace_cache(self) -> None:
        self._score_trace_cache = None
        self._pending_score_update = None

    def _refresh_trace_cache_state_usage(self) -> None:
        if self._score_trace_cache is None:
            return

        state_usage_counts = Counter()
        for group_cache in self._score_trace_cache["groups"].values():
            for trace in group_cache["traces"]:
                for state in self._trace_reachable_states(trace):
                    state_usage_counts[state] += 1

        active_states = tuple(sorted(state_usage_counts))
        inactive_states = tuple(
            state for state in self._state_choices if state not in state_usage_counts
        )
        self._score_trace_cache["state_usage_counts"] = state_usage_counts
        self._score_trace_cache["active_states"] = active_states
        self._score_trace_cache["inactive_states"] = inactive_states

    def _choose_state_from_usage(
        self,
        candidates: Tuple[int, ...],
        usage_counts: Counter,
    ) -> int:
        if len(candidates) == 1:
            return candidates[0]

        weights = [usage_counts.get(state, 1) for state in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]

    def _choose_mutation_source_state(self) -> int:
        if self._score_trace_cache is None:
            return random.choice(self._state_choices)

        active_states = self._score_trace_cache.get("active_states", ())
        if active_states:
            usage_counts = self._score_trace_cache.get("state_usage_counts", Counter())
            if random.random() < 0.85:
                return self._choose_state_from_usage(active_states, usage_counts)
            return random.choice(active_states)

        return random.choice(self._state_choices)

    def _choose_transition_target_state(self) -> int:
        if self._score_trace_cache is None:
            return random.choice(self._state_choices)

        usage_counts = self._score_trace_cache.get("state_usage_counts", Counter())
        active_states = self._score_trace_cache.get("active_states", ())
        inactive_states = self._score_trace_cache.get("inactive_states", ())

        if inactive_states:
            inactive_ratio = len(inactive_states) / max(1, len(self._state_choices))
            growth_probability = min(0.75, 0.2 + 0.6 * inactive_ratio)
            if random.random() < growth_probability:
                return random.choice(inactive_states)

        if active_states:
            if random.random() < 0.8:
                return self._choose_state_from_usage(active_states, usage_counts)
            return random.choice(active_states)

        return random.choice(self._state_choices)

    def _sample_groups(self):
        return (
            ("pos", self.pos_samples),
            ("neg", self.neg_samples),
            ("golden", self.golden_pos_samples),
        )

    def _golden_weight(self) -> float:
        if len(self.golden_pos_samples) == 0:
            return 0.0
        return (len(self.pos_samples) + len(self.neg_samples)) / (len(self.golden_pos_samples) * 10)

    def _weighted_total(self) -> float:
        golden_weight = self._golden_weight()
        return len(self.pos_samples) + len(self.neg_samples) + golden_weight * len(self.golden_pos_samples)

    def _trace_contribution(
        self,
        group_name: str,
        trace: ExecutionTrace,
        final_states: Set[int] | None = None,
    ) -> float:
        if final_states is None:
            final_states = self.mydra.final_states
        accepted = trace.completed and trace.final_state in final_states
        if group_name == "pos":
            return 1.0 if accepted else 0.0
        if group_name == "neg":
            return 0.0 if accepted else 1.0
        return self._golden_weight() if accepted else 0.0

    def _trace_reachable_states(self, trace: ExecutionTrace) -> Set[int]:
        states = set(trace.source_states)
        if trace.completed and trace.final_state is not None:
            states.add(trace.final_state)
        return states

    def _build_trace_score_cache(self) -> float:
        golden_weight = self._golden_weight()
        weighted_total = self._weighted_total()
        correct = 0.0
        groups = {}

        for group_name, samples in self._sample_groups():
            traces = []
            by_source_state = defaultdict(set)
            completed_by_final = Counter()
            incomplete_count = 0

            for idx, sample in enumerate(samples):
                trace = self.mydra.evaluate_input_trace(sample)
                traces.append(trace)
                correct += self._trace_contribution(group_name, trace)

                for state in set(trace.source_states):
                    by_source_state[state].add(idx)

                if trace.completed and trace.final_state is not None:
                    completed_by_final[trace.final_state] += 1
                else:
                    incomplete_count += 1

            groups[group_name] = {
                "traces": traces,
                "by_source_state": dict(by_source_state),
                "completed_by_final": completed_by_final,
                "incomplete_count": incomplete_count,
                "total_completed": len(samples) - incomplete_count,
            }

        self._score_trace_cache = {
            "correct": correct,
            "weighted_total": weighted_total,
            "golden_weight": golden_weight,
            "groups": groups,
        }
        self._refresh_trace_cache_state_usage()
        self._pending_score_update = None
        raw_score = correct / weighted_total if weighted_total > 0 else 0.0
        return self._apply_trap_penalty(raw_score)

    def _correct_from_final_states(self, final_states: Set[int]) -> float:
        if self._score_trace_cache is None:
            return 0.0

        cache = self._score_trace_cache
        pos_counts = cache["groups"]["pos"]["completed_by_final"]
        neg_counts = cache["groups"]["neg"]["completed_by_final"]
        golden_counts = cache["groups"]["golden"]["completed_by_final"]

        pos_correct = sum(pos_counts[state] for state in final_states)
        neg_accept = sum(neg_counts[state] for state in final_states)
        neg_correct = cache["groups"]["neg"]["incomplete_count"] + cache["groups"]["neg"]["total_completed"] - neg_accept
        golden_correct = cache["golden_weight"] * sum(golden_counts[state] for state in final_states)
        return pos_correct + neg_correct + golden_correct

    def _score_final_state_mutation_from_cache(self) -> float:
        if self._score_trace_cache is None:
            return self._build_trace_score_cache()

        candidate_correct = self._correct_from_final_states(self.mydra.final_states)
        self._pending_score_update = {
            "type": "final_states",
            "candidate_correct": candidate_correct,
        }
        weighted_total = self._score_trace_cache["weighted_total"]
        raw_score = candidate_correct / weighted_total if weighted_total > 0 else 0.0
        return self._apply_trap_penalty(raw_score)

    def _score_transition_mutation_from_cache(
        self,
        mutated_state: int,
        original_final_states: Set[int] | None = None,
    ) -> float:
        if self._score_trace_cache is None:
            return self._build_trace_score_cache()

        cache = self._score_trace_cache
        candidate_correct = cache["correct"]
        group_updates = {}
        rescored_samples = 0

        for group_name, samples in self._sample_groups():
            group_cache = cache["groups"][group_name]
            affected_indices = group_cache["by_source_state"].get(mutated_state, ())
            if not affected_indices:
                continue

            updates = {}
            for idx in affected_indices:
                rescored_samples += 1
                old_trace = group_cache["traces"][idx]
                old_contribution = self._trace_contribution(
                    group_name,
                    old_trace,
                    final_states=original_final_states,
                )
                new_trace = self.mydra.evaluate_input_trace(samples[idx])
                new_contribution = self._trace_contribution(group_name, new_trace)
                candidate_correct += new_contribution - old_contribution
                if new_trace != old_trace:
                    updates[idx] = (old_trace, new_trace)

            if updates:
                group_updates[group_name] = updates

        self._pending_score_update = {
            "type": "transition",
            "candidate_correct": candidate_correct,
            "group_updates": group_updates,
            "rescored_samples": rescored_samples,
        }
        weighted_total = cache["weighted_total"]
        raw_score = candidate_correct / weighted_total if weighted_total > 0 else 0.0
        return self._apply_trap_penalty(raw_score)

    def _score_candidate_after_last_mutation(self) -> float:
        if not hasattr(self, "_last_mutation"):
            if self._score_trace_cache is None:
                return self._build_trace_score_cache()
            weighted_total = self._score_trace_cache["weighted_total"]
            return self._score_trace_cache["correct"] / weighted_total if weighted_total > 0 else 0.0

        mutation_type = self._last_mutation[0]
        if mutation_type == "transition":
            return self._score_transition_mutation_from_cache(self._last_mutation[1])
        if mutation_type == "frontier_split":
            return self._score_transition_mutation_from_cache(
                self._last_mutation[1],
                original_final_states=self._last_mutation[3],
            )
        if mutation_type == "final_states":
            return self._score_final_state_mutation_from_cache()
        return self.compute_score()

    def _commit_pending_score_update(self) -> None:
        if self._score_trace_cache is None or self._pending_score_update is None:
            return

        cache = self._score_trace_cache
        pending = self._pending_score_update

        if pending["type"] == "transition":
            for group_name, updates in pending["group_updates"].items():
                group_cache = cache["groups"][group_name]
                traces = group_cache["traces"]
                by_source_state = group_cache["by_source_state"]
                completed_by_final = group_cache["completed_by_final"]

                for idx, (old_trace, new_trace) in updates.items():
                    traces[idx] = new_trace

                    old_states = set(old_trace.source_states)
                    new_states = set(new_trace.source_states)

                    for state in old_states - new_states:
                        indices = by_source_state.get(state)
                        if indices is None:
                            continue
                        indices.discard(idx)
                        if not indices:
                            del by_source_state[state]

                    for state in new_states - old_states:
                        by_source_state.setdefault(state, set()).add(idx)

                    if old_trace.completed and old_trace.final_state is not None:
                        completed_by_final[old_trace.final_state] -= 1
                        if completed_by_final[old_trace.final_state] == 0:
                            del completed_by_final[old_trace.final_state]
                    else:
                        group_cache["incomplete_count"] -= 1

                    if new_trace.completed and new_trace.final_state is not None:
                        completed_by_final[new_trace.final_state] += 1
                    else:
                        group_cache["incomplete_count"] += 1

                group_cache["total_completed"] = len(traces) - group_cache["incomplete_count"]

            self._refresh_trace_cache_state_usage()

        cache["correct"] = pending["candidate_correct"]
        self._pending_score_update = None

    def _discard_pending_score_update(self) -> None:
        self._pending_score_update = None

    def _empty_operator_stats(self) -> Dict:
        return {
            "proposals": 0,
            "accepted": 0,
            "accepted_improving": 0,
            "accepted_non_improving": 0,
            "delta_sum": 0.0,
            "mutation_time": 0.0,
            "scoring_time": 0.0,
            "rollback_time": 0.0,
            "rescored_samples": 0,
        }

    def _init_search_stats(self, algorithm: str, initial_score: float) -> None:
        self.search_stats = {
            "algorithm": algorithm,
            "initial_score": initial_score,
            "best_score": self.best_score,
            "proposals": 0,
            "accepted": 0,
            "accepted_improving": 0,
            "accepted_non_improving": 0,
            "restarts": 0,
            "rescored_samples": 0,
            "timing": {
                "mutation": 0.0,
                "scoring": 0.0,
                "rollback": 0.0,
                "bookkeeping": 0.0,
                "total": 0.0,
            },
            "operators": {
                "frontier_split": self._empty_operator_stats(),
                "mutate_transition": self._empty_operator_stats(),
                "mutate_final_states": self._empty_operator_stats(),
                "systematic_mutate_final_states": self._empty_operator_stats(),
            },
            "best_score_over_time": [(0.0, self.best_score)],
        }

    def _record_best_score(self, start_time: float, score: float) -> None:
        if self.search_stats is None:
            return
        elapsed = round(time.monotonic() - start_time, 6)
        self.search_stats["best_score"] = score
        self.search_stats["best_score_over_time"].append((elapsed, score))

    def get_search_stats(self) -> Dict:
        if self.search_stats is None:
            return {}

        summary = copy.deepcopy(self.search_stats)
        if summary["proposals"] > 0:
            summary["acceptance_rate"] = summary["accepted"] / summary["proposals"]
            summary["improving_acceptance_rate"] = (
                summary["accepted_improving"] / summary["proposals"]
            )
            summary["non_improving_acceptance_rate"] = (
                summary["accepted_non_improving"] / summary["proposals"]
            )
            summary["avg_rescored_samples"] = summary["rescored_samples"] / summary["proposals"]
        else:
            summary["acceptance_rate"] = 0.0
            summary["improving_acceptance_rate"] = 0.0
            summary["non_improving_acceptance_rate"] = 0.0
            summary["avg_rescored_samples"] = 0.0

        if self._score_trace_cache is not None:
            summary["active_states"] = len(self._score_trace_cache.get("active_states", ()))
            summary["inactive_states"] = len(self._score_trace_cache.get("inactive_states", ()))

        for operator_stats in summary["operators"].values():
            proposals = operator_stats["proposals"]
            if proposals == 0:
                operator_stats["acceptance_rate"] = 0.0
                operator_stats["improving_acceptance_rate"] = 0.0
                operator_stats["non_improving_acceptance_rate"] = 0.0
                operator_stats["avg_delta"] = 0.0
                operator_stats["avg_mutation_time"] = 0.0
                operator_stats["avg_scoring_time"] = 0.0
                operator_stats["avg_rollback_time"] = 0.0
                operator_stats["avg_rescored_samples"] = 0.0
                continue

            operator_stats["acceptance_rate"] = operator_stats["accepted"] / proposals
            operator_stats["improving_acceptance_rate"] = (
                operator_stats["accepted_improving"] / proposals
            )
            operator_stats["non_improving_acceptance_rate"] = (
                operator_stats["accepted_non_improving"] / proposals
            )
            operator_stats["avg_delta"] = operator_stats["delta_sum"] / proposals
            operator_stats["avg_mutation_time"] = operator_stats["mutation_time"] / proposals
            operator_stats["avg_scoring_time"] = operator_stats["scoring_time"] / proposals
            operator_stats["avg_rollback_time"] = operator_stats["rollback_time"] / proposals
            operator_stats["avg_rescored_samples"] = operator_stats["rescored_samples"] / proposals

        return summary

    def _weighted_sample_margin(self) -> float:
        if self._score_trace_cache is None:
            return 0.0
        weighted_total = self._score_trace_cache.get("weighted_total", 0.0)
        return 1.0 / weighted_total if weighted_total > 0 else 0.0

    def _current_active_state_count(self) -> int:
        if self._score_trace_cache is None:
            return 0
        return len(self._score_trace_cache.get("active_states", ()))

    def _estimate_candidate_active_state_count(self) -> int:
        if self._score_trace_cache is None:
            return 0
        if self._pending_score_update is None or self._pending_score_update.get("type") != "transition":
            return self._current_active_state_count()

        usage_counts = Counter(self._score_trace_cache.get("state_usage_counts", Counter()))
        for updates in self._pending_score_update.get("group_updates", {}).values():
            for old_trace, new_trace in updates.values():
                old_states = self._trace_reachable_states(old_trace)
                new_states = self._trace_reachable_states(new_trace)
                for state in old_states - new_states:
                    usage_counts[state] -= 1
                    if usage_counts[state] <= 0:
                        del usage_counts[state]
                for state in new_states - old_states:
                    usage_counts[state] += 1
        return len(usage_counts)

    def _is_structure_expanding_mutation(self) -> bool:
        return hasattr(self, "_last_mutation") and self._last_mutation[0] == "frontier_split"

    def _should_accept_structure_expansion(self, current_score: float, new_score: float) -> bool:
        if not self._is_structure_expanding_mutation():
            return False
        if new_score < current_score - self._weighted_sample_margin():
            return False
        return self._estimate_candidate_active_state_count() > self._current_active_state_count()

    def _primary_distinguished_value(self):
        if self.theory != "Integer" or not self.constants:
            return None
        return float(self.constants[0])

    def _constant_condition_ids(self, constant_value: float):
        if self.theory != "Integer":
            return (None, None)

        eq_condition = None
        neq_condition = None
        for condition_id, condition in self.condition_map.items():
            if len(condition) != 2:
                continue
            bound, operator = condition
            if bound is None or not bound.startswith("c"):
                continue
            const_idx = int(bound[1:])
            if float(self.constants[const_idx]) != float(constant_value):
                continue
            if operator == Operator.EQ:
                eq_condition = condition_id
            elif operator == Operator.NEQ:
                neq_condition = condition_id
        return (eq_condition, neq_condition)

    def _primary_constant_condition_ids(self):
        distinguished_value = self._primary_distinguished_value()
        if distinguished_value is None:
            return (None, None)
        return self._constant_condition_ids(distinguished_value)

    def _extract_leading_constant_prefix_len(
        self,
        sample: List[float],
        distinguished_value: float,
    ) -> int | None:
        prefix_len = 0
        while prefix_len < len(sample) and float(sample[prefix_len]) == distinguished_value:
            prefix_len += 1

        if prefix_len == len(sample):
            return None
        if any(float(value) == distinguished_value for value in sample[prefix_len:]):
            return None
        return prefix_len

    def _iter_counting_frontier_records(self, misclassified_only: bool = True):
        if self._score_trace_cache is None:
            self._build_trace_score_cache()

        distinguished_value = self._primary_distinguished_value()
        if distinguished_value is None:
            return

        for group_name, samples in self._sample_groups():
            expected_positive = group_name != "neg"
            traces = self._score_trace_cache["groups"][group_name]["traces"]

            for idx, sample in enumerate(samples):
                prefix_len = self._extract_leading_constant_prefix_len(
                    sample,
                    distinguished_value,
                )
                if prefix_len is None or prefix_len == 0:
                    continue

                trace = traces[idx]
                if len(trace.source_states) <= prefix_len:
                    continue

                accepted = trace.completed and trace.final_state in self.mydra.final_states
                misclassified = accepted != expected_positive
                if group_name == "golden":
                    misclassified = not accepted

                if misclassified_only and not misclassified:
                    continue

                yield {
                    "group_name": group_name,
                    "idx": idx,
                    "sample": sample,
                    "expected_positive": expected_positive,
                    "misclassified": misclassified,
                    "prefix_len": prefix_len,
                    "frontier_state": trace.source_states[prefix_len],
                    "predecessor_state": trace.source_states[prefix_len - 1],
                }

    def _choose_counting_frontier_split_candidate(self):
        records = list(self._iter_counting_frontier_records(misclassified_only=True))
        if not records:
            records = list(self._iter_counting_frontier_records(misclassified_only=False))
        if not records:
            return None

        by_state_then_prefix = defaultdict(lambda: defaultdict(list))
        for record in records:
            by_state_then_prefix[record["frontier_state"]][record["prefix_len"]].append(record)

        best_candidate = None
        for frontier_state, prefix_map in by_state_then_prefix.items():
            prefixes = sorted(prefix_map)
            for zero_prefix_len in prefixes:
                next_prefix = zero_prefix_len + 1
                if next_prefix not in prefix_map:
                    continue

                left_records = prefix_map[zero_prefix_len]
                right_records = prefix_map[next_prefix]
                conflict_labels = {
                    record["expected_positive"]
                    for record in left_records + right_records
                }
                has_misclassified = any(
                    record["misclassified"]
                    for record in left_records + right_records
                )
                representative = next(
                    (
                        record
                        for record in right_records
                        if record["misclassified"]
                    ),
                    right_records[0],
                )
                priority = (
                    0 if has_misclassified else 1,
                    0 if len(conflict_labels) > 1 else 1,
                    zero_prefix_len,
                    -(len(left_records) + len(right_records)),
                )
                if best_candidate is None or priority < best_candidate["priority"]:
                    best_candidate = {
                        "priority": priority,
                        "split_kind": "counting",
                        "frontier_state": frontier_state,
                        "predecessor_state": representative["predecessor_state"],
                        "prefix_len": zero_prefix_len,
                        "incoming_is_distinguished": True,
                    }

        return best_candidate

    def _iter_suffix_frontier_records(self, misclassified_only: bool = False):
        if self._score_trace_cache is None:
            self._build_trace_score_cache()

        distinguished_value = self._primary_distinguished_value()
        if distinguished_value is None:
            return

        for group_name, samples in self._sample_groups():
            expected_positive = group_name != "neg"
            traces = self._score_trace_cache["groups"][group_name]["traces"]

            for idx, sample in enumerate(samples):
                trace = traces[idx]
                accepted = trace.completed and trace.final_state in self.mydra.final_states
                misclassified = accepted != expected_positive
                if group_name == "golden":
                    misclassified = not accepted
                if misclassified_only and not misclassified:
                    continue

                for pos in range(1, len(trace.source_states)):
                    incoming_symbol = float(sample[pos - 1])
                    yield {
                        "group_name": group_name,
                        "idx": idx,
                        "expected_positive": expected_positive,
                        "misclassified": misclassified,
                        "frontier_state": trace.source_states[pos],
                        "predecessor_state": trace.source_states[pos - 1],
                        "incoming_is_distinguished": incoming_symbol == distinguished_value,
                        "continuation": tuple(sample[pos:]),
                    }

    def _choose_suffix_frontier_split_candidate(self):
        records = list(self._iter_suffix_frontier_records(misclassified_only=False))
        if not records:
            return None
        distinguished_value = self._primary_distinguished_value()

        grouped = defaultdict(lambda: {True: [], False: []})
        for record in records:
            key = (record["frontier_state"], record["continuation"])
            grouped[key][record["incoming_is_distinguished"]].append(record)

        best_candidate = None
        for (frontier_state, continuation), grouped_records in grouped.items():
            left_records = grouped_records[True]
            right_records = grouped_records[False]
            if not left_records or not right_records:
                continue

            left_labels = {record["expected_positive"] for record in left_records}
            right_labels = {record["expected_positive"] for record in right_records}
            if len(left_labels | right_labels) < 2:
                continue

            left_misclassified = sum(1 for record in left_records if record["misclassified"])
            right_misclassified = sum(1 for record in right_records if record["misclassified"])
            has_misclassified = left_misclassified > 0 or right_misclassified > 0

            if left_misclassified > right_misclassified:
                redirect_is_distinguished = True
            elif right_misclassified > left_misclassified:
                redirect_is_distinguished = False
            else:
                redirect_is_distinguished = len(left_records) <= len(right_records)

            redirect_records = grouped_records[redirect_is_distinguished]
            predecessor_state, support = Counter(
                record["predecessor_state"] for record in redirect_records
            ).most_common(1)[0]

            priority = (
                0 if has_misclassified else 1,
                -max(left_misclassified, right_misclassified),
                -support,
                -len(continuation),
                -(len(left_records) + len(right_records)),
            )
            if best_candidate is None or priority < best_candidate["priority"]:
                best_candidate = {
                    "priority": priority,
                    "split_kind": "suffix",
                    "frontier_state": frontier_state,
                    "predecessor_state": predecessor_state,
                    "incoming_is_distinguished": redirect_is_distinguished,
                    "continuation_len": len(continuation),
                    "next_symbol_is_distinguished": (
                        continuation[0] == distinguished_value if continuation else None
                    ),
                    "support": len(left_records) + len(right_records),
                }

        return best_candidate

    def _choose_frontier_split_candidate(self):
        counting_candidate = self._choose_counting_frontier_split_candidate()
        if counting_candidate is not None:
            return counting_candidate
        return self._choose_suffix_frontier_split_candidate()

    def _frontier_split_available(self) -> bool:
        if self.num_registers != 0 or self.theory != "Integer":
            return False
        if self._score_trace_cache is None:
            return False
        if not self._score_trace_cache.get("inactive_states"):
            return False
        eq_condition, neq_condition = self._primary_constant_condition_ids()
        return eq_condition is not None and neq_condition is not None

    def _rebuild_predecessor_split(
        self,
        predecessor_state: int,
        new_state: int,
        desired_condition_id: int,
        other_condition_id: int,
    ):
        predecessor_transitions = self.mydra.transitions.get(predecessor_state, []).copy()
        if not predecessor_transitions:
            return None

        redirected = False
        new_predecessor_transitions = []
        for condition_id, next_state in predecessor_transitions:
            if condition_id == desired_condition_id:
                new_predecessor_transitions.append((condition_id, new_state))
                redirected = True
            else:
                new_predecessor_transitions.append((condition_id, next_state))

        if redirected:
            return new_predecessor_transitions

        if len(predecessor_transitions) == 1:
            condition_id, next_state = predecessor_transitions[0]
            if condition_id == other_condition_id:
                return [
                    (desired_condition_id, new_state),
                    (other_condition_id, next_state),
                ]
            if condition_id == desired_condition_id:
                return [(desired_condition_id, new_state)]

            condition = self.condition_map[condition_id]
            if len(condition) == 2 and condition[1] == Operator.TT:
                return [
                    (desired_condition_id, new_state),
                    (other_condition_id, next_state),
                ]

        tt_target_state = None
        for condition_id, next_state in predecessor_transitions:
            condition = self.condition_map[condition_id]
            if len(condition) == 2 and condition[1] == Operator.TT:
                tt_target_state = next_state
                break

        if tt_target_state is not None:
            return [
                (desired_condition_id, new_state),
                (other_condition_id, tt_target_state),
            ]

        return None

    def _resolve_branch_target_state(
        self,
        state: int,
        desired_condition_id: int,
        other_condition_id: int,
    ):
        state_transitions = self.mydra.transitions.get(state, ())
        if not state_transitions:
            return None

        for condition_id, next_state in state_transitions:
            if condition_id == desired_condition_id:
                return next_state

        for condition_id, next_state in state_transitions:
            condition = self.condition_map[condition_id]
            if len(condition) == 2 and condition[1] == Operator.TT:
                return next_state

        for condition_id, next_state in state_transitions:
            if condition_id == other_condition_id:
                return next_state

        return state_transitions[0][1]

    def mutate_frontier_split(self) -> bool:
        if not self._frontier_split_available():
            return False

        candidate = self._choose_frontier_split_candidate()
        if candidate is None:
            return False

        eq_condition, neq_condition = self._primary_constant_condition_ids()
        predecessor_state = candidate["predecessor_state"]
        frontier_state = candidate["frontier_state"]
        inactive_states = list(self._score_trace_cache["inactive_states"])
        new_state = inactive_states[0]
        specialized_branch_state = None
        if (
            candidate["split_kind"] == "suffix"
            and candidate.get("next_symbol_is_distinguished") is not None
            and len(inactive_states) > 1
        ):
            specialized_branch_state = inactive_states[1]

        original_final_states = self.mydra.final_states.copy()
        state_snapshots = {
            predecessor_state: (
                self.mydra.transitions.get(predecessor_state, []).copy(),
                self.mydra.register_transitions.get(predecessor_state, []).copy(),
            ),
            new_state: (
                self.mydra.transitions.get(new_state, []).copy(),
                self.mydra.register_transitions.get(new_state, []).copy(),
            ),
        }
        if specialized_branch_state is not None:
            state_snapshots[specialized_branch_state] = (
                self.mydra.transitions.get(specialized_branch_state, []).copy(),
                self.mydra.register_transitions.get(specialized_branch_state, []).copy(),
            )

        self.mydra.transitions[new_state] = self.mydra.transitions.get(frontier_state, []).copy()
        self.mydra.register_transitions[new_state] = self.mydra.register_transitions.get(frontier_state, []).copy()
        if frontier_state in original_final_states:
            self.mydra.final_states.add(new_state)
        else:
            self.mydra.final_states.discard(new_state)

        desired_condition = eq_condition if candidate["incoming_is_distinguished"] else neq_condition
        other_condition = neq_condition if candidate["incoming_is_distinguished"] else eq_condition
        new_predecessor_transitions = self._rebuild_predecessor_split(
            predecessor_state,
            new_state,
            desired_condition,
            other_condition,
        )
        if new_predecessor_transitions is None:
            self.mydra.transitions[predecessor_state] = state_snapshots[predecessor_state][0]
            self.mydra.register_transitions[predecessor_state] = state_snapshots[predecessor_state][1]
            self.mydra.transitions[new_state] = state_snapshots[new_state][0]
            self.mydra.register_transitions[new_state] = state_snapshots[new_state][1]
            self.mydra.final_states = original_final_states
            return False

        self.mydra.transitions[predecessor_state] = new_predecessor_transitions
        self.mydra.register_transitions[predecessor_state] = [
            (self._assignment_choices[0], next_state)
            for _, next_state in new_predecessor_transitions
        ]

        specialized_condition = None
        branch_target_state = None
        if specialized_branch_state is not None:
            specialized_condition = (
                eq_condition
                if candidate["next_symbol_is_distinguished"]
                else neq_condition
            )
            alternate_condition = (
                neq_condition if specialized_condition == eq_condition else eq_condition
            )
            branch_target_state = self._resolve_branch_target_state(
                frontier_state,
                specialized_condition,
                alternate_condition,
            )
            if branch_target_state is not None:
                self.mydra.transitions[specialized_branch_state] = self.mydra.transitions.get(branch_target_state, []).copy()
                self.mydra.register_transitions[specialized_branch_state] = self.mydra.register_transitions.get(branch_target_state, []).copy()
                if branch_target_state in original_final_states:
                    self.mydra.final_states.add(specialized_branch_state)
                else:
                    self.mydra.final_states.discard(specialized_branch_state)

                specialized_transitions = self._rebuild_predecessor_split(
                    new_state,
                    specialized_branch_state,
                    specialized_condition,
                    alternate_condition,
                )
                if specialized_transitions is None:
                    for state, (original_transitions, original_register_transitions) in state_snapshots.items():
                        self.mydra.transitions[state] = original_transitions
                        self.mydra.register_transitions[state] = original_register_transitions
                    self.mydra.final_states = original_final_states
                    return False

                self.mydra.transitions[new_state] = specialized_transitions
                self.mydra.register_transitions[new_state] = [
                    (self._assignment_choices[0], next_state)
                    for _, next_state in specialized_transitions
                ]

        self._last_mutation = (
            "frontier_split",
            predecessor_state,
            state_snapshots,
            original_final_states,
            {
                "frontier_state": frontier_state,
                "new_state": new_state,
                "split_kind": candidate["split_kind"],
                "prefix_len": candidate.get("prefix_len"),
                "continuation_len": candidate.get("continuation_len"),
                "incoming_is_distinguished": candidate["incoming_is_distinguished"],
                "specialized_branch_state": specialized_branch_state,
                "specialized_condition": specialized_condition,
                "specialized_target_state": branch_target_state,
            },
        )
        if self.debug:
            print(
                "Frontier split mutation: "
                f"split state {frontier_state} via predecessor {predecessor_state} "
                f"into new state {new_state} ({candidate['split_kind']})."
            )
        return True

    def _compatible_partner_conditions(self, condition_id: int):
        partners = []
        for left_id, right_id in self._deterministic_pair_choices:
            if left_id == condition_id:
                partners.append(right_id)
            elif right_id == condition_id:
                partners.append(left_id)
        return tuple(partners)

    def _mutate_transition_locally(
        self,
        state: int,
        original_transitions: List[Tuple[int, int]],
        original_register_transitions: List[Tuple[int, int]],
    ) -> bool:
        if not original_transitions:
            return False

        new_transitions = original_transitions.copy()
        new_register_transitions = original_register_transitions.copy()

        if len(original_transitions) == 2:
            branch_idx = random.randrange(2)
            condition_id, current_target = new_transitions[branch_idx]
            next_state = self._choose_transition_target_state()
            if len(self._state_choices) > 1:
                attempts = 0
                while next_state == current_target and attempts < 4:
                    next_state = self._choose_transition_target_state()
                    attempts += 1

            new_transitions[branch_idx] = (condition_id, next_state)
            if len(new_register_transitions) == 2:
                assignment_id, _ = new_register_transitions[branch_idx]
                new_register_transitions[branch_idx] = (assignment_id, next_state)
            else:
                new_register_transitions = [
                    (random.choice(self._assignment_choices), branch_target)
                    for _, branch_target in new_transitions
                ]

            self.mydra.transitions[state] = new_transitions
            self.mydra.register_transitions[state] = new_register_transitions
            return True

        if len(original_transitions) == 1 and not self.exactly_one:
            condition_id, next_state = original_transitions[0]
            partners = self._compatible_partner_conditions(condition_id)
            if partners:
                partner_condition = random.choice(partners)
                partner_target = self._choose_transition_target_state()
                if len(self._state_choices) > 1:
                    attempts = 0
                    while partner_target == next_state and attempts < 4:
                        partner_target = self._choose_transition_target_state()
                        attempts += 1

                if random.random() < 0.5:
                    new_transitions = [
                        (condition_id, next_state),
                        (partner_condition, partner_target),
                    ]
                else:
                    new_transitions = [
                        (partner_condition, partner_target),
                        (condition_id, next_state),
                    ]
                original_assignment = (
                    original_register_transitions[0][0]
                    if original_register_transitions
                    else random.choice(self._assignment_choices)
                )
                new_register_transitions = [
                    (original_assignment, branch_target)
                    if branch_target == next_state
                    else (random.choice(self._assignment_choices), branch_target)
                    for _, branch_target in new_transitions
                ]

                self.mydra.transitions[state] = new_transitions
                self.mydra.register_transitions[state] = new_register_transitions
                return True

        return False

    def _apply_search_mutation(
        self,
        no_improvement_steps: int = 0,
        growth_focus_steps: int = 0,
    ) -> str:
        split_probability = 0.16 if self._frontier_split_available() else 0.0
        transition_probability = 0.67 if split_probability > 0.0 else 0.82
        final_probability = 0.12 if split_probability > 0.0 else 0.13
        r = random.random()

        if split_probability > 0.0 and r < split_probability:
            if self.mutate_frontier_split():
                return "frontier_split"

        if r < split_probability + transition_probability:
            self.mutate_transition()
            return "mutate_transition"
        if r < split_probability + transition_probability + final_probability:
            self.mutate_final_states()
            return "mutate_final_states"

        self.systematic_mutate_final_states()
        return "systematic_mutate_final_states"


    def change_transition_and_register(self, current_score: float, force_apply: bool = False) -> bool:
        """
        Attempt to improve the score by combining transition flips and register assignment changes,
        while ensuring determinism.

        Args:
            current_score (float): The current score of the DRA.
            force_apply (bool): If True, apply a random modification when no improvement is found.

        Returns:
            bool: True if the change improves the score, False otherwise.
        """
        best_score = current_score
        best_transitions = self.mydra.transitions.copy()
        best_register_transitions = self.mydra.register_transitions.copy()

        if force_apply:
            if self.debug:
                print("Applying a random transition and register assignment as no improvement was found.")
            state = random.choice(list(self.mydra.states))
            condition_id = random.choice(list(self.condition_map.keys()))
            assignment_id = random.choice(list(self.assignment_map.keys()))
            next_state = random.choice(list(self.mydra.states))

            # Apply random changes
            self.mydra.transitions[state] = [(condition_id, next_state)]
            self.mydra.register_transitions[state] = [(assignment_id, next_state)]
            if self.debug:
                print(f"Random modification applied: State {state}, Condition {condition_id}, Assignment {assignment_id}")
            return True

        # Systematic exploration for each state
        for state in self.mydra.states:
            original_transitions = self.mydra.transitions[state].copy()
            original_register_transitions = self.mydra.register_transitions[state].copy()

            # Step 1: Try all single outgoing transitions
            for condition_id in self.condition_map.keys():
                for next_state in self.mydra.states:
                    # Apply a single transition
                    self.mydra.transitions[state] = [(condition_id, next_state)]

                    # Test all register assignments for the same transition
                    for assignment_id in self.assignment_map.keys():
                        self.mydra.register_transitions[state] = [(assignment_id, next_state)]

                        # Compute the score
                        candidate_score = self.compute_score()

                        # Check for improvement
                        if candidate_score > best_score:
                            best_score = candidate_score
                            best_transitions[state] = [(condition_id, next_state)]
                            best_register_transitions[state] = [(assignment_id, next_state)]
                            if self.debug:
                                print(f"Improved: State {state}, Condition {condition_id}, "
                                f"Assignment {assignment_id}, Score: {candidate_score}")

                    # Revert register transitions
                    self.mydra.register_transitions[state] = original_register_transitions

            # Step 2: Try all deterministic pairs
            for condition_pair in self.pairwise_det_map:
                for next_state_1 in self.mydra.states:
                    for next_state_2 in self.mydra.states:
                        if next_state_1 == next_state_2:
                            continue

                        # Apply the deterministic pair of transitions
                        self.mydra.transitions[state] = [
                            (condition_pair[0], next_state_1),
                            (condition_pair[1], next_state_2)
                        ]

                        # Test all register assignments for each pair transition
                        for assignment_id_1 in self.assignment_map.keys():
                            for assignment_id_2 in self.assignment_map.keys():
                                self.mydra.register_transitions[state] = [
                                    (assignment_id_1, next_state_1),
                                    (assignment_id_2, next_state_2)
                                ]

                                # Compute the score
                                candidate_score = self.compute_score()

                                # Check for improvement
                                if candidate_score > best_score:
                                    best_score = candidate_score
                                    best_transitions[state] = [
                                        (condition_pair[0], next_state_1),
                                        (condition_pair[1], next_state_2)
                                    ]
                                    best_register_transitions[state] = [
                                        (assignment_id_1, next_state_1),
                                        (assignment_id_2, next_state_2)
                                    ]
                                    if self.debug:
                                        print(f"Improved: State {state}, Conditions {condition_pair}, "
                                        f"Assignments {assignment_id_1}, {assignment_id_2}, Score: {candidate_score}")

                        # Revert register transitions
                        self.mydra.register_transitions[state] = original_register_transitions

            # Revert transitions to original
            self.mydra.transitions[state] = original_transitions

        # Apply the best transitions and register assignments if improvement occurred
        if best_score > current_score:
            self.mydra.transitions = best_transitions
            self.mydra.register_transitions = best_register_transitions
            return True

        if self.debug:
            print("No improvement by changing transitions and register assignments.")
        return False



    def change_initial_state(self, current_score: float, force_apply: bool = False) -> bool:
        """
        Change the initial state of the DRA to improve the score or apply a change forcibly.

        Args:
            current_score (float): The current score of the DRA.
            force_apply (bool): If True, apply a random change even without improvement.

        Returns:
            bool: True if the change improves the score, False otherwise.
        """
        best_score = current_score
        best_initial_state = self.mydra.initial_state
        candidates = list(self.mydra.states)
        # If no improvement, forcibly apply a random change if force_apply is True
        if force_apply:
            random_state = random.choice(candidates)
            self.mydra.initial_state = random_state
            if self.debug:
                print(f"Forced change to initial state: {random_state}")
            return True
        for candidate_state in candidates:
            if candidate_state == self.mydra.initial_state:
                continue

            # Temporarily change the initial state
            original_initial_state = self.mydra.initial_state
            self.mydra.initial_state = candidate_state

            # Compute score
            candidate_score = self.compute_score()

            if candidate_score > best_score:
                best_score = candidate_score
                best_initial_state = candidate_state

            # Revert change
            self.mydra.initial_state = original_initial_state

        # Apply the best state if improvement occurred
        if best_score > current_score:
            self.mydra.initial_state = best_initial_state
            if self.debug:
                print(f"Changed initial state to {best_initial_state} with improved score: {best_score}")
            return True



        if self.debug:
            print("No improvement by changing the initial state.")
        return False


    def modify_final_states(self, current_score: float, force_apply: bool = False) -> bool:
        """
        Modify the final states of the DRA to improve the score or apply a change forcibly.

        Args:
            current_score (float): The current score of the DRA.
            force_apply (bool): If True, apply a random change even without improvement.

        Returns:
            bool: True if the change improves the score, False otherwise.
        """
        best_score = current_score
        best_final_states = self.mydra.final_states.copy()
        candidates = list(self.mydra.states)
        # If no improvement, forcibly apply a random change if force_apply is True
        if force_apply:
            random_state = random.choice(candidates)
            if random_state in self.mydra.final_states:
                self.mydra.final_states.remove(random_state)
            else:
                self.mydra.final_states.add(random_state)
            if self.debug:
                print(f"Forced modification to final states: {self.mydra.final_states}")
            return True
        for candidate_state in candidates:
            # Toggle candidate_state in the final states
            original_final_states = self.mydra.final_states.copy()

            if candidate_state in self.mydra.final_states:
                self.mydra.final_states.remove(candidate_state)
            else:
                self.mydra.final_states.add(candidate_state)

            # Compute score
            candidate_score = self.compute_score()

            if candidate_score > best_score:
                best_score = candidate_score
                best_final_states = self.mydra.final_states.copy()

            # Revert change
            self.mydra.final_states = original_final_states

        # Apply the best change if improvement occurred
        if best_score > current_score:
            self.mydra.final_states = best_final_states
            if self.debug:
                print(f"Modified final states to {best_final_states} with improved score: {best_score}")
            return True



        if self.debug:
            print("No improvement by modifying the final states.")
        return False



    def flip_transition(self, current_score: float, force_apply: bool = False) -> None:
        """
        Systematically explore all possible changes to transitions.
        1. Try all single outgoing transitions and destination states for improvement.
        2. If no improvement is found, try all deterministic pairs and all pairs of outgoing states.
        3. If `force_apply` is True, apply a random modification even if no improvement is found.

        Args:
            current_score (float): The current score of the DRA.
            force_apply (bool): If True, apply a random modification when no improvement is found.
        """
        best_score = current_score
        best_transitions = self.mydra.transitions.copy()  # Keep track of the best transition configuration
        if force_apply:
            if self.debug:
                print("Applying a random transition as no improvement was found.")
            state = random.choice(list(self.mydra.states))
            if random.random() < 0.5:
                # Randomly add a single transition
                condition_id = random.choice(list(self.condition_map.keys()))
                next_state = random.choice(list(self.mydra.states))
                self.mydra.transitions[state] = [(condition_id, next_state)]
                print(f"Random single transition applied: State {state}, Condition {condition_id}, Next {next_state}")
            else:
                # Randomly add a deterministic pair
                if self.pairwise_det_map:
                    condition_pair = random.choice(list(self.pairwise_det_map))  # Convert set to list
                    next_state_1 = random.choice(list(self.mydra.states))
                    next_state_2 = random.choice(list(self.mydra.states))
                    self.mydra.transitions[state] = [
                        (condition_pair[0], next_state_1),
                        (condition_pair[1], next_state_2),
                    ]
                    if self.debug:
                        print(f"Random pair transition applied: State {state}, Conditions {condition_pair}, "
                        f"Next {next_state_1}, {next_state_2}")

        # Step 1: Explore all single outgoing transitions
        for state in self.mydra.states:
            original_transitions = self.mydra.transitions[state].copy()  # Save original transitions

            for condition_id in self.condition_map.keys():
                for next_state in self.mydra.states:
                    # Temporarily apply the change: clear existing transitions and add a single transition
                    self.mydra.transitions[state] = [(condition_id, next_state)]
                    candidate_score = self.compute_score()

                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_transitions[state] = [(condition_id, next_state)]
                        if self.debug:
                            print(f"Improved by flipping to single transition: State {state}, "
                            f"Condition {condition_id}, Next {next_state}, Score: {candidate_score}")

            # Revert to original transitions
            self.mydra.transitions[state] = original_transitions

        # Step 2: Explore all deterministic pairs if no improvement is found
        if best_score == current_score:
            for state in self.mydra.states:
                original_transitions = self.mydra.transitions[state].copy()  # Save original transitions

                for condition_pair in self.pairwise_det_map:
                    for next_state_1 in self.mydra.states:
                        for next_state_2 in self.mydra.states:
                            # Temporarily apply the pair of transitions
                            self.mydra.transitions[state] = [
                                (condition_pair[0], next_state_1),
                                (condition_pair[1], next_state_2),
                            ]
                            candidate_score = self.compute_score()

                            if candidate_score > best_score:
                                best_score = candidate_score
                                best_transitions[state] = [
                                    (condition_pair[0], next_state_1),
                                    (condition_pair[1], next_state_2),
                                ]
                                if self.debug:
                                    print(f"Improved by flipping to pair transition: State {state}, "
                                    f"Conditions {condition_pair}, Next {next_state_1}, {next_state_2}, Score: {candidate_score}")

                # Revert to original transitions
                self.mydra.transitions[state] = original_transitions

        # Step 3: Apply the best transitions found if any improvement occurred
        if best_score > current_score:
            self.mydra.transitions = best_transitions
            if self.debug:
                print(f"Best transitions applied with improved score: {best_score}")
        else:
            if self.debug:
                print("No improvement found by flipping transitions.")

            # Apply a random modification if force_apply is True

    def compute_pairwise_det_map_dra(self) -> Set[Tuple[int, int]]:
        """
        Compute deterministic pairs for the (left, op, right) format (EQ/NEQ/TRUE).
        Returns:
            Set[Tuple[int, int]]: Set of deterministic pairs (i, j).
        """
        if self.theory == "Integer":
            deterministic_pairs = set()
            if not self.states or self.exactly_one:
                return deterministic_pairs
            for i in self.condition_map.keys():
                cond_i = self.condition_map.get(i)
                # print(cond_i)
                # if not isinstance(cond_i, (list, tuple)) or len(cond_i) != 3:
                    # print('?')
                    # continue  # skip malformed entries
                left_i, op_i = cond_i
                # print(left_i, op_i)
                # If op_i is TRUE, it always overlaps, so skip all pairs with i
                if op_i == Operator.TT:
                    continue
                for j in self.condition_map.keys():
                    if j <= i:
                        continue
                    cond_j = self.condition_map.get(j)
                    # if not isinstance(cond_j, (list, tuple)) or len(cond_j) != 3:
                    #     continue
                    left_j, op_j = cond_j
                    # If op_j is TRUE, it always overlaps, so skip all pairs with j
                    if op_j == Operator.TT:
                        continue
                    # Deterministic if the two conditions never overlap
                    # For EQ/NEQ, overlap only if both are EQ and (left_i, right_i) == (left_j, right_j)
                    if op_i == op_j and left_i == left_j:
                        # Both EQ or both NEQ, same variables: not deterministic
                        continue
                    # Otherwise, treat as deterministic
                    deterministic_pairs.add((i, j))
            return deterministic_pairs
        else:
            deterministic_pairs = set()
            if self.states == 0 or self.exactly_one:
                return deterministic_pairs

            # Reuse a single z3.Solver() (push/pop) across the whole O(n^2)
            # sweep instead of constructing a fresh Solver per pair -- with a
            # few registers/constants the condition map (and therefore the
            # number of pairs) grows quickly, and Solver() construction, not
            # the actual solving, dominated the runtime here.
            solver = Solver()

            for i in self.condition_map.keys():
                lower_i, _, upper_i, _ = self.condition_map.get(i)

                # Skip unbounded intervals (both bounds are 'nop')
                if lower_i == "nop" and upper_i == "nop":
                    if self.debug:
                        print(f"Skipping transition {i} because it is unbounded (always overlaps).")
                    continue

                for j in self.condition_map.keys():
                    if j <= i:  # Skip symmetric pairs and self-pairs
                        continue

                    lower_j, _, upper_j, _ = self.condition_map.get(j)

                    # Skip if both bounds of j are unbounded
                    if lower_j == "nop" and upper_j == "nop":
                        continue

                    # Check determinism
                    transitions = [i, j]
                    if is_deterministic_z3(transitions, self.condition_map, self.constants, solver=solver):
                        deterministic_pairs.add((i, j))
                        #print(f"Deterministic: {(i, j)} -> {self.condition_map.get(i)} | {self.condition_map.get(j)}")
                    #else:
                    #   print(f"Non-deterministic: {(i, j)} -> {self.condition_map.get(i)} | {self.condition_map.get(j)}")

            return deterministic_pairs

    def generate_synchronized_transitions(self) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]]:
        """
        Generate synchronized transitions and register transitions for the DRA.
        For each state, transitions and their corresponding register assignments are generated together.

        Returns:
            Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]]:
                A tuple containing the transitions and register transitions dictionaries.
        """
        transitions = {}
        register_transitions = {}

        for state in self.states:
            num_transitions = random.choice([1, 2])  # Randomly decide number of transitions
            if num_transitions == 1 or self.exactly_one == True:
                # Single transition
                condition_id = random.choice(self._condition_choices)
                next_state = random.choice(self._state_choices)
                transitions[state] = [(condition_id, next_state)]

                # Synchronized register transition
                assignment_id = random.choice(self._assignment_choices)
                register_transitions[state] = [(assignment_id, next_state)]
            elif num_transitions == 2 and self._deterministic_pair_choices:
            #     # Deterministic pair of transitions
                pair = random.choice(self._deterministic_pair_choices)
                next_state_1 = random.choice(self._state_choices)
                next_state_2 = random.choice(self._state_choices)
                while next_state_2 == next_state_1:
                    next_state_2 = random.choice(self._state_choices)
                transitions[state] = [
                    (pair[0], next_state_1),
                    (pair[1], next_state_2)
                ]
                # Synchronized register transitions for the pair
                assignment_id_1 = random.choice(self._assignment_choices)
                assignment_id_2 = random.choice(self._assignment_choices)
                register_transitions[state] = [
                    (assignment_id_1, next_state_1),
                    (assignment_id_2, next_state_2)
                ]
            else:
                # Fallback for single transition
                condition_id = random.choice(self._condition_choices)
                next_state = random.choice(self._state_choices)
                transitions[state] = [(condition_id, next_state)]

                # Synchronized register transition
                assignment_id = random.choice(self._assignment_choices)
                register_transitions[state] = [(assignment_id, next_state)]

        return transitions, register_transitions

    def _acceptance_probability(self, current_score: float, new_score: float, temperature: float) -> float:
        """
        Calculate the probability of accepting a new state in simulated annealing.

        Args:
            current_score (float): Score of the current state.
            new_score (float): Score of the proposed state.
            temperature (float): Current temperature.

        Returns:
            float: The acceptance probability.
        """
        if new_score > current_score:
            return 1.0  # Always accept improvements
        return math.exp((new_score - current_score) / temperature)  # Probabilistically accept worse solutions

    def mutate_final_states(self) -> None:
        """
        Randomly mutate the set of final states in the DRA.
        """
        # Save the original final states for rollback
        original_final_states = self.mydra.final_states.copy()

        if self._score_trace_cache is not None:
            pos_counts = self._score_trace_cache["groups"]["pos"]["completed_by_final"]
            neg_counts = self._score_trace_cache["groups"]["neg"]["completed_by_final"]
            golden_counts = self._score_trace_cache["groups"]["golden"]["completed_by_final"]
            golden_weight = self._score_trace_cache["golden_weight"]

            gains = {
                state: pos_counts[state] + golden_weight * golden_counts[state] - neg_counts[state]
                for state in self.mydra.states
            }

            add_candidates = [state for state in (self.mydra.states - self.mydra.final_states) if gains[state] > 0]
            remove_candidates = [state for state in self.mydra.final_states if gains[state] < 0]

            if add_candidates or remove_candidates:
                best_add = max(add_candidates, key=lambda state: gains[state], default=None)
                best_remove = min(remove_candidates, key=lambda state: gains[state], default=None)

                add_gain = gains[best_add] if best_add is not None else float("-inf")
                remove_gain = -gains[best_remove] if best_remove is not None else float("-inf")

                if add_gain >= remove_gain and best_add is not None:
                    self.mydra.final_states.add(best_add)
                    self._last_mutation = ("final_states", original_final_states)
                    if self.debug:
                        print(f"Guided add of state {best_add} to final states.")
                    return
                if best_remove is not None:
                    self.mydra.final_states.remove(best_remove)
                    self._last_mutation = ("final_states", original_final_states)
                    if self.debug:
                        print(f"Guided removal of state {best_remove} from final states.")
                    return

        # Decide whether to add or remove a state
        if random.random() < 0.5:
            # Add a state, but only if there are non-final states available
            non_final_states = self.mydra.states - self.mydra.final_states
            if non_final_states:
                new_state = random.choice(list(non_final_states))
                self.mydra.final_states.add(new_state)
                if self.debug:
                    print(f"Added state {new_state} to final states.")
            else:
                if self.debug:
                    print("No non-final states available to add.")
        else:
            # Remove a state, but only if there are final states available
            if self.mydra.final_states:
                remove_state = random.choice(list(self.mydra.final_states))
                self.mydra.final_states.remove(remove_state)
                if self.debug:
                    print(f"Removed state {remove_state} from final states.")
            else:
                if self.debug:
                    print("No final states available to remove.")

        # Save mutation details for rollback
        self._last_mutation = ("final_states", original_final_states)

        from itertools import chain, combinations

    def systematic_mutate_final_states(self) -> None:
        """
        Systematically explore all possible combinations of final states
        and pick the combination that results in the best score.
        """
        if self._score_trace_cache is not None:
            original_final_states = self.mydra.final_states.copy()
            pos_counts = self._score_trace_cache["groups"]["pos"]["completed_by_final"]
            neg_counts = self._score_trace_cache["groups"]["neg"]["completed_by_final"]
            golden_counts = self._score_trace_cache["groups"]["golden"]["completed_by_final"]
            golden_weight = self._score_trace_cache["golden_weight"]

            # Final-state decisions decompose additively once traces are cached.
            gains = {}
            for state in self.mydra.states:
                gains[state] = pos_counts[state] + golden_weight * golden_counts[state] - neg_counts[state]

            best_final_states = {
                state
                for state, gain in gains.items()
                if gain > 0 or (gain == 0 and state in original_final_states)
            }
            best_score = self._correct_from_final_states(best_final_states)
            weighted_total = self._score_trace_cache["weighted_total"]

            self.mydra.final_states = best_final_states
            self._last_mutation = ("final_states", original_final_states)
            if self.debug:
                normalized_score = best_score / weighted_total if weighted_total > 0 else 0.0
                print(
                    f"Systematic mutation completed from cached traces. "
                    f"Best final states: {best_final_states}, Best score: {normalized_score}"
                )
            return

        def all_subsets(states):
            """
            Generate all subsets of a given set of states.
            """
            states_list = list(states)
            return chain.from_iterable(combinations(states_list, r) for r in range(len(states_list) + 1))

        # Save the original final states for rollback
        original_final_states = self.mydra.final_states.copy()
        best_score = self.compute_score()
        best_final_states = original_final_states

        if self.debug:
            print("Starting systematic exploration of final states.")

        # Explore all possible subsets of states as potential final states
        for candidate_final_states in all_subsets(self.mydra.states):
            candidate_final_states = set(candidate_final_states)  # Convert to set for consistency

            # Temporarily set the candidate final states
            self.mydra.final_states = candidate_final_states

            # Compute the score
            candidate_score = self.compute_score()
            if self.debug:
                print(f"Tested final states: {candidate_final_states}, Score: {candidate_score}")

            # Update the best score and best final states if improvement is found
            if candidate_score > best_score:
                best_score = candidate_score
                best_final_states = candidate_final_states
                if self.debug:
                    print(f"New best score: {best_score} with final states: {best_final_states}")

        # Apply the best final states
        self.mydra.final_states = best_final_states

        # Save the mutation for rollback
        self._last_mutation = ("final_states", original_final_states)
        if self.debug:
            print(f"Systematic mutation completed. Best final states: {best_final_states}, Best score: {best_score}")



    def mutate_transition(self) -> None:
        """
        Randomly mutate a transition and ensure a synchronized register assignment.
        This method respects deterministic pairing when mutating multiple transitions.
        """
        state = self._choose_mutation_source_state()
        if state not in self.mydra.transitions:
            condition_id = random.choice(self._condition_choices)
            next_state = self._choose_transition_target_state()
            self.mydra.transitions[state] = [(condition_id, next_state)]
            self.mydra.register_transitions[state] = [(random.choice(self._assignment_choices), next_state)]

        original_transitions = self.mydra.transitions[state].copy()
        original_register_transitions = self.mydra.register_transitions[state].copy()

        local_edit_probability = 0.7 if len(original_transitions) >= 1 else 0.0
        mutated = random.random() < local_edit_probability and self._mutate_transition_locally(
            state,
            original_transitions,
            original_register_transitions,
        )

        if not mutated:
            use_single_transition = random.random() < 0.5 or self.exactly_one
            new_transitions = []
            new_register_transitions = []

            if use_single_transition:
                condition_id = random.choice(self._condition_choices)
                use_top = random.random() < 0.1
                if use_top:
                    condition_id = self._condition_choices[-1]
                next_state = self._choose_transition_target_state()
                new_transitions = [(condition_id, next_state)]
                new_register_transitions = [(random.choice(self._assignment_choices), next_state)]
            elif self._deterministic_pair_choices:
                condition_pair = random.choice(self._deterministic_pair_choices)
                next_state_1 = self._choose_transition_target_state()
                next_state_2 = self._choose_transition_target_state()
                while next_state_2 == next_state_1:
                    next_state_2 = self._choose_transition_target_state()

                new_transitions = [
                    (condition_pair[0], next_state_1),
                    (condition_pair[1], next_state_2),
                ]
                new_register_transitions = [
                    (random.choice(self._assignment_choices), next_state_1),
                    (random.choice(self._assignment_choices), next_state_2),
                ]
            else:
                condition_id = random.choice(self._condition_choices)
                next_state = self._choose_transition_target_state()
                new_transitions = [(condition_id, next_state)]
                new_register_transitions = [(random.choice(self._assignment_choices), next_state)]

            self.mydra.transitions[state] = new_transitions
            self.mydra.register_transitions[state] = new_register_transitions

        self._last_mutation = ("transition", state, original_transitions, original_register_transitions)
        if self.debug:
            print(
                "Mutation completed with synchronization. "
                f"Transitions: {self.mydra.transitions[state]}, "
                f"Register Assignments: {self.mydra.register_transitions[state]}"
            )




    def revert_last_mutation(self) -> None:
        """
        Revert the last mutation to restore the previous state of the DRA.
        """
        if not hasattr(self, "_last_mutation"):
            self._discard_pending_score_update()
            return  # No mutation to revert

        mutation_type = self._last_mutation[0]
        if mutation_type == "transition":
            _, state, original_transitions, original_register_transitions = self._last_mutation
            self.mydra.transitions[state] = original_transitions
            self.mydra.register_transitions[state] = original_register_transitions
            if self.debug:
                print(f"Reverted transition mutation for state {state}.")
        elif mutation_type == "frontier_split":
            _, _, state_snapshots, original_final_states, _ = self._last_mutation
            for state, (original_transitions, original_register_transitions) in state_snapshots.items():
                self.mydra.transitions[state] = original_transitions
                self.mydra.register_transitions[state] = original_register_transitions
            self.mydra.final_states = original_final_states
            if self.debug:
                print("Reverted frontier split mutation.")
        elif mutation_type == "final_states":
            _, original_final_states = self._last_mutation
            self.mydra.final_states = original_final_states
            if self.debug:
                print(f"Reverted final states to original: {original_final_states}")

        # Clear the last mutation
        del self._last_mutation
        self._discard_pending_score_update()




    
    def run(self, max_iterations: int, target_score: float = 1.0) -> None:
        """
        Run the local search to optimize the DRA.

        Args:
            max_iterations (int): Maximum number of iterations to perform.
            target_score (float): The target score to stop the search (default is 1.0).
        """
        if self.debug:
            print(f"Starting local search with initial score: {self.best_score}")

        for iteration in range(max_iterations):
            # Perform a mutation
            self.mutation()

            # Compute the new score
            current_score = self.compute_score()

            # Check for improvement
            if current_score > self.best_score:
                self.best_score = current_score
                #self.best_dra = self.save_dra_state()  # Save the improved DRA state
                if self.debug:
                    print(f"Iteration {iteration + 1}: Improved score to {current_score}")
            else:
                if self.debug:
                    print(f"Iteration {iteration + 1}: New Score {current_score}")

            # Stop if the target score is achieved
            if self.best_score >= target_score:
                if self.debug:
                    print(f"Target score {target_score} achieved at iteration {iteration + 1}.")
                break
        else:
            if self.debug:
                print(f"Finished {max_iterations} iterations. Best score: {self.best_score}")

        # Load and visualize the best DRA
        #self.load_dra_state(self.best_dra)
        if self.debug:
            print("Final optimized DRA:")
        self.mydra.visualize("final_dra")

    def save_dra_state(self) -> Dict:
        """
        Save the best state of the DRA (not just current mydra).
        Returns:
            Dict: A dictionary containing the best DRA's configuration.
        """
        return {
            "initial_state": self.best_dra.initial_state,
            "final_states": self.best_dra.final_states.copy(),
            "transitions": {k: v.copy() for k, v in self.best_dra.transitions.items()},
            "register_transitions": {k: v.copy() for k, v in self.best_dra.register_transitions.items()},
        }

    def load_dra_state(self, dra_state: Dict) -> None:
        """
        Load a previously saved DRA state.

        Args:
            dra_state (Dict): A dictionary containing a saved DRA configuration.
        """
        self.mydra.initial_state = dra_state["initial_state"]
        self.mydra.final_states = dra_state["final_states"]
        self.mydra.transitions = {k: v.copy() for k, v in dra_state["transitions"].items()}
        self.mydra.register_transitions = {k: v.copy() for k, v in dra_state["register_transitions"].items()}
        self._invalidate_score_trace_cache()


    def change_register_assignment(self, current_score: float, force_apply: bool = False) -> bool:
        """
        Change the register assignments of the DRA to improve the score.
        Ensure at most one register assignment per transition.

        Args:
            current_score (float): The current score of the DRA.
            force_apply (bool): If True, apply a random modification when no improvement is found.

        Returns:
            bool: True if the change improves the score, False otherwise.
        """
        best_score = current_score
        best_register_transitions = self.mydra.register_transitions.copy()  # Track the best configuration

        # Step 1: Try deleting existing register assignments
        for state, assignments in self.mydra.register_transitions.items():
            if len(assignments) > 1:
                for i, (assignment_id, next_state) in enumerate(assignments):
                    # Temporarily remove the assignment
                    original_assignments = assignments[:]
                    self.mydra.register_transitions[state] = assignments[:i] + assignments[i+1:]

                    # Compute score
                    candidate_score = self.compute_score()

                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_register_transitions[state] = assignments[:i] + assignments[i+1:]
                        if self.debug:
                            print(f"Improved by deleting register assignment: State {state}, "
                            f"Assignment {assignment_id}, Score: {candidate_score}")

                    # Revert to original assignments
                    self.mydra.register_transitions[state] = original_assignments

        # Step 2: Try adding new register assignments
        for state in self.mydra.states:
            if len(self.mydra.register_transitions[state]) < 1:  # Ensure at most one assignment
                for assignment_id in self.assignment_map.keys():
                    for next_state in self.mydra.states:
                        # Skip adding duplicate assignments
                        if (assignment_id, next_state) in self.mydra.register_transitions[state]:
                            continue

                        # Temporarily add the assignment
                        original_assignments = self.mydra.register_transitions[state][:]
                        self.mydra.register_transitions[state].append((assignment_id, next_state))

                        # Compute score
                        candidate_score = self.compute_score()

                        if candidate_score > best_score:
                            best_score = candidate_score
                            best_register_transitions[state] = [(assignment_id, next_state)]
                            if self.debug:
                                print(f"Improved by adding register assignment: State {state}, "
                                f"Assignment {assignment_id}, Next {next_state}, Score: {candidate_score}")

                        # Revert to original assignments
                        self.mydra.register_transitions[state] = original_assignments

        # Apply the best transitions found if any improvement occurred
        if best_score > current_score:
            self.mydra.register_transitions = best_register_transitions
            return True

        # If no improvement is found, apply a random modification if force_apply is True
        if force_apply:
            if self.debug:
                print("No improvement by changing register assignments. Applying a random modification.")
            state = random.choice(list(self.mydra.states))
            assignment_id = random.choice(list(self.assignment_map.keys()))
            next_state = random.choice(list(self.mydra.states))

            # Apply a random register assignment
            self.mydra.register_transitions[state] = [(assignment_id, next_state)]
            if self.debug:
                print(f"Random register assignment applied: State {state}, "
                f"Assignment {assignment_id}, Next {next_state}")
            return True

        if self.debug:
            print("No improvement by changing register assignments.")
        return False

    def mutate_transition_specific(self, transition):
        """
        Mutate a specific transition in the automaton, similar to mutate_transition but targeted.
        """
        state = transition[0] if isinstance(transition, tuple) else random.choice(list(self.mydra.states))
        # Save original for rollback
        original_transitions = self.mydra.transitions[state].copy() if state in self.mydra.transitions else []
        original_register_transitions = self.mydra.register_transitions[state].copy() if state in self.mydra.register_transitions else []
        # Mutate the specific transition
        condition_id = random.choice(list(self.condition_map.keys()))
        next_state = random.choice(list(self.mydra.states))
        assignment_id = random.choice(list(self.assignment_map.keys()))
        self.mydra.transitions[state] = [(condition_id, next_state)]
        self.mydra.register_transitions[state] = [(assignment_id, next_state)]
        self._last_mutation = ("transition", state, original_transitions, original_register_transitions)

    def mutate_final_state_specific(self, state):
        """
        Mutate a specific final state: add or remove the given state from final states.
        """
        original_final_states = self.mydra.final_states.copy()
        if state in self.mydra.final_states:
            self.mydra.final_states.remove(state)
        else:
            self.mydra.final_states.add(state)
        self._last_mutation = ("final_states", original_final_states)

    def _reset_to_initial(self):
        """
        Reset the automaton to its initial configuration (random transitions, final states).
        """
        self.transitions, self.register_transitions = self.generate_synchronized_transitions()
        self.initial_state = random.choice(list(self.states))
        self.final_states = set(random.sample(list(self.states), random.randint(0, len(self.states))))
        # Re-initialize mydra
        self.mydra = DRA(
            self.states,
            self.num_registers,
            self.constants,
            self.condition_map,
            self.assignment_map,
            self.transitions,
            self.register_transitions,
            self.initial_state,
            self.final_states,
            theory=self.theory,
        )
        self._invalidate_score_trace_cache()

    def hill_climbing(self, timeout : float, restart_iterations = 10000) -> None:
        """
        Perform hill climbing to optimize the DRA.

        Args:
            max_iterations (int): Maximum number of iterations to perform.
            timeout (float): Time allowed for hill climbing (in seconds).
        """
        start_time = time.monotonic()
        current_score = self._build_trace_score_cache()
        self._init_search_stats("hill_climbing", current_score)

        iteration = 0
        while time.monotonic() - start_time < timeout:
            if iteration >= restart_iterations:
                reset_start = time.perf_counter()
                self._reset_to_initial()
                self.search_stats["restarts"] += 1
                self.search_stats["timing"]["bookkeeping"] += time.perf_counter() - reset_start

                rescore_start = time.perf_counter()
                current_score = self._build_trace_score_cache()
                self.search_stats["timing"]["scoring"] += time.perf_counter() - rescore_start
                if current_score > self.best_score:
                    bookkeeping_start = time.perf_counter()
                    self.best_score = current_score
                    self.best_dra = copy.deepcopy(self.mydra)
                    self.search_stats["timing"]["bookkeeping"] += time.perf_counter() - bookkeeping_start
                    self._record_best_score(start_time, current_score)
                iteration = 0

            iteration += 1

            mutation_start = time.perf_counter()
            operator_name = self._apply_search_mutation()
            mutation_time = time.perf_counter() - mutation_start
            self.search_stats["timing"]["mutation"] += mutation_time

            # Compute the new score
            scoring_start = time.perf_counter()
            new_score = self._score_candidate_after_last_mutation()
            scoring_time = time.perf_counter() - scoring_start
            self.search_stats["timing"]["scoring"] += scoring_time

            operator_stats = self.search_stats["operators"][operator_name]
            operator_stats["proposals"] += 1
            operator_stats["mutation_time"] += mutation_time
            operator_stats["scoring_time"] += scoring_time
            self.search_stats["proposals"] += 1
            rescored_samples = self._pending_score_update.get("rescored_samples", 0) if self._pending_score_update else 0
            operator_stats["rescored_samples"] += rescored_samples
            self.search_stats["rescored_samples"] += rescored_samples

            score_delta = new_score - current_score
            operator_stats["delta_sum"] += score_delta

            accept_neutral_growth = self._should_accept_structure_expansion(current_score, new_score)

            # Accept the mutation if it improves the score or is a neutral/slightly-worse growth move.
            if new_score > current_score or accept_neutral_growth:
                improved = new_score > current_score
                iteration = 0
                current_score = new_score
                self._commit_pending_score_update()
                operator_stats["accepted"] += 1
                self.search_stats["accepted"] += 1
                if improved:
                    operator_stats["accepted_improving"] += 1
                    self.search_stats["accepted_improving"] += 1
                else:
                    operator_stats["accepted_non_improving"] += 1
                    self.search_stats["accepted_non_improving"] += 1

                if new_score > self.best_score:
                    bookkeeping_start = time.perf_counter()
                    self.best_score = new_score
                    self.best_dra = copy.deepcopy(self.mydra)
                    self.search_stats["timing"]["bookkeeping"] += time.perf_counter() - bookkeeping_start
                    self._record_best_score(start_time, new_score)
            else:
                # Revert the mutation if no improvement
                rollback_start = time.perf_counter()
                self.revert_last_mutation()
                rollback_time = time.perf_counter() - rollback_start
                operator_stats["rollback_time"] += rollback_time
                self.search_stats["timing"]["rollback"] += rollback_time
                if self.debug:
                    print(f"Iteration {iteration + 1}: No improvement, reverted changes.")

            # Early termination if the score reaches 1.0
            if current_score >= 1.0:
                if self.debug:
                    print(f"Perfect score of 1.0 reached at iteration {iteration + 1}. Terminating early.")
                if current_score > self.best_score:
                    bookkeeping_start = time.perf_counter()
                    self.best_score = current_score
                    self.best_dra = copy.deepcopy(self.mydra)
                    self.search_stats["timing"]["bookkeeping"] += time.perf_counter() - bookkeeping_start
                    self._record_best_score(start_time, current_score)
                break

        self.search_stats["timing"]["total"] = time.monotonic() - start_time
        print('timeout with hc:', self.search_stats["timing"]["total"], self.best_score)
        
    def simulated_annealing(
        self,
        initial_temperature: float,
        cooling_rate: float,
        max_iterations: int,
        timeout: float = None,
        min_temperature: float = 1e-5
    ) -> None:
        """
        Perform simulated annealing to optimize the DRA.

        Args:
            initial_temperature (float): Starting temperature for the annealing process.
            cooling_rate (float): Rate at which the temperature decreases (e.g., 0.99).
            max_iterations (int): Maximum number of iterations to perform.
            timeout (float, optional): Maximum time allowed (in seconds). If None, run until max_iterations.
            min_temperature (float): Minimum temperature before reheat or stop.
        """
        current_temperature = initial_temperature
        current_score = self._build_trace_score_cache()
        start_time = time.monotonic()
        self._init_search_stats("simulated_annealing", current_score)
        no_improvement_steps = 0
        iteration = 0

        while iteration < max_iterations:
            # Timeout check
            if timeout is not None and (time.monotonic() - start_time) >= timeout:
                self.search_stats["timing"]["total"] = time.monotonic() - start_time
                # if self.debug:
                print(f"[Timeout reached] Stopped after {iteration} iterations, "
                    f"{time.monotonic() - start_time:.2f}s elapsed. Best Score={self.best_score:.4f}.")
                break

            # Randomly choose a mutation type
            mutation_start = time.perf_counter()
            operator_name = self._apply_search_mutation()
            mutation_time = time.perf_counter() - mutation_start
            self.search_stats["timing"]["mutation"] += mutation_time

            # Compute the new score
            scoring_start = time.perf_counter()
            new_score = self._score_candidate_after_last_mutation()
            scoring_time = time.perf_counter() - scoring_start
            self.search_stats["timing"]["scoring"] += scoring_time

            operator_stats = self.search_stats["operators"][operator_name]
            operator_stats["proposals"] += 1
            operator_stats["mutation_time"] += mutation_time
            operator_stats["scoring_time"] += scoring_time
            self.search_stats["proposals"] += 1
            rescored_samples = self._pending_score_update.get("rescored_samples", 0) if self._pending_score_update else 0
            operator_stats["rescored_samples"] += rescored_samples
            self.search_stats["rescored_samples"] += rescored_samples
            operator_stats["delta_sum"] += new_score - current_score

            # Early termination condition
            if new_score >= 1.0:
                current_score = new_score
                self._commit_pending_score_update()
                self.best_score = current_score
                self.best_dra = copy.deepcopy(self.mydra)
                operator_stats["accepted"] += 1
                operator_stats["accepted_improving"] += 1
                self.search_stats["accepted"] += 1
                self.search_stats["accepted_improving"] += 1
                self._record_best_score(start_time, current_score)
                self.search_stats["timing"]["total"] = time.monotonic() - start_time
                if self.debug:
                    print(f"[Perfect score reached] {current_score} at iteration {iteration}.")
                break

            # Acceptance decision
            accept_neutral_growth = self._should_accept_structure_expansion(current_score, new_score)
            if accept_neutral_growth or (
                self._acceptance_probability(current_score, new_score, current_temperature) > random.random()
            ):
                improved = new_score > current_score
                current_score = new_score
                self._commit_pending_score_update()
                operator_stats["accepted"] += 1
                self.search_stats["accepted"] += 1
                if improved:
                    operator_stats["accepted_improving"] += 1
                    self.search_stats["accepted_improving"] += 1
                else:
                    operator_stats["accepted_non_improving"] += 1
                    self.search_stats["accepted_non_improving"] += 1
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_dra = copy.deepcopy(self.mydra)
                    self._record_best_score(start_time, current_score)
                no_improvement_steps = 0 if improved else no_improvement_steps + 1
            else:
                # Revert mutation if not accepted
                rollback_start = time.perf_counter()
                self.revert_last_mutation()
                rollback_time = time.perf_counter() - rollback_start
                operator_stats["rollback_time"] += rollback_time
                self.search_stats["timing"]["rollback"] += rollback_time
                no_improvement_steps += 1

            # Temperature schedule
            current_temperature *= cooling_rate
            if current_temperature < min_temperature:
                current_temperature = initial_temperature

            iteration += 1

            # Optional debug output
            if self.debug and iteration % 500 == 0:
                elapsed = time.monotonic() - start_time
                print(f"SA[{iteration}] | Best={self.best_score:.4f} | Temp={current_temperature:.5f} | "
                    f"Elapsed={elapsed:.1f}s")

        self.search_stats["timing"]["total"] = time.monotonic() - start_time
        if self.debug:
            print(f"Simulated annealing finished after {iteration} iterations "
                f"(Best Score={self.best_score:.4f}).")


    def ant_colony_optimization(self, n_ants=10, n_iterations=1000, evaporation_rate=0.5, alpha=1.0, beta=2.0, timeout=600):
        """
        Perform Ant Colony Optimization to optimize the DRA.
        Args:
            n_ants (int): Number of ants per iteration.
            n_iterations (int): Number of iterations.
            evaporation_rate (float): Rate at which pheromone evaporates.
            alpha (float): Pheromone importance.
            beta (float): Heuristic importance.
            timeout (float): Maximum time allowed (seconds).
        """
        import time
        start_time = time.monotonic()
        pheromone = {}
        best_score = self.compute_score()
        best_dra = copy.deepcopy(self.mydra)
        if self.debug:
            print(f"Starting Ant Colony Optimization with initial score: {best_score}")

        for iteration in range(n_iterations):
            if time.monotonic() - start_time > timeout:
                print(f"Timeout reached for ACO at iteration {iteration+1}.")
                break
            ants = []
            scores = []
            for _ in range(n_ants):
                # Each ant starts from current best
                self._reset_to_initial()
                # Apply a series of targeted mutations
                for _ in range(3):
                    if random.random() < 0.5:
                        self.mutate_transition_specific(None)
                    else:
                        state = random.choice(list(self.states))
                        self.mutate_final_state_specific(state)
                score = self.compute_score()
                ants.append(copy.deepcopy(self.mydra))
                scores.append(score)
                if score > best_score:
                    best_score = score
                    best_dra = copy.deepcopy(self.mydra)
            # Output information for each iteration
            # print(f"ACO Iteration {iteration+1}: Best score so far: {best_score}")
            # print(f"    Best automaton: initial_state={best_dra.initial_state}, num_final_states={len(best_dra.final_states)}")
            # print(f"    Transitions: {sum(len(v) for v in best_dra.transitions.values())}, Register assignments: {sum(len(v) for v in best_dra.register_transitions.values())}")
            if best_score >= 1.0:
                # print(f"Perfect score reached at iteration {iteration+1}.")
                break
        self.best_score = best_score
        self.best_dra = best_dra

    def large_neighborhood_search(self, n_iterations=1000, destruction_size=2, timeout=600):
        """
        Perform Large Neighborhood Search (LNS) to optimize the DRA.
        Args:
            n_iterations (int): Number of iterations.
            destruction_size (int): Number of elements to destroy per iteration.
            timeout (float): Maximum time allowed (seconds).
        """
        import time
        start_time = time.monotonic()
        best_score = self.compute_score()
        best_dra = copy.deepcopy(self.mydra)

        for iteration in range(n_iterations):
            if time.monotonic() - start_time > timeout:
                print(f"Timeout reached for LNS at iteration {iteration+1}.")
                break
            # Destroy: randomly select transitions and/or final states to randomize
            destroyed_states = random.sample(list(self.states), min(destruction_size, len(self.states)))
            destroyed_transitions = []
            destroyed_final_states = []
            # Save originals for rollback
            original_transitions = {}
            original_register_transitions = {}
            original_final_states = self.mydra.final_states.copy()
            for state in destroyed_states:
                if state in self.mydra.transitions:
                    original_transitions[state] = self.mydra.transitions[state].copy()
                    original_register_transitions[state] = self.mydra.register_transitions[state].copy()
                    # Randomize transitions and register assignments
                    condition_id = random.choice(list(self.condition_map.keys()))
                    next_state = random.choice(list(self.states))
                    assignment_id = random.choice(list(self.assignment_map.keys()))
                    self.mydra.transitions[state] = [(condition_id, next_state)]
                    self.mydra.register_transitions[state] = [(assignment_id, next_state)]
                    destroyed_transitions.append(state)
                # Randomly add/remove final state
                if random.random() < 0.5:
                    if state in self.mydra.final_states:
                        self.mydra.final_states.remove(state)
                        destroyed_final_states.append(state)
                    else:
                        self.mydra.final_states.add(state)
                        destroyed_final_states.append(state)
            # Repair: apply a few local mutations
            for _ in range(3):
                if random.random() < 0.5:
                    self.mutate_transition_specific(None)
                else:
                    state = random.choice(list(self.states))
                    self.mutate_final_state_specific(state)
            score = self.compute_score()
            # Accept if improved, else revert
            if score > best_score:
                best_score = score
                best_dra = copy.deepcopy(self.mydra)
                # print(f"LNS[{iteration+1}]: {best_score}")
            else:
                # Revert destroyed transitions and final states
                for state in destroyed_transitions:
                    self.mydra.transitions[state] = original_transitions[state]
                    self.mydra.register_transitions[state] = original_register_transitions[state]
                self.mydra.final_states = original_final_states.copy()
                # print(f"LNS Iteration {iteration+1}: No improvement, reverted changes.")
            if best_score >= 1.0:
                # print(f"Perfect score reached at iteration {iteration+1}.")
                break
        self.best_score = best_score
        self.best_dra = best_dra
         

    def particle_swarm_optimization(self, swarm_size=10, n_iterations=1000, timeout=600):
        """
        Perform Particle Swarm Optimization (PSO) to optimize the DRA.
        Args:
            swarm_size (int): Number of particles in the swarm.
            n_iterations (int): Number of iterations.
            timeout (float): Maximum time allowed (seconds).
        """
        import time
        start_time = time.monotonic()
        # Each particle is a tuple: (dra, best_dra, best_score)
        swarm = []
        for _ in range(swarm_size):
            self._reset_to_initial()
            dra = copy.deepcopy(self.mydra)
            score = self.compute_score()
            swarm.append({'dra': dra, 'best_dra': copy.deepcopy(dra), 'best_score': score})
        global_best_score = max(p['best_score'] for p in swarm)
        global_best_dra = copy.deepcopy(max(swarm, key=lambda p: p['best_score'])['best_dra'])

        for iteration in range(n_iterations):
            if time.monotonic() - start_time > timeout:
                print(f"Timeout reached for PSO at iteration {iteration+1}.")
                break
            for i, particle in enumerate(swarm):
                # Random mutation (velocity)
                self.mydra = copy.deepcopy(particle['dra'])
                for _ in range(2):
                    if random.random() < 0.5:
                        self.mutate_transition_specific(None)
                    else:
                        state = random.choice(list(self.states))
                        self.mutate_final_state_specific(state)
                # Move toward personal best
                if random.random() < 0.5:
                    self.mydra = copy.deepcopy(particle['best_dra'])
                # Move toward global best
                if random.random() < 0.5:
                    self.mydra = copy.deepcopy(global_best_dra)
                score = self.compute_score()
                # Update personal best
                if score > particle['best_score']:
                    particle['best_score'] = score
                    particle['best_dra'] = copy.deepcopy(self.mydra)
                # Update global best
                if score > global_best_score:
                    global_best_score = score
                    global_best_dra = copy.deepcopy(self.mydra)
                    # print(f"PSO[{iteration+1}]: {global_best_score}")
            # print(f"    Global best automaton: initial_state={global_best_dra.initial_state}, num_final_states={len(global_best_dra.final_states)}")
            # print(f"    Transitions: {sum(len(v) for v in global_best_dra.transitions.values())}, Register assignments: {sum(len(v) for v in global_best_dra.register_transitions.values())}")
            if global_best_score >= 1.0:
                # print(f"Perfect score reached at iteration {iteration+1}.")
                break
        self.best_score = global_best_score
        self.best_dra = global_best_dra
       

    def genetic_algorithm(self, population_size=10, generations=1000, mutation_rate=0.3, timeout=600):
        """
        Perform Genetic Algorithm (GA) to optimize the DRA.
        Args:
            population_size (int): Number of individuals in the population.
            generations (int): Number of generations.
            mutation_rate (float): Probability of mutation per individual.
            timeout (float): Maximum time allowed (seconds).
        """
        import time
        start_time = time.monotonic()
        # Initialize population
        population = []
        for _ in range(population_size):
            self._reset_to_initial()
            dra = copy.deepcopy(self.mydra)
            score = self.compute_score()
            population.append({'dra': dra, 'score': score})

        for gen in range(generations):
            if time.monotonic() - start_time > timeout:
                print(f"Timeout reached for GA at generation {gen+1}.")
                break
            # Selection: sort by score, keep top half
            population.sort(key=lambda p: p['score'], reverse=True)
            survivors = population[:population_size//2]
            # Crossover: create new individuals by combining parents
            offspring = []
            while len(offspring) < population_size - len(survivors):
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                child_dra = copy.deepcopy(parent1['dra'])
                # Crossover transitions and final states
                for state in self.states:
                    if random.random() < 0.5:
                        child_dra.transitions[state] = copy.deepcopy(parent2['dra'].transitions[state])
                        child_dra.register_transitions[state] = copy.deepcopy(parent2['dra'].register_transitions[state])
                # Crossover final states
                if random.random() < 0.5:
                    child_dra.final_states = copy.deepcopy(parent2['dra'].final_states)
                # Evaluate child
                self.mydra = child_dra
                score = self.compute_score()
                offspring.append({'dra': copy.deepcopy(child_dra), 'score': score})
            # Mutation
            for individual in offspring:
                if random.random() < mutation_rate:
                    self.mydra = copy.deepcopy(individual['dra'])
                    for _ in range(2):
                        if random.random() < 0.5:
                            self.mutate_transition_specific(None)
                        else:
                            state = random.choice(list(self.states))
                            self.mutate_final_state_specific(state)
                    individual['dra'] = copy.deepcopy(self.mydra)
                    individual['score'] = self.compute_score()
            # Form new population
            population = survivors + offspring
            # Output info
            best_score = max(p['score'] for p in population)
            best_dra = max(population, key=lambda p: p['score'])['dra']
            # if gen % 500 == 0:
                # print(f"GA[{gen+1}]: {best_score}")
            # print(f"    Best automaton: initial_state={best_dra.initial_state}, num_final_states={len(best_dra.final_states)}")
            # print(f"    Transitions: {sum(len(v) for v in best_dra.transitions.values())}, Register assignments: {sum(len(v) for v in best_dra.register_transitions.values())}")
            if best_score >= 1.0:
                # print(f"Perfect score reached at generation {gen+1}.")
                break
        # Final best
        self.best_score = best_score
        self.best_dra = copy.deepcopy(best_dra)
        

    def tabu_search(self, n_iterations=1000, tabu_size=50, timeout=600):
        """
        Perform Tabu Search to optimize the DRA.
        Args:
            n_iterations (int): Number of iterations.
            tabu_size (int): Size of the tabu list.
            timeout (float): Maximum time allowed (seconds).
        """
        import time
        start_time = time.monotonic()
        tabu_list = []
        best_score = self.compute_score()
        best_dra = copy.deepcopy(self.mydra)

        for iteration in range(n_iterations):
            if time.monotonic() - start_time > timeout:
                print(f"Timeout reached for Tabu Search at iteration {iteration+1}.")
                break
            # Save current state for tabu
            current_state = {
                'initial_state': self.mydra.initial_state,
                'final_states': tuple(sorted(self.mydra.final_states)),
                'transitions': tuple((k, tuple(v)) for k, v in self.mydra.transitions.items()),
                'register_transitions': tuple((k, tuple(v)) for k, v in self.mydra.register_transitions.items()),
            }
            # Generate candidate moves
            candidates = []
            for _ in range(5):
                self.mydra = copy.deepcopy(best_dra)
                if random.random() < 0.5:
                    self.mutate_transition_specific(None)
                else:
                    state = random.choice(list(self.states))
                    self.mutate_final_state_specific(state)
                candidate_state = {
                    'initial_state': self.mydra.initial_state,
                    'final_states': tuple(sorted(self.mydra.final_states)),
                    'transitions': tuple((k, tuple(v)) for k, v in self.mydra.transitions.items()),
                    'register_transitions': tuple((k, tuple(v)) for k, v in self.mydra.register_transitions.items()),
                }
                score = self.compute_score()
                if candidate_state not in tabu_list:
                    candidates.append((score, copy.deepcopy(self.mydra), candidate_state))
            # Select best non-tabu candidate
            if candidates:
                candidates.sort(reverse=True, key=lambda x: x[0])
                score, dra, state_repr = candidates[0]
                self.mydra = copy.deepcopy(dra)
                if score > best_score:
                    best_score = score
                    best_dra = copy.deepcopy(dra)
                    # print(f"Tabu[{iteration+1}]: {best_score}")
                else:
                    # print(f"Tabu Iteration {iteration+1}: No improvement, best candidate score: {score}")
                    pass
                # Update tabu list
                tabu_list.append(state_repr)
                if len(tabu_list) > tabu_size:
                    tabu_list.pop(0)
            else:
                # print(f"Tabu Iteration {iteration+1}: All candidates tabu, skipping.")
                pass
            if best_score >= 1.0:
                # print(f"Perfect score reached at iteration {iteration+1}.")
                break
        self.best_score = best_score
        self.best_dra = best_dra
