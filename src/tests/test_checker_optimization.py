"""Differential and integration checks for compiled reuse and liveness."""
from fractions import Fraction

import pytest
import z3

from test_robustness import (
    METRICS, below_three, contains_equal_pair, increasing,
)
from register_automata.ra import RegisterAutomata, Transition
from register_automata.raa import RAA, RAA_Transition
from register_automata.robustness import RobustnessChecker, find_witness


@pytest.mark.parametrize("factory", [increasing, contains_equal_pair, below_three])
@pytest.mark.parametrize("metric", METRICS)
def test_pruned_and_full_search_agree_across_reused_queries(factory, metric):
    dra, raa = factory(), METRICS[metric]()
    pruned = RobustnessChecker(dra, raa)
    full = RobustnessChecker(dra, raa, prune_registers=False)
    for v in ([1, 2, 3], [3, 1], [2, 2, 5], [], [0]):
        for delta in (Fraction(1, 2), Fraction(3, 2)):
            a = pruned.find_witness(v, delta, timeout=10)
            b = full.find_witness(v, delta, timeout=10)
            assert (a.robust, a.infimum) == (b.robust, b.infimum)


def test_repeated_queries_do_not_recompile_or_complement(monkeypatch):
    from register_automata import robustness
    dra = increasing()
    checker = RobustnessChecker(dra, METRICS["hamming"]())

    def unexpected(*args, **kwargs):
        pytest.fail("a compiled query rebuilt an automaton")

    monkeypatch.setattr(robustness, "compile_guard", unexpected)
    monkeypatch.setattr(RegisterAutomata, "complement", unexpected)
    assert checker.find_witness([1, 2, 3], 1).robust
    assert not checker.find_witness([1, 2, 3], Fraction(3, 2)).robust
    assert not checker.find_witness([3, 2, 1], Fraction(5, 2)).robust
    assert checker.find_witness([10, 20], 1).robust


def test_liveness_follows_copies_and_overwrites():
    c, r0, r1, unused = z3.Reals("curr r0 r1 r_unused")
    dra = RegisterAutomata(4, [
        Transition(c, 0, True, {"r0": "curr", "r_unused": "curr"}, 1),
        # Simultaneous assignment: r1 receives the OLD r0.
        Transition(c, 1, True, {"r0": 0, "r1": "r0"}, 2),
        Transition(c, 2, c == r1, {}, 3),
        Transition(c, 3, True, {"r_unused": "curr"}, 3),
    ], [r0, r1, unused], [3])
    checker = RobustnessChecker(dra, METRICS["last_letter"]())
    view = checker.targets[False]
    assert view.regs == ["r0", "r1"]
    assert view.live[0] == frozenset()
    assert view.live[1] == {"r0"}
    assert view.live[2] == {"r1"}
    assert view.live[3] == frozenset()
    result = checker.find_witness([1, 5, 1], Fraction(1, 2))
    assert not result.robust and result.infimum == 0
    assert result.witness[:2] == [1, 5]
    assert not checker.accepts(result.witness)


def test_liveness_fixed_point_and_first_match_guards():
    c, r0, r1 = z3.Reals("curr r0 r1")
    dra = RegisterAutomata(4, [
        Transition(c, 0, True, {"r0": "curr"}, 1),
        Transition(c, 1, c == 9, {"r1": "r0"}, 2),
        Transition(c, 1, True, {}, 1),
        Transition(c, 2, c == r1, {}, 3),
        Transition(c, 2, True, {}, 2),
    ], [r0, r1], [3])
    checker = RobustnessChecker(dra, METRICS["hamming"]())
    assert "r0" in checker.targets[False].live[1]
    assert "r1" in checker.targets[False].live[2]
    assert checker.accepts([2, 8, 9, 2])
    result = checker.find_witness([2, 8, 9, 2], Fraction(3, 2))
    assert not result.robust and result.cost == 1


def test_forgetting_dead_values_reduces_the_search():
    c, r0 = z3.Reals("curr r0")
    # r0 matters for the second letter, then is overwritten without being read.
    dra = RegisterAutomata(3, [
        Transition(c, 0, True, {"r0": "curr"}, 1),
        Transition(c, 1, c < r0, {}, 2),
        Transition(c, 2, True, {"r0": "curr"}, 2),
    ], [r0], [2])
    raa = METRICS["l1"]()
    pruned = RobustnessChecker(dra, raa)
    full = RobustnessChecker(dra, raa, prune_registers=False)
    v = [10, 5, 3, 2, 1, 4]
    a = pruned.find_witness(v, Fraction(1, 2))
    b = full.find_witness(v, Fraction(1, 2))
    assert a.robust and b.robust
    assert a.expanded < b.expanded


def test_dead_raa_assignment_keeps_original_head_requirements():
    c, c1, c2, reg = z3.Reals("curr curr1 curr2 r_metric")
    dra = RegisterAutomata(2, [Transition(c, 0, True, {}, 1)], [], [1])
    raa = RAA([0, 1], 0, [1], [
        RAA_Transition(c1, c2, 0, True, {"r_metric": "curr2"},
                       1, [0, 0, 0], 1),
    ], [reg])
    checker = RobustnessChecker(dra, raa)
    assert checker.raa.regs == []
    # The dead write still requires a fixed input letter; v is exhausted.
    assert checker.find_witness([], 1).robust


@pytest.mark.parametrize("delta", [0, -1])
def test_nonpositive_open_radius_is_empty(delta):
    assert RobustnessChecker(below_three(), METRICS["edit"]()).find_witness(
        [5], delta).robust


@pytest.mark.parametrize("timeout", [0, -1])
def test_expired_budget_never_produces_a_verdict(timeout):
    checker = RobustnessChecker(increasing(), METRICS["hamming"]())
    with pytest.raises(TimeoutError):
        checker.find_witness([1, 2, 3], 1, timeout=timeout)
    with pytest.raises(TimeoutError):
        find_witness(increasing(), METRICS["hamming"](), [1, 2, 3],
                     1, True, timeout=timeout)


def test_legacy_entry_point_uses_exact_boundary_and_real_zero_cost_witness(monkeypatch):
    import robustness_check as legacy

    def no_network(*args, **kwargs):
        pytest.fail("DRA checking must not need a neural model")

    monkeypatch.setattr(legacy, "predict_with_model", no_network)
    args = dict(dra=increasing(), inc=1, language="synthetic", silent=True)
    assert legacy.evaluate_robustness(delta=1, metric_name="hamming",
                                      vectors=[[1, 2, 3]], **args) == (
                                          True, None, False, 1, 1)
    robust, witness, timed_out, delta, index = legacy.evaluate_robustness(
        delta=Fraction(1, 10), metric_name="dtw", vectors=[[1, 2, 3]], **args)
    assert not robust and not timed_out and delta == Fraction(1, 10) and index == 1
    assert witness != [1, 2, 3]
    assert not RobustnessChecker(args["dra"], METRICS["dtw"]()).accepts(witness)


def test_legacy_resume_timeout_and_empty_witness():
    from robustness_check import evaluate_robustness
    c = z3.Real("curr")
    dra = RegisterAutomata(2, [Transition(c, 0, True, {}, 1),
                             Transition(c, 1, True, {}, 1)], [], [1])
    args = dict(dra=dra, delta=Fraction(3, 2), inc=1, metric_name="edit",
                language="synthetic", vectors=[[1]], silent=True)
    assert evaluate_robustness(timeout=0, **args) == (
        False, None, True, Fraction(3, 2), 0)
    assert evaluate_robustness(**args) == (False, [], False, Fraction(3, 2), 1)


def test_pac_validation_reuses_one_checker_and_confirms_network_flip(monkeypatch):
    import pac_pipeline as pac
    creations = []
    real_checker = RobustnessChecker

    def make_checker(*args, **kwargs):
        checker = real_checker(*args, **kwargs)
        creations.append(checker)
        return checker

    monkeypatch.setattr(pac, "RobustnessChecker", make_checker)
    monkeypatch.setattr(pac, "_draw", lambda *args: [[1, 2, 3], [3, 2, 1]])
    monkeypatch.setattr(pac, "_label", lambda samples, *args: [
        all(x < y for x, y in zip(s, s[1:])) for s in samples])
    out = pac.validate(increasing(), "synthetic", "language", "hamming",
                       1, p=0.95, gamma=0.05, epsilon=0.05, max_len=3)
    assert out.status == "accepted" and out.checked == 2
    assert len(creations) == 1
    creations.clear()
    out = pac.validate(increasing(), "synthetic", "language", "hamming",
                       Fraction(3, 2), p=0.95, gamma=0.05, epsilon=0.05, max_len=3)
    assert out.status == "genuine" and out.witness is not None
    assert len(creations) == 1
