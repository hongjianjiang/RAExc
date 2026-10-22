"""Tests for the exact DRA robustness checker (register_automata.robustness).

Run from the repository root:  python3 -m pytest src/tests -q
"""
import itertools
import os
import sys
from fractions import Fraction

import pytest
import z3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from register_automata.ra import RegisterAutomata, Transition  # noqa: E402
from register_automata.raa import RAA, RAA_Transition  # noqa: E402
from register_automata.metric_automata import (  # noqa: E402
    create_dtw_raa, create_edit_raa, create_hamming_raa, create_l1_raa,
    create_last_letter_raa)
from register_automata.robustness import _CompiledRAA, _CompiledDRA, find_witness  # noqa: E402

curr = z3.Real("curr")
r0 = z3.Real("r0")

METRICS = {
    "hamming": create_hamming_raa,
    "l1": create_l1_raa,
    "edit": create_edit_raa,
    "dtw": create_dtw_raa,
    "last_letter": create_last_letter_raa,
}


def increasing(strict=True):
    """1-register DRA: strictly (or non-strictly) increasing sequences."""
    up = curr > r0 if strict else curr >= r0
    ts = [
        Transition(curr, 0, True, {"r0": "curr"}, 1),
        Transition(curr, 1, up, {"r0": "curr"}, 1),
    ]
    return RegisterAutomata(2, ts, [r0], [1])


def below_three():
    """0-register DRA: every letter is < 3."""
    ts = [Transition(curr, 0, curr < 3, {}, 0)]
    return RegisterAutomata(1, ts, [], [0])


def contains_equal_pair():
    """1-register DRA: first letter repeats somewhere later."""
    ts = [
        Transition(curr, 0, True, {"r0": "curr"}, 1),
        Transition(curr, 1, curr == r0, {}, 2),
        Transition(curr, 1, curr != r0, {}, 1),
        Transition(curr, 2, True, {}, 2),
    ]
    return RegisterAutomata(3, ts, [r0], [2])


def check(dra, metric, v, delta):
    accepted = _CompiledDRA(dra, set()).run([Fraction(repr(float(x))) for x in v])
    return find_witness(dra, METRICS[metric](), v, delta, accepted, timeout=60)


# --- boundary cases with known answers --------------------------------------

def test_last_letter_strict_guard_attained_infimum():
    # Violating "strictly increasing" needs w3 <= 2: cost 3 - w3 >= 1, attained.
    dra = increasing(strict=True)
    assert check(dra, "last_letter", [1, 2, 3], 1).robust
    res = check(dra, "last_letter", [1, 2, 3], Fraction(101, 100))
    assert not res.robust and res.infimum == 1 and res.cost == 1


def test_last_letter_nonstrict_guard_unattained_infimum():
    # Violating "non-decreasing" needs w3 < 2: infimum 1 is not attained.
    dra = increasing(strict=False)
    assert check(dra, "last_letter", [1, 2, 3], 1).robust
    res = check(dra, "last_letter", [1, 2, 3], Fraction(3, 2))
    assert not res.robust and res.infimum == 1 and 1 < res.cost < Fraction(3, 2)
    assert res.witness[:2] == [1.0, 2.0] and res.witness[2] < 2


def test_hamming_one_change():
    dra = increasing()
    assert check(dra, "hamming", [1, 2, 3], 1).robust          # D < 1: w = v only
    res = check(dra, "hamming", [1, 2, 3], Fraction(3, 2))     # D <= 1
    assert not res.robust and res.cost == 1 and len(res.witness) == 3


def test_dtw_stretching_breaks_strict_increase():
    # w = 1,2,2,3 has DTW distance 0 to v but is not strictly increasing.
    res = check(increasing(), "dtw", [1, 2, 3], Fraction(1, 10))
    assert not res.robust and res.cost == 0 and len(res.witness) > 3


def test_dtw_insertion_slightly_below_breaks_nonstrict_increase():
    # Repeating 1 slightly lowered (1, 1-e, 2, 3) costs only e under DTW, so
    # the infimum is 0 but not attained.
    res = check(increasing(strict=False), "dtw", [1, 2, 3], Fraction(1, 2))
    assert not res.robust and res.infimum == 0 and 0 < res.cost < Fraction(1, 2)


def test_rejected_sample_flips_into_language():
    # v = 3,1 is rejected by "all < 3"; lowering the first letter to just
    # below 3 costs more than 0 but arbitrarily little (strict guard).
    dra = below_three()
    res = check(dra, "l1", [3, 1], Fraction(1, 100))
    assert not res.robust and 0 < res.cost < Fraction(1, 100)
    assert check(dra, "hamming", [5, 4], Fraction(3, 2)).robust  # two changes needed


def test_register_equality_and_constants_from_v():
    dra = contains_equal_pair()
    v = [1, 5, 1]
    assert _CompiledDRA(dra, set()).run([Fraction(x) for x in v])
    res = check(dra, "hamming", v, Fraction(3, 2))
    assert not res.robust and res.cost == 1


def test_edit_insertion_and_deletion():
    dra = increasing()
    res = check(dra, "edit", [1, 2, 3], Fraction(3, 2))
    assert not res.robust and res.cost == 1


# --- brute-force cross-check -------------------------------------------------

def _grid(v, dra_consts):
    pts = sorted({Fraction(x) for x in v} | set(dra_consts) | {Fraction(0)})
    extra = set(pts)
    for a, b in zip(pts, pts[1:]):
        extra.add((a + b) / 2)
    extra.add(pts[0] - 1)
    extra.add(pts[-1] + 1)
    return sorted(extra)


@pytest.mark.parametrize("metric", sorted(METRICS))
@pytest.mark.parametrize("dra_factory,consts", [
    (increasing, []), (lambda: increasing(False), []),
    (below_three, [Fraction(3)]), (contains_equal_pair, []),
])
@pytest.mark.parametrize("v", [[1, 2, 3], [3, 1], [2, 2, 5]])
@pytest.mark.parametrize("delta", [Fraction(1, 2), Fraction(3, 2)])
def test_brute_force_witnesses_are_found(metric, dra_factory, consts, v, delta):
    """Every grid witness found by brute force is matched by the checker,
    and every checker witness is (re)verified concretely."""
    dra = dra_factory()
    raa = _CompiledRAA(METRICS[metric](), set())
    cdra = _CompiledDRA(dra, set())
    vf = [Fraction(x) for x in v]
    accepted = cdra.run(vf)
    grid = _grid(v, consts)
    brute = None
    for n in range(1, len(v) + 2):
        for w in itertools.product(grid, repeat=n):
            if cdra.run(list(w)) != accepted:
                c = raa.min_cost(vf, list(w))
                if c < delta and (brute is None or c < brute):
                    brute = c
    res = check(dra, metric, v, delta)
    if brute is not None:
        assert not res.robust, f"missed witness of cost {brute}"
        assert res.infimum <= brute
    if not res.robust:
        assert res.cost < delta
        w = [Fraction(repr(x)) for x in res.witness]
        assert cdra.run(w) != accepted
        assert raa.min_cost(vf, w) == res.cost


# --- metric RAAs compute their distances exactly ----------------------------

def _levenshtein(a, b):
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return d[len(a)][len(b)]


def _dtw(a, b):
    inf = float("inf")
    d = [[inf] * (len(b) + 1) for _ in range(len(a) + 1)]
    d[0][0] = 0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = abs(a[i - 1] - b[j - 1]) + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[len(a)][len(b)]


def test_edit_raa_handles_operations_at_the_ends():
    raa = _CompiledRAA(create_edit_raa(), set())
    f = lambda xs: [Fraction(x) for x in xs]
    assert raa.min_cost(f([1, 2, 3]), f([1, 2])) == 1      # trailing deletion
    assert raa.min_cost(f([1, 2]), f([1, 2, 3])) == 1      # trailing insertion
    assert raa.min_cost(f([1, 2, 3]), f([2, 3])) == 1      # leading deletion
    assert raa.min_cost(f([5]), f([1, 2, 3])) == 3


def test_edit_and_dtw_raas_match_dynamic_programs():
    import random
    rng = random.Random(1)
    edit = _CompiledRAA(create_edit_raa(), set())
    dtw = _CompiledRAA(create_dtw_raa(), set())
    for _ in range(300):
        v = [Fraction(rng.randint(0, 3)) for _ in range(rng.randint(1, 5))]
        w = [Fraction(rng.randint(0, 3)) for _ in range(rng.randint(1, 5))]
        assert edit.min_cost(v, w) == _levenshtein(w, v), (v, w)
        assert dtw.min_cost(v, w) == _dtw(w, v), (v, w)


def test_checker_finds_trailing_edit_witness():
    # "length exactly 3": v = 1,2,3 flips only by changing the length, which
    # one trailing deletion or insertion does at cost 1.
    ts = [Transition(curr, k, True, {}, k + 1) for k in range(3)]
    dra = RegisterAutomata(4, ts, [], [3])
    res = check(dra, "edit", [1, 2, 3], Fraction(3, 2))
    assert not res.robust and res.cost == 1 and len(res.witness) in (2, 4)


@pytest.mark.parametrize("direction", [-1, 1])
def test_registered_raa_preserves_pending_letter_and_constant_updates(direction):
    # First read w[0] while only advancing v, then consume that same pending
    # letter.  A register update fixes w[1] to 1/2.  The strict first guard
    # requires an infinitesimal rank of either sign at the cheapest base.
    c1, c2, metric_reg = z3.Reals("curr1 curr2 r_metric")
    first_guard = (z3.And(c1 > c2, c1 < 1) if direction > 0 else
                   z3.And(c1 < c2, c1 > -1))
    transitions = [
        RAA_Transition(c1, c2, 0, first_guard,
                       {"r_metric": "curr1"}, 1, [0, 0, 0], 2),
        RAA_Transition(c1, c2, 1, c1 == metric_reg,
                       {"r_metric": Fraction(1, 2)}, 2, [0, 0, 0], 1),
        RAA_Transition(c1, c2, 2, c1 == metric_reg,
                       {}, 3, [0, 0, 0], 3),
    ]
    raa = RAA([0, 1, 2, 3], 0, [3], transitions, [metric_reg])
    dra = RegisterAutomata(1, [Transition(curr, 0, curr == 0,
                                        {"r0": "curr"}, 0)], [r0], [0])
    res = find_witness(dra, raa, [0, 0], Fraction(1, 2), True, timeout=10)
    assert not res.robust and res.cost == res.infimum == 0
    assert len(res.witness) == 2 and res.witness[1] == 0.5
    assert 0 < direction * res.witness[0] < 1
    assert not _CompiledDRA(dra, set()).run(res.witness)
    assert _CompiledRAA(raa, set()).min_cost([0, 0], res.witness) == 0
