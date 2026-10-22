"""Exact local-robustness checking for DRAs (Lemmas 1 and 4 of the paper).

Given a DRA ``A``, a sequence ``v``, a perturbation-cost RAA ``A_D`` and a
radius ``delta``, :func:`find_witness` decides whether some sequence ``w``
with ``A_D(v, w) < delta`` is accepted by a *flip automaton* (the complement
of ``A`` when ``v`` is accepted, ``A`` itself otherwise), and if so returns a
concrete, verified witness.

Conventions (as in ``metric_automata``): head 1 of the RAA reads the witness
``w`` (``curr1``), head 2 reads the fixed sequence ``v`` (``curr2``), and
``move`` is 1, 2 or 3 (= both).  The cost of a transition is
``a1 * curr1 + a2 * curr2 + b`` for ``cost_function = [a1, a2, b]``.

The search follows the proof of Lemma 4:

* The product of ``A_D`` and the flip automaton is explored on the fly.  The
  DRA component reads a witness letter only when head 1 advances, so
  ``move = 2`` transitions, which reread the current letter of ``w``, do not
  feed it to the DRA twice.
* A vertex records the RAA state, the DRA state, the position in ``v``, live
  register values, and the *pending* value of the current letter of ``w``
  once it has been read, so repeated reads of one letter agree.
  Backward liveness follows register copies and overwrites; dead registers
  are removed globally or reset before normalising a destination vertex.
* Values are *tagged*: a base constant from the finite set ``C`` plus an
  infinitesimal rank.  Values sharing a base are ordered by rank (rank 0 is
  the constant itself), which realises every strict and disequality atom.
  Edge weights are evaluated at the base values, so the shortest path weight
  is the infimum of the cost; a concrete witness is then realised close to
  the bases and checked concretely.
* Dijkstra's algorithm runs on the fly and never expands a vertex whose
  distance is already ``>= delta``.

Guards must be Boolean combinations of order atoms (``<, <=, ==, !=, >=, >``)
between variables and rational constants; arithmetic inside guards is
rejected with ``NotImplementedError``.
"""

import heapq
import copy
import itertools
import numbers
import time
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import z3


# ---------------------------------------------------------------------------
# Guard and assignment compilation
# ---------------------------------------------------------------------------

def _to_fraction(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, bool):
        raise TypeError("boolean is not a data value")
    if isinstance(x, numbers.Integral):  # int and numpy integers
        return Fraction(int(x))
    if isinstance(x, float):
        # z3 reads a Python float through its decimal repr; do the same so
        # that data values agree with the DRA's own z3-based semantics
        return Fraction(repr(x))
    if isinstance(x, z3.ExprRef) and z3.is_rational_value(x):
        return Fraction(x.numerator_as_long(), x.denominator_as_long())
    if isinstance(x, z3.ExprRef) and z3.is_int_value(x):
        return Fraction(x.as_long())
    if isinstance(x, z3.ExprRef) and z3.is_algebraic_value(x):
        raise NotImplementedError(f"irrational constant {x}")
    if isinstance(x, str):
        return Fraction(x)
    if isinstance(x, numbers.Real):  # e.g. numpy float32
        return Fraction(repr(float(x)))
    raise TypeError(f"cannot convert {x!r} to a rational")


_CMP = {
    z3.Z3_OP_LT: lambda a, b: a < b,
    z3.Z3_OP_LE: lambda a, b: a <= b,
    z3.Z3_OP_GT: lambda a, b: a > b,
    z3.Z3_OP_GE: lambda a, b: a >= b,
    z3.Z3_OP_EQ: lambda a, b: a == b,
}


def _compile_term(e, consts: set):
    """Compile a data term (variable or rational constant) to env -> value."""
    if z3.is_rational_value(e) or z3.is_int_value(e):
        c = _to_fraction(e)
        consts.add(c)
        return lambda env, c=c: c
    if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
        name = e.decl().name()
        return lambda env, name=name: env[name]
    if z3.is_app(e) and e.decl().kind() == z3.Z3_OP_TO_REAL:
        return _compile_term(e.arg(0), consts)
    if z3.is_app(e) and e.decl().kind() == z3.Z3_OP_UMINUS:
        inner = e.arg(0)
        if z3.is_rational_value(inner) or z3.is_int_value(inner):
            c = -_to_fraction(inner)
            consts.add(c)
            return lambda env, c=c: c
    raise NotImplementedError(f"unsupported term in guard: {e}")


def compile_guard(g, consts: set):
    """Compile a guard (z3 formula, bool, or 'True'/'False') to env -> bool.

    Every rational constant met in the guard is added to ``consts``.
    """
    if g is True or g is None:
        return lambda env: True
    if g is False:
        return lambda env: False
    if isinstance(g, str):
        s = g.strip()
        if s == "True":
            return lambda env: True
        if s == "False":
            return lambda env: False
        raise NotImplementedError(f"string guard {g!r}")
    if z3.is_true(g):
        return lambda env: True
    if z3.is_false(g):
        return lambda env: False
    k = g.decl().kind()
    if k == z3.Z3_OP_AND:
        parts = [compile_guard(c, consts) for c in g.children()]
        return lambda env: all(p(env) for p in parts)
    if k == z3.Z3_OP_OR:
        parts = [compile_guard(c, consts) for c in g.children()]
        return lambda env: any(p(env) for p in parts)
    if k == z3.Z3_OP_NOT:
        p = compile_guard(g.arg(0), consts)
        return lambda env: not p(env)
    if k == z3.Z3_OP_DISTINCT:
        terms = [_compile_term(c, consts) for c in g.children()]
        def distinct(env):
            vals = [t(env) for t in terms]
            return len(set(vals)) == len(vals)
        return distinct
    if k in _CMP:
        a = _compile_term(g.arg(0), consts)
        b = _compile_term(g.arg(1), consts)
        op = _CMP[k]
        return lambda env: op(a(env), b(env))
    raise NotImplementedError(f"unsupported guard: {g}")


def _compile_assignment(assignment: dict, letter_names: Tuple[str, ...], consts: set):
    """Compile a register assignment to a list of (register, env -> value).

    ``letter_names`` are the names meaning "copy the current letter"
    (``curr``/``c`` for DRAs; ``curr1``/``c1`` and ``curr2``/``c2`` for RAAs,
    resolved by the caller through the environment).
    """
    out = []
    for reg, val in assignment.items():
        reg = str(reg)
        if isinstance(val, str):
            if val in letter_names:
                out.append((reg, lambda env, n=val: env[n]))
            elif val.startswith("r"):
                out.append((reg, lambda env, n=val: env[n]))
            else:
                c = _to_fraction(val)
                consts.add(c)
                out.append((reg, lambda env, c=c: c))
        elif isinstance(val, (int, float, Fraction)) and not isinstance(val, bool):
            c = _to_fraction(val)
            consts.add(c)
            out.append((reg, lambda env, c=c: c))
        elif isinstance(val, z3.ExprRef):
            out.append((reg, _compile_term(val, consts)))
        else:
            raise NotImplementedError(f"unsupported assignment {reg} := {val!r}")
    return out


def _mentions(g, assignment) -> set:
    """Names of the variables read by a guard and an assignment."""
    names = set()

    def walk(e):
        if isinstance(e, z3.ExprRef):
            if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                names.add(e.decl().name())
            for c in e.children():
                walk(c)

    walk(g)
    for val in assignment.values():
        if isinstance(val, str):
            names.add(val)
        else:
            walk(val)
    return names


def _flow(t):
    """Register reads and simultaneous writes for backward liveness."""
    return (t.start_state, _mentions(t.guard, {}),
            {str(r): _mentions(None, {r: value})
             for r, value in t.assignment.items()}, t.end_state)


def _liveness(flows, regs, states, always_live=()):
    """Conservative fixed point, including copy dependencies and loops.

    Every outgoing guard matters under first-match semantics.  Only writes
    whose results are live at the destination need their right-hand sides.
    """
    regs = set(regs)
    pinned = regs.intersection(always_live)
    live = {q: set(pinned) for q in states}
    changed = True
    while changed:
        changed = False
        for src, reads, writes, dst in reversed(flows):
            needed = (reads & regs) | (live[dst] - writes.keys())
            for reg in live[dst].intersection(writes):
                needed.update(writes[reg] & regs)
            if not needed.issubset(live[src]):
                live[src].update(needed)
                changed = True
    return {q: frozenset(rs) for q, rs in live.items()}


class _CompiledDRA:
    """A DRA over one letter variable ``curr``, first-match semantics."""

    def __init__(self, dra, consts: set):
        self.states = set(range(dra.states))
        self.regs = [str(r) for r in dra.register_vars]
        self.final = set(dra.final_states)
        self.by_state: Dict[int, list] = {}
        letter = ("curr", "c")
        self.flows = []
        for t in dra.transitions:
            g = compile_guard(t.guard, consts)
            a = _compile_assignment(t.assignment, letter, consts)
            self.by_state.setdefault(t.start_state, []).append((g, a, t.end_state))
            self.flows.append(_flow(t))

    def search_copy(self, complement=False, prune_registers=True, always_live=()):
        """Prepare a search view without recompiling guards or mutating self."""
        out = copy.copy(self)
        out.states = set(self.states)
        out.by_state = {q: list(ts) for q, ts in self.by_state.items()}
        out.flows = list(self.flows)
        if complement:
            # An unconditional last transition implements rejection when no
            # original guard matched, including for incomplete DRAs.
            sink = max(out.states, default=-1) + 1
            out.states.add(sink)
            for q in out.states:
                out.by_state.setdefault(q, []).append((lambda env: True, [], sink))
                out.flows.append((q, set(), {}, sink))
            out.final = out.states - self.final
        out.live = (_liveness(out.flows, self.regs, out.states, always_live)
                    if prune_registers else
                    {q: frozenset(self.regs) for q in out.states})
        used = set().union(*out.live.values())
        out.regs = [r for r in self.regs if r in used]
        out.by_state = {
            q: [(g, [(r, f) for r, f in a if r in out.live[dst]], dst)
                for g, a, dst in ts]
            for q, ts in out.by_state.items()
        }
        return out

    def step(self, state, regs: dict, letter):
        """Return (next_state, new_regs) or None if no transition is enabled."""
        env = dict(regs)
        env["curr"] = letter
        env["c"] = letter
        for g, a, q in self.by_state.get(state, ()):
            if g(env):
                new = dict(regs)
                for reg, f in a:
                    new[reg] = f(env)
                return q, new
        return None

    def run(self, word) -> bool:
        state, regs = 0, {r: Fraction(0) for r in self.regs}
        for x in word:
            nxt = self.step(state, regs, x)
            if nxt is None:
                return False
            state, regs = nxt
        return state in self.final


class _CompiledRAA:
    def __init__(self, raa, consts: set):
        self.regs = [str(r) for r in raa.register_vars]
        self.initial = raa.initial_state
        self.final = set(raa.final_states)
        self.transitions = []
        self.flows = []
        letters = ("curr1", "c1", "curr2", "c2")
        for idx, t in enumerate(raa.transitions):
            if t.move not in (1, 2, 3):
                raise ValueError(f"unknown move {t.move}")
            g = compile_guard(t.guard, consts)
            a = _compile_assignment(t.assignment, letters, consts)
            a1, a2, b = (_to_fraction(x) for x in t.cost_function)
            # A transition needs the letter under a head iff it reads that
            # letter (guard, assignment or cost) or advances that head.  This
            # lets one-sided steps such as insertions and deletions fire once
            # the other sequence is exhausted.
            used = _mentions(t.guard, t.assignment)
            needs1 = bool(used & {"curr1", "c1"}) or a1 != 0 or t.move in (1, 3)
            needs2 = bool(used & {"curr2", "c2"}) or a2 != 0 or t.move in (2, 3)
            self.transitions.append(
                (idx, t.start_state, g, a, t.end_state, t.move, a1, a2, b,
                 needs1, needs2))
            self.flows.append(_flow(t))
        self.by_state: Dict[int, list] = {}
        for tr in self.transitions:
            self.by_state.setdefault(tr[1], []).append(tr)

    def search_copy(self, prune_registers=True, always_live=()):
        out = copy.copy(self)
        states = {self.initial} | self.final
        for src, _, _, dst in self.flows:
            states.update((src, dst))
        out.live = (_liveness(self.flows, self.regs, states, always_live)
                    if prune_registers else
                    {q: frozenset(self.regs) for q in states})
        used = set().union(*out.live.values())
        out.regs = [r for r in self.regs if r in used]
        # Retain the original head requirements even for removed assignments:
        # a dead write must not enable a transition past the end of a tape.
        out.transitions = [
            tr[:3] + ([(r, f) for r, f in tr[3] if r in out.live[tr[4]]],) + tr[4:]
            for tr in self.transitions
        ]
        out.by_state = {}
        for tr in out.transitions:
            out.by_state.setdefault(tr[1], []).append(tr)
        return out

    def min_cost(self, v: Sequence[Fraction], w: Sequence[Fraction], deadline=None):
        """Exact minimum accepting-run cost of (w, v); inf if no run."""
        start = (self.initial, 0, 0, tuple(Fraction(0) for _ in self.regs))
        best = {start: Fraction(0)}
        heap = [(Fraction(0), 0, start)]
        tie = itertools.count(1)
        while heap:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("robustness check timed out")
            d, _, key = heapq.heappop(heap)
            if best.get(key, None) != d:
                continue
            q, h1, h2, regs = key
            if q in self.final and h1 == len(w) and h2 == len(v):
                return d
            env = dict(zip(self.regs, regs))
            if h1 < len(w):
                env.update(curr1=w[h1], c1=w[h1])
            if h2 < len(v):
                env.update(curr2=v[h2], c2=v[h2])
            for (_, _, g, a, q2, mv, a1, a2, b, n1, n2) in self.by_state.get(q, ()):
                if (n1 and h1 >= len(w)) or (n2 and h2 >= len(v)):
                    continue
                if not g(env):
                    continue
                cost = (a1 * w[h1] if a1 else 0) + (a2 * v[h2] if a2 else 0) + b
                new = dict(zip(self.regs, regs))
                for reg, f in a:
                    new[reg] = f(env)
                nh1 = h1 + (1 if mv in (1, 3) else 0)
                nh2 = h2 + (1 if mv in (2, 3) else 0)
                k2 = (q2, nh1, nh2, tuple(new[r] for r in self.regs))
                nd = d + cost
                if nd < best.get(k2, float("inf")):
                    best[k2] = nd
                    heapq.heappush(heap, (nd, next(tie), k2))
        return float("inf")


# ---------------------------------------------------------------------------
# Tagged values
# ---------------------------------------------------------------------------
# A tagged value is (b, r): base index b into the sorted constant list and an
# infinitesimal rank r.  (b, 0) is the constant itself; values with the same
# base are ordered by r.  Ranks are renormalised after every step so that the
# vertex key only records the order type.

def _normalise(values: List[Optional[Tuple[int, Fraction]]]):
    """Renormalise ranks per base, keeping 0 fixed; None entries pass through."""
    by_base: Dict[int, set] = {}
    for x in values:
        if x is not None:
            by_base.setdefault(x[0], set()).add(x[1])
    remap = {}
    for b, ranks in by_base.items():
        neg = sorted(r for r in ranks if r < 0)
        pos = sorted(r for r in ranks if r > 0)
        for i, r in enumerate(reversed(neg)):
            remap[(b, r)] = -(i + 1)
        for i, r in enumerate(pos):
            remap[(b, r)] = i + 1
        remap[(b, 0)] = 0
    return [None if x is None else (x[0], remap[(x[0], x[1])]) for x in values]


class _Search:
    def __init__(self, flip_dra, raa, v, delta, timeout, extra_consts=()):
        self.deadline = None if timeout is None else time.monotonic() + timeout
        consts: set = {Fraction(0)}
        self.dra = (flip_dra if isinstance(flip_dra, _CompiledDRA) else
                    _CompiledDRA(flip_dra, consts))
        self.raa = raa if isinstance(raa, _CompiledRAA) else _CompiledRAA(raa, consts)
        self.v = [_to_fraction(x) for x in v]
        self.delta = _to_fraction(delta)
        consts.update(self.v)
        consts.update(_to_fraction(c) for c in extra_consts)
        # alpha-bound of Lemma 1, so that +-alpha belong to C as in the paper
        alpha = max((abs(c) for c in consts), default=Fraction(0))
        for (_, _, _, _, _, _, a1, a2, b, _, _) in self.raa.transitions:
            if a1 != 0:
                for vi in self.v:
                    bb = a2 * vi + b
                    alpha = max(alpha, abs(bb / a1), abs((self.delta - bb) / a1))
        alpha += 1
        consts.update({alpha, -alpha})
        self.C = sorted(consts)
        self.index = {c: i for i, c in enumerate(self.C)}
        gaps = [b - a for a, b in zip(self.C, self.C[1:])]
        self.mingap = min(gaps) if gaps else Fraction(1)
        if set(self.dra.regs) & set(self.raa.regs):
            raise ValueError("DRA and RAA register names must be disjoint")
        self.regs = self.dra.regs + self.raa.regs
        self.reg_index = {r: i for i, r in enumerate(self.regs)}
        self.dra_reg_index = {r: i for i, r in enumerate(self.dra.regs)}
        self.dead_dra = {
            q: tuple(i for i, r in enumerate(self.dra.regs) if r not in live)
            for q, live in getattr(self.dra, "live", {}).items()
        }
        self.dead_raa = {
            q: tuple(len(self.dra.regs) + i for i, r in enumerate(self.raa.regs)
                     if r not in live)
            for q, live in getattr(self.raa, "live", {}).items()
        }
        self.n_live = len(self.regs) + 1  # registers and the pending letter
        self.eta = self.mingap / (4 * (self.n_live + 3))
        # _val only receives normalised tags.  At most n_live values are
        # present, so all their integer ranks fit in this finite table.
        self.tag_values = {
            (b, r): c + r * self.eta
            for b, c in enumerate(self.C)
            for r in range(-self.n_live, self.n_live + 1)
        }

    def _forget_dead(self, regs, q, p, zero):
        for i in self.dead_dra.get(p, ()):
            regs[i] = zero
        for i in self.dead_raa.get(q, ()):
            regs[i] = zero

    def _check_timeout(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("robustness check timed out")

    # concrete stand-in used only to evaluate order atoms on tagged values
    def _val(self, x):
        return self.tag_values[x]

    def _slots(self, live: List[Tuple[int, int]]):
        """All order positions for a fresh letter: (base, rank).

        Exact positions (a constant, or equal to a live value) come first, so
        that on equal path weight Dijkstra prefers witnesses that attain the
        cost rather than approach it.
        """
        at_base: Dict[int, set] = {}
        for b, r in live:
            at_base.setdefault(b, set()).add(r)
        ranks_of = {b: sorted(at_base.get(b, set()) | {0}) for b in range(len(self.C))}
        for b, ranks in ranks_of.items():
            for r in ranks:
                yield (b, Fraction(r))
        for b, ranks in ranks_of.items():
            yield (b, Fraction(ranks[0]) - Fraction(1, 2))
            for r1, r2 in zip(ranks, ranks[1:]):
                yield (b, Fraction(r1 + r2, 2))
            yield (b, Fraction(ranks[-1]) + Fraction(1, 2))

    def run(self):
        self._check_timeout()
        if self.delta <= 0:
            return None, None, 0
        nd = len(self.dra.regs)
        zero = (self.index[Fraction(0)], 0)
        source = (self.raa.initial, 0, 0, tuple([zero] * len(self.regs)), None)
        dist = {source: (Fraction(0), 0)}
        parent: Dict[tuple, tuple] = {}
        tie = itertools.count(1)
        heap = [(Fraction(0), 0, 0, source)]
        expanded = 0
        while heap:
            d, inexact, _, key = heapq.heappop(heap)
            if dist.get(key) != (d, inexact):
                continue
            expanded += 1
            self._check_timeout()
            q, p, i, regs, pend = key
            if (pend is None and i == len(self.v) and q in self.raa.final
                    and p in self.dra.final):
                return d, self._path(parent, key), expanded
            has_v = i < len(self.v)
            letter_choices = None  # fresh-letter slots, computed on demand
            # Several RAA transitions inspect the same letter (in particular
            # the three DTW head moves).  Preparation depends only on the
            # source vertex and letter, and DRA stepping also on p, so reuse
            # both within this expansion without retaining a global cache.
            prepared = {}
            dra_steps = {}
            for (tidx, _, g, a, q2, mv, a1, a2, b, n1, n2) in self.raa.by_state.get(q, ()):
                if n2 and not has_v:
                    continue
                if n1:
                    # the step reads the current witness letter: reuse the
                    # pending one, or choose a fresh one
                    if letter_choices is None:
                        live = list(regs) + ([pend] if pend is not None else [])
                        letter_choices = [pend] if pend is not None else list(self._slots(live))
                    choices = letter_choices
                else:
                    choices = [None]  # the step reads no witness letter
                for c in choices:
                    self._check_timeout()
                    if c not in prepared:
                        vals = _normalise(list(regs) + [c if c is not None else pend])
                        tregs, tc = vals[:-1], vals[-1]
                        env = {r: self._val(x) for r, x in zip(self.regs, tregs)}
                        tag_env = dict(zip(self.regs, tregs))
                        if c is not None:
                            cval = self._val(tc)
                            env.update(curr1=cval, c1=cval)
                            tag_env.update(curr1=tc, c1=tc)
                        if has_v:
                            vtag = (self.index[self.v[i]], 0)
                            env.update(curr2=self.v[i], c2=self.v[i])
                            tag_env.update(curr2=vtag, c2=vtag)
                        prepared[c] = tregs, tc, env, tag_env
                    tregs, tc, env, tag_env = prepared[c]
                    if not g(env):
                        continue
                    w_edge = ((a1 * self.C[tc[0]]) if a1 else 0) \
                        + ((a2 * self.v[i]) if a2 else 0) + b
                    if w_edge < 0:
                        raise ValueError(
                            f"negative accumulator increment {w_edge} on RAA "
                            f"transition {tidx}; Dijkstra needs non-negative costs")
                    ndist = d + w_edge
                    if ndist >= self.delta:
                        continue
                    # RAA register update (tagged values are copied)
                    new_regs = list(tregs)
                    for reg, f in a:
                        val = f(tag_env)
                        if isinstance(val, Fraction):  # constant assignment
                            val = (self.index[val], 0)
                        new_regs[self.reg_index[reg]] = val
                    p2 = p
                    if mv in (1, 3):  # head 1 consumes the letter: the DRA reads it
                        if c not in dra_steps:
                            dra_steps[c] = self._dra_step(p, tregs[:nd], tc)
                        step = dra_steps[c]
                        if step is None:
                            continue
                        p2, dra_new = step
                        new_regs[:nd] = dra_new
                    pend2 = None if mv in (1, 3) else tc
                    i2 = i + (1 if mv in (2, 3) else 0)
                    self._forget_dead(new_regs, q2, p2, zero)
                    norm = _normalise(new_regs + [pend2])
                    k2 = (q2, p2, i2, tuple(norm[:-1]), norm[-1])
                    # prefer exact letters on equal weight, so attained
                    # minima are returned exactly
                    ninexact = inexact + (1 if c is not None and c[1] != int(c[1]) else 0)
                    if (ndist, ninexact) < dist.get(k2, (self.delta, 0)):
                        dist[k2] = (ndist, ninexact)
                        parent[k2] = (key, tidx, c, mv)
                        heapq.heappush(heap, (ndist, ninexact, next(tie), k2))
        self._check_timeout()
        return None, None, expanded

    def _dra_step(self, state, tregs, tc):
        """DRA step on tagged values; returns (state, tagged registers)."""
        env = {r: self._val(x) for r, x in zip(self.dra.regs, tregs)}
        env["curr"] = env["c"] = self._val(tc)
        for g, a, q in self.dra.by_state.get(state, ()):
            if g(env):
                tag_env = dict(zip(self.dra.regs, tregs))
                tag_env["curr"] = tag_env["c"] = tc
                new = list(tregs)
                for reg, f in a:
                    val = f(tag_env)
                    if isinstance(val, Fraction):
                        val = (self.index[val], 0)
                    new[self.dra_reg_index[reg]] = val
                return q, new
        return None

    def _path(self, parent, key):
        steps = []
        while key in parent:
            prev, tidx, c, mv = parent[key]
            steps.append((prev, tidx, c, mv))
            key = prev
        steps.reverse()
        return steps

    def realise(self, base_cost, steps):
        """Turn a tagged path into concrete rational letters of w.

        Each fresh letter is placed strictly between the concrete values of its
        live neighbours at the same base, so every order atom along the path
        keeps its truth value.  All letters stay within eps0 of their base, so
        the realised cost exceeds the base cost by less than (delta-cost)/4.
        """
        slack = self.delta - base_cost
        total_a = sum(abs(self.raa.transitions[t][6]) for (_, t, _, _) in steps)
        self._eps_cap = min(self.mingap / 4, slack / (4 * (1 + total_a)))
        letters: List[Fraction] = []
        conc: Dict[int, Fraction] = {}  # register index -> concrete value
        zero = Fraction(0)
        for j in range(len(self.regs)):
            conc[j] = zero
        pending_conc = None
        for (prev, tidx, c, mv) in steps:
            self._check_timeout()
            _, _, i, regs, pend = prev
            if c is None or pend is not None:
                cv = pending_conc  # no fresh letter (None if none is pending)
            else:
                cv = self._place(c, regs, conc)
            # replay the step on concrete values using the tagged order
            tr = self.raa.transitions[tidx]
            a = tr[3]
            env = {self.regs[j]: conc[j] for j in range(len(self.regs))}
            if cv is not None:
                env.update(curr1=cv, c1=cv)
            if i < len(self.v):
                env.update(curr2=self.v[i], c2=self.v[i])
            new = dict(conc)
            for reg, f in a:
                new[self.reg_index[reg]] = f(env)
            p2 = prev[1]
            if mv in (1, 3):
                nd = len(self.dra.regs)
                dregs = {self.dra.regs[j]: conc[j] for j in range(nd)}
                step = self.dra.step(prev[1], dregs, cv)
                if step is None:
                    raise AssertionError("realised witness leaves the DRA")
                p2, dnew = step
                for j in range(nd):
                    new[j] = dnew[self.dra.regs[j]]
                letters.append(cv)
                pending_conc = None
            else:
                pending_conc = cv
            # Match the canonical register tuple stored in the next vertex.
            # Dead values cannot affect a later guard before being overwritten.
            self._forget_dead(new, tr[4], p2, zero)
            conc = new
        return letters

    def _place(self, c, regs, conc):
        """Concrete value for a fresh letter at tagged slot c = (b, rank)."""
        b, rank = c
        base = self.C[b]
        if rank == int(rank) and rank == 0:
            return base
        lower, upper = base - self._eps_cap, base + self._eps_cap
        below = [conc[j] for j, x in enumerate(regs) if x[0] == b and x[1] < rank]
        above = [conc[j] for j, x in enumerate(regs) if x[0] == b and x[1] > rank]
        equal = [conc[j] for j, x in enumerate(regs) if x[0] == b and x[1] == rank]
        if equal:
            return equal[0]
        if rank > 0:
            below.append(base)
        else:
            above.append(base)
        lo = max(below) if below else lower
        hi = min(above) if above else upper
        return (lo + hi) / 2


class RobustnessResult:
    """Outcome of one local-robustness query."""

    def __init__(self, robust, witness=None, cost=None, infimum=None,
                 expanded=0, seconds=0.0):
        self.robust = robust          # True: no flip within delta
        self.witness = witness        # list of floats, if not robust
        self.cost = cost              # concrete A_D(v, witness) < delta
        self.infimum = infimum        # shortest-path weight (infimum cost)
        self.expanded = expanded
        self.seconds = seconds

    def __repr__(self):
        if self.robust:
            return f"RobustnessResult(robust, expanded={self.expanded})"
        return (f"RobustnessResult(non-robust, cost={float(self.cost):.6g}, "
                f"infimum={float(self.infimum):.6g}, witness={self.witness})")


class RobustnessChecker:
    """Reusable compiled checker for one DRA and perturbation RAA.

    Compilation captures the automata at construction time; construct a new
    checker after changing them.  Query-specific constants, paths and search
    caches are discarded after each call.  ``prune_registers=False`` retains
    every register for differential checking and benchmarking.
    """

    def __init__(self, dra, raa, *, prune_registers=True):
        consts = {Fraction(0)}
        self.original = _CompiledDRA(dra, consts)
        self.metric = _CompiledRAA(raa, consts)
        if set(self.original.regs) & set(self.metric.regs):
            raise ValueError("DRA and RAA register names must be disjoint")
        # Conservatively pin any value read by the other component.
        def reads(flows):
            names = set()
            for _, guard_reads, writes, _ in flows:
                names.update(guard_reads)
                for rhs in writes.values():
                    names.update(rhs)
            return names
        self.raa = self.metric.search_copy(prune_registers, reads(self.original.flows))
        self.targets = {
            accepted: self.original.search_copy(
                complement=accepted, prune_registers=prune_registers,
                always_live=reads(self.metric.flows))
            for accepted in (False, True)
        }
        self.consts = frozenset(consts)

    def accepts(self, word):
        """Classify a sequence using the compiled original DRA."""
        return self.original.run([_to_fraction(x) for x in word])

    def find_witness(self, v, delta, accepted=None, timeout=None):
        """Find a verified flip at cost strictly below delta, or prove none.

        ``accepted`` may provide the known classification of v.  A timeout
        raises ``TimeoutError`` and never returns a robustness verdict.
        """
        t0 = time.monotonic()
        if timeout is not None and timeout <= 0:
            raise TimeoutError("robustness check timed out")
        if accepted is None:
            accepted = self.accepts(v)
        remaining = None if timeout is None else timeout - (time.monotonic() - t0)
        search = _Search(self.targets[bool(accepted)], self.raa, v, delta,
                         remaining, extra_consts=self.consts)
        infimum, steps, expanded = search.run()
        if infimum is None:
            return RobustnessResult(True, expanded=expanded, seconds=time.monotonic() - t0)
        letters = search.realise(infimum, steps)
        # Verify against the full original automata and the float sequence
        # callers receive, independently of register pruning in the search.
        witness = [float(x) for x in letters]
        w_check = [_to_fraction(x) for x in witness]
        cost = self.metric.min_cost(search.v, w_check, deadline=search.deadline)
        search._check_timeout()
        if not cost < search.delta:
            raise AssertionError(f"realised witness has cost {cost} >= delta")
        if self.original.run(w_check) == accepted:
            raise AssertionError("realised witness does not flip the DRA")
        search._check_timeout()
        return RobustnessResult(False, witness=witness, cost=cost,
                                infimum=infimum, expanded=expanded,
                                seconds=time.monotonic() - t0)


def find_witness(dra, raa, v, delta, accepted: bool, timeout: Optional[float] = None):
    """One-shot exact query; use RobustnessChecker to check many sequences.

    A witness has RAA cost strictly below delta and flips the DRA verdict.
    Raises ``TimeoutError`` if the budget, including compilation, is exhausted.
    """
    t0 = time.monotonic()
    if timeout is not None and timeout <= 0:
        raise TimeoutError("robustness check timed out")
    checker = RobustnessChecker(dra, raa)
    remaining = None if timeout is None else timeout - (time.monotonic() - t0)
    result = checker.find_witness(v, delta, accepted, timeout=remaining)
    result.seconds = time.monotonic() - t0
    return result
