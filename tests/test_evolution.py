"""Multi-component evolution (forward synthesis network).

All tests drive the evolver with a *fake* forward model, so they run everywhere
without the GNN checkpoint or a GPU. The fake model maps a set of reactant SMILES
to a fixed product list, which lets us assert the exact scores, depths, score
propagation and memory/disk equivalence.
"""

from __future__ import annotations

import pytest

from synomega.chem.mol import canonicalize as C
from synomega.forward import MultiComponentEvolution
from synomega.forward.base import ForwardModel, ForwardPrediction

# Real, distinct, parseable molecules used as graph nodes.
A, B, E = "CCO", "CCN", "CCC"          # sources
X, Z, Y = "CCCO", "CCCN", "CCCCO"      # products
P, D = "CCCCN", "CCCCC"                # extra products / source


class FakeForward(ForwardModel):
    """Returns predefined products for specific reactant sets, [] otherwise."""

    name = "fake"

    def __init__(self, rules: dict[frozenset, list[tuple[str, float]]]):
        # canonicalize both keys and product SMILES up front
        self.rules = {
            frozenset(C(s) for s in key): [(C(p), sc) for p, sc in prods]
            for key, prods in rules.items()
        }

    def predict(self, reactants: str, top_k: int = 10):
        key = frozenset(C(p) for p in reactants.split(".") if p)
        prods = self.rules.get(key, [])
        return [
            ForwardPrediction(product=p, score=sc, template_id=i)
            for i, (p, sc) in enumerate(prods[:top_k])
        ]


def _by_smiles(result):
    return {m.smiles: m for m in result.molecules()}


# --------------------------------------------------------------- basics

def test_basic_invariants():
    model = FakeForward({frozenset([A, B]): [(X, 0.4)]})
    ev = MultiComponentEvolution(model, max_depth=3, score_threshold=0.1)
    res = ev.evolve([A, B, E])
    pool = _by_smiles(res)

    # sources pinned at (1.0, 0), no step / parents
    for s in (A, B, E):
        m = pool[C(s)]
        assert m.total_score == 1.0 and m.depth == 0
        assert m.step_score is None and m.parent_a is None

    # product: total = min(1,1)*0.4, depth = max(0,0)+1
    x = pool[C(X)]
    assert x.total_score == pytest.approx(0.4)
    assert x.depth == 1 and x.step_score == pytest.approx(0.4)
    assert {x.parent_a, x.parent_b} == {C(A), C(B)}
    assert res.stats["num_sources"] == 3
    res.close()


def test_duplicate_keeps_max_score_and_its_depth():
    # P made two ways in the same round; keep the higher-scoring edge.
    model = FakeForward({
        frozenset([A, B]): [(P, 0.2)],
        frozenset([E, D]): [(P, 0.7)],
    })
    ev = MultiComponentEvolution(model, max_depth=3, score_threshold=0.1)
    res = ev.evolve([A, B, E, D])
    p = _by_smiles(res)[C(P)]
    assert p.total_score == pytest.approx(0.7)
    assert {p.parent_a, p.parent_b} == {C(E), C(D)}
    res.close()


# ------------------------------------------------------ score propagation

def _propagation_model():
    # round1: A+B -> X(0.3) and Z(0.9)
    # round2: Z+A -> X(0.9)  [X improves 0.3 -> 0.81]
    #         X+A -> Y(0.5)  [made from stale X, must be revised upward]
    return FakeForward({
        frozenset([A, B]): [(X, 0.3), (Z, 0.9)],
        frozenset([A, Z]): [(X, 0.9)],
        frozenset([A, X]): [(Y, 0.5)],
    })


def test_score_propagates_downstream_after_reactant_improves():
    ev = MultiComponentEvolution(
        _propagation_model(), max_depth=6, score_threshold=0.1
    )
    res = ev.evolve([A, B])
    pool = _by_smiles(res)

    # X improved from 0.3 to min(1,0.9)*0.9 = 0.81, depth = max(depth Z=1,0)+1 = 2
    x = pool[C(X)]
    assert x.total_score == pytest.approx(0.81)
    assert x.depth == 2
    assert {x.parent_a, x.parent_b} == {C(A), C(Z)}

    # Y = min(X=0.81, 1)*0.5 = 0.405  (NOT the stale 0.15), depth = max(2,0)+1 = 3
    y = pool[C(Y)]
    assert y.total_score == pytest.approx(0.405)
    assert y.depth == 3
    res.close()


def test_memory_and_disk_give_identical_results(tmp_path):
    def run(mode):
        ev = MultiComponentEvolution(
            _propagation_model(), max_depth=6, score_threshold=0.1,
            mode=mode, work_dir=(tmp_path / "wd") if mode == "disk" else None,
        )
        res = ev.evolve([A, B])
        snap = sorted(
            (
                m.smiles,
                round(m.total_score, 9),
                m.depth,
                None if m.step_score is None else round(m.step_score, 9),
                tuple(sorted(p for p in (m.parent_a, m.parent_b) if p)),
            )
            for m in res.molecules()
        )
        res.close()
        return snap

    assert run("memory") == run("disk")


# --------------------------------------------------------- pruning knobs

def test_threshold_blocks_selection():
    # X scores 0.3; with threshold 0.5 it cannot react further, so Y never forms.
    model = FakeForward({
        frozenset([A, B]): [(X, 0.3)],
        frozenset([A, X]): [(Y, 0.9)],
    })
    ev = MultiComponentEvolution(model, max_depth=5, score_threshold=0.5)
    res = ev.evolve([A, B])
    pool = _by_smiles(res)
    assert C(X) in pool            # recorded...
    assert C(Y) not in pool        # ...but never used as a reactant
    res.close()


def test_max_depth_limits_tree():
    model = FakeForward({
        frozenset([A, B]): [(X, 0.9)],
        frozenset([A, X]): [(Y, 0.9)],
    })
    ev = MultiComponentEvolution(model, max_depth=1, score_threshold=0.1)
    res = ev.evolve([A, B])
    pool = _by_smiles(res)
    assert C(X) in pool            # depth 1, allowed
    assert C(Y) not in pool        # would be depth 2 > max_depth
    res.close()


def test_self_pair_toggle():
    model = FakeForward({frozenset([A]): [(X, 0.9)]})   # A + A -> X

    on = MultiComponentEvolution(model, max_depth=3, score_threshold=0.1,
                                 allow_self_pair=True).evolve([A])
    assert C(X) in _by_smiles(on)
    on.close()

    off = MultiComponentEvolution(model, max_depth=3, score_threshold=0.1,
                                  allow_self_pair=False).evolve([A])
    assert C(X) not in _by_smiles(off)
    off.close()


def test_skips_unparseable_reactants():
    model = FakeForward({})
    ev = MultiComponentEvolution(model, max_depth=2, score_threshold=0.1)
    with pytest.warns(UserWarning):
        res = ev.evolve([A, "this-is-not-a-smiles!!"])
    assert res.stats["num_sources"] == 1
    res.close()


def test_all_unparseable_raises():
    ev = MultiComponentEvolution(FakeForward({}), max_depth=2, score_threshold=0.1)
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):
            ev.evolve(["nope!!", "also-bad??"])


def test_disk_mode_requires_work_dir():
    ev = MultiComponentEvolution(FakeForward({}), max_depth=2, score_threshold=0.1,
                                 mode="disk")
    with pytest.raises(ValueError):
        ev.evolve([A, B])


def test_frontier_width_limits_pairing():
    # All three sources score 1.0; width=2 keeps the top two by (total, smiles):
    # CCC, CCN — dropping CCO (A), so the A+B reaction never fires.
    model = FakeForward({frozenset([A, B]): [(X, 0.9)]})

    wide = MultiComponentEvolution(model, max_depth=3, score_threshold=0.1)
    assert C(X) in _by_smiles(wide.evolve([A, B, E]))

    narrow = MultiComponentEvolution(model, max_depth=3, score_threshold=0.1,
                                     frontier_width=2)
    assert C(X) not in _by_smiles(narrow.evolve([A, B, E]))


def test_max_reactions_caps_run():
    model = FakeForward({
        frozenset([A, B]): [(X, 0.9)],
        frozenset([A, E]): [(Z, 0.9)],
        frozenset([B, E]): [(P, 0.9)],
    })
    ev = MultiComponentEvolution(model, max_depth=3, score_threshold=0.1,
                                 max_reactions=1)
    res = ev.evolve([A, B, E])
    assert res.stats["num_reactions"] == 1
    assert res.stats["termination"] == "max_reactions"
    res.close()


def test_disk_reuse_same_workdir_is_clean(tmp_path):
    # Regression: a second evolve() on the same work_dir must not read stale
    # molecules/edges/reacted from the first (it would skip all pairs, rounds=0).
    model = FakeForward({frozenset([A, B]): [(X, 0.4)]})
    wd = tmp_path / "wd"
    ev = MultiComponentEvolution(model, max_depth=3, score_threshold=0.1,
                                 mode="disk", work_dir=wd)

    r1 = ev.evolve([A, B])
    s1 = (r1.num_molecules, r1.stats["rounds"], r1.stats["num_reactions"])
    r1.close()

    r2 = ev.evolve([A, B])
    s2 = (r2.num_molecules, r2.stats["rounds"], r2.stats["num_reactions"])
    assert r2.stats["rounds"] > 0            # not skipped due to stale reacted set
    r2.close()

    assert s1 == s2


def test_result_context_manager(tmp_path):
    model = FakeForward({frozenset([A, B]): [(X, 0.4)]})
    ev = MultiComponentEvolution(model, max_depth=3, score_threshold=0.1,
                                 mode="disk", work_dir=tmp_path / "wd")
    with ev.evolve([A, B]) as res:
        assert res.num_molecules >= 3


def test_best_route_backtracks():
    res = MultiComponentEvolution(
        _propagation_model(), max_depth=6, score_threshold=0.1
    ).evolve([A, B])
    route = res.best_route(C(Y))
    # Y <- (X, A) and X <- (Z, A); both edges present, sources not expanded.
    products = {e.product for e in route}
    assert C(Y) in products and C(X) in products
    res.close()
