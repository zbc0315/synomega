"""Search: graph semantics, all three algorithms, and budget enforcement."""

from __future__ import annotations

import pytest

from synomega.chem import Molecule
from synomega.planner import Planner
from synomega.search import ALGORITHMS, AndOrGraph, Budget
from synomega.stock import EmptyStock, InMemoryStock

ALGOS = sorted(ALGORITHMS)


# ---------------------------------------------------------------- graph


def test_and_or_solved_semantics():
    """A reaction is solved only when EVERY reactant is."""
    target = Molecule.of("CCOC(C)=O")
    graph = AndOrGraph.create(target, in_stock=False)

    acid, _ = graph.get_or_create(Molecule.of("CC(=O)O"), 1, in_stock=True)
    alcohol, _ = graph.get_or_create(Molecule.of("CCO"), 1, in_stock=False)
    rxn = graph.add_reaction(graph.root, [acid, alcohol], score=0.9)

    assert rxn is not None
    assert not rxn.solved          # alcohol not yet solved
    assert not graph.root.solved

    graph.mark_solved(alcohol)     # now both children are solved
    assert rxn.solved
    assert graph.root.solved       # propagated to the root


def test_molecules_are_deduplicated_across_branches():
    target = Molecule.of("CCOC(C)=O")
    graph = AndOrGraph.create(target, in_stock=False)
    a, created_first = graph.get_or_create(Molecule.of("CCO"), 1, in_stock=False)
    b, created_second = graph.get_or_create(Molecule.of("OCC"), 2, in_stock=False)
    assert created_first and not created_second
    assert a is b
    assert graph.num_molecules == 2  # root + ethanol, not 3


def test_cycles_are_rejected():
    """A route that makes X from X is not a route."""
    target = Molecule.of("CCOC(C)=O")
    graph = AndOrGraph.create(target, in_stock=False)
    inter, _ = graph.get_or_create(Molecule.of("CCO"), 1, in_stock=False)
    graph.add_reaction(graph.root, [inter], score=0.9)
    # Now try to make the intermediate from the target itself.
    assert graph.add_reaction(inter, [graph.root], score=0.9) is None


# ------------------------------------------------------------ algorithms


@pytest.mark.parametrize("algo", ALGOS)
def test_one_step_target_is_solved(algo, fake_model, stock):
    """Acetanilide <- acetic acid + aniline; aniline <- nitrobenzene (in stock)."""
    planner = Planner(fake_model, stock, algorithm=algo, cache=False)
    result = planner.plan("CC(=O)Nc1ccccc1", max_depth=4, time_limit=10)
    assert result.solved
    route = result.best_route
    assert route is not None
    assert route.bb_coverage == 1.0
    assert all(leaf.in_stock for leaf in route.leaves)


@pytest.mark.parametrize("algo", ALGOS)
def test_target_already_in_stock_needs_no_search(algo, fake_model, stock):
    planner = Planner(fake_model, stock, algorithm=algo, cache=False)
    result = planner.plan("CCO", max_depth=3, time_limit=5)
    assert result.solved
    assert result.stats.expansions == 0


@pytest.mark.parametrize("algo", ALGOS)
def test_unsolvable_with_empty_stock(algo, fake_model):
    planner = Planner(fake_model, EmptyStock(), algorithm=algo, cache=False)
    result = planner.plan("CC(=O)Nc1ccccc1", max_depth=2,
                          time_limit=5, max_expansions=20)
    assert not result.solved


@pytest.mark.parametrize("algo", ALGOS)
def test_depth_limit_blocks_a_too_deep_route(algo, fake_model):
    """Aniline is reachable only via nitrobenzene, so depth 1 must fail."""
    stock = InMemoryStock.from_smiles(["CC(=O)O", "O=[N+]([O-])c1ccccc1"])
    planner = Planner(fake_model, stock, algorithm=algo, cache=False)

    shallow = planner.plan("CC(=O)Nc1ccccc1", max_depth=1,
                           time_limit=5, max_expansions=50)
    assert not shallow.solved

    deep = planner.plan("CC(=O)Nc1ccccc1", max_depth=3,
                        time_limit=10, max_expansions=200)
    assert deep.solved


@pytest.mark.parametrize("algo", ALGOS)
def test_expansion_budget_is_respected(algo, fake_model):
    planner = Planner(fake_model, EmptyStock(), algorithm=algo, cache=False)
    result = planner.plan("CC(=O)Nc1ccccc1", max_depth=10,
                          time_limit=30, max_expansions=3)
    assert result.stats.expansions <= 3
    assert result.stats.terminated_by in {"expansions", "exhausted"}


def test_unknown_algorithm_raises(fake_model, stock):
    with pytest.raises(ValueError, match="unknown algorithm"):
        Planner(fake_model, stock, algorithm="does-not-exist")


# ---------------------------------------------------------------- budget


def test_budget_reports_time_exhaustion():
    budget = Budget(time_limit=0.0, max_expansions=1000)
    budget.start()
    assert budget.exhausted(0) == "time"
