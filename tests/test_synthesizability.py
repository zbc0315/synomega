"""Synthesizability metrics, stock backends, and the expansion cache."""

from __future__ import annotations

from synomega import Planner, SynthesizabilityScorer
from synomega.singlestep.cache import CachedModel
from synomega.stock import EmptyStock, InMemoryStock
from synomega.synthesizability.metrics import BatchReport, MoleculeReport


# ----------------------------------------------------------------- stock


def test_stock_membership_is_representation_independent():
    stock = InMemoryStock.from_smiles(["CCO"])
    assert "OCC" in stock          # different SMILES, same molecule
    assert "C(C)O" in stock
    assert "CC(=O)O" not in stock
    assert "garbage!!" not in stock


def test_stock_roundtrips_through_keys_file(tmp_path):
    stock = InMemoryStock.from_smiles(["CCO", "CC(=O)O"])
    path = tmp_path / "keys.txt"
    stock.save_keys(path)
    reloaded = InMemoryStock.from_keys_file(path)
    assert len(reloaded) == 2
    assert "CCO" in reloaded


def test_stock_from_file(tmp_path):
    path = tmp_path / "cat.smi"
    path.write_text("CCO ethanol\nCC(=O)O acetic-acid\n\n# comment\n")
    stock = InMemoryStock.from_file(path)
    assert len(stock) == 2
    assert "CCO" in stock


# ----------------------------------------------------------------- cache


def test_cache_avoids_repeat_model_calls(fake_model, stock):
    cached = CachedModel(fake_model)
    cached.predict("CC(=O)Nc1ccccc1", top_k=10)
    calls_after_first = fake_model.calls
    cached.predict("CC(=O)Nc1ccccc1", top_k=10)
    assert fake_model.calls == calls_after_first    # served from cache
    assert cached.hits == 1


def test_cache_recomputes_when_wider_beam_requested(fake_model):
    cached = CachedModel(fake_model)
    cached.predict("CC(=O)Nc1ccccc1", top_k=1)
    before = fake_model.calls
    cached.predict("CC(=O)Nc1ccccc1", top_k=50)     # wider than cached
    assert fake_model.calls == before + 1


def test_cache_survives_process_via_sqlite(tmp_path, fake_model):
    db = tmp_path / "cache.sqlite"
    first = CachedModel(fake_model, disk_path=db)
    first.predict("CC(=O)Nc1ccccc1", top_k=10)
    first.close()

    fresh_model_calls = fake_model.calls
    second = CachedModel(fake_model, disk_path=db)
    preds = second.predict("CC(=O)Nc1ccccc1", top_k=10)
    assert fake_model.calls == fresh_model_calls     # no recompute
    assert preds and preds[0].score > 0
    second.close()


# ----------------------------------------------------------- scoring


def test_scores_a_solvable_target(fake_model, stock):
    planner = Planner(fake_model, stock, algorithm="bfs", cache=False)
    scorer = SynthesizabilityScorer(planner)
    report = scorer.score("CC(=O)Nc1ccccc1", max_steps=4)

    assert report.solved
    assert report.bb_coverage == 1.0
    assert report.min_steps is not None and report.min_steps >= 1
    assert report.min_route_depth is not None
    assert report.num_leaves == report.num_purchasable_leaves
    assert report.error is None


def test_unsolvable_target_reports_partial_coverage(fake_model):
    """Acetic acid buyable, aniline route unavailable -> partial credit."""
    stock = InMemoryStock.from_smiles(["CC(=O)O"])
    planner = Planner(fake_model, stock, algorithm="bfs", cache=False)
    scorer = SynthesizabilityScorer(planner)
    report = scorer.score("CC(=O)Nc1ccccc1", max_steps=1)

    assert not report.solved
    assert report.min_steps is None
    # Partial credit is the point: this is a near-miss, not a total failure.
    assert 0.0 < report.bb_coverage < 1.0


def test_completely_unsolvable_scores_zero(fake_model):
    planner = Planner(fake_model, EmptyStock(), algorithm="bfs", cache=False)
    scorer = SynthesizabilityScorer(planner)
    report = scorer.score("CC(=O)Nc1ccccc1", max_steps=2)
    assert not report.solved
    assert report.bb_coverage == 0.0


def test_unparseable_target_is_reported_not_raised(fake_model, stock):
    planner = Planner(fake_model, stock, algorithm="bfs", cache=False)
    scorer = SynthesizabilityScorer(planner)
    report = scorer.score("this is not a molecule", max_steps=3)
    assert report.error is not None
    assert not report.solved


# ------------------------------------------------------------- batch


def test_batch_report_aggregates(fake_model, stock):
    planner = Planner(fake_model, stock, algorithm="bfs", cache=False)
    scorer = SynthesizabilityScorer(planner)
    targets = ["CC(=O)Nc1ccccc1", "CCO", "CC(=O)O"]
    report = scorer.score_batch(targets, max_steps=4, progress=False)

    assert report.n == 3
    assert report.solve_rate == 1.0           # all three reachable
    assert report.mean_bb_coverage == 1.0
    assert "solve_rate" in report.summary()


def test_solve_rate_ignores_errored_targets():
    reports = [
        MoleculeReport("A", True, 1.0, 2, 2, 1),
        MoleculeReport("B", False, 0.5, None, None, 0),
        MoleculeReport("C", False, 0.0, None, None, 0, error="boom"),
    ]
    batch = BatchReport(reports=reports, max_steps=5)
    assert batch.n == 3
    assert batch.n_evaluated == 2
    assert batch.solve_rate == 0.5            # 1 of 2 valid, not 1 of 3


def test_solve_rate_at_depth():
    reports = [
        MoleculeReport("A", True, 1.0, 1, 1, 1),
        MoleculeReport("B", True, 1.0, 4, 4, 1),
    ]
    batch = BatchReport(reports=reports, max_steps=5)
    assert batch.solve_rate_at(1) == 0.5      # only A is a 1-step route
    assert batch.solve_rate_at(4) == 1.0
    assert batch.depth_histogram == {1: 1, 4: 1}


def test_score_is_synscore_of_unpurchasable_count():
    # SynScore = 1/(U+1)**U, U = non-purchasable leaves of the best route.
    solved = MoleculeReport("A", True, 1.0, 1, 1, 1,
                            leaves=[("x", True), ("y", True)])
    one = MoleculeReport("B", False, 0.5, None, None, 0,
                         leaves=[("x", True), ("y", False)])
    two = MoleculeReport("C", False, 0.33, None, None, 0,
                         leaves=[("x", False), ("y", False), ("z", True)])
    no_route = MoleculeReport("D", False, 0.0, None, None, 0, leaves=[])
    assert solved.num_unpurchasable_leaves == 0 and solved.score == 1.0
    assert one.score == 0.5
    assert abs(two.score - 1.0 / 9.0) < 1e-9
    assert no_route.score == 0.0
