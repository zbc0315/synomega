"""Synthesizability scoring: can this molecule be made, in N steps, from stock?"""

from __future__ import annotations

import sys
import time
from typing import Iterable, Sequence

from ..route.route import Route
from .metrics import BatchReport, MoleculeReport


class SynthesizabilityScorer:
    """Runs a planner and turns the outcome into synthesizability metrics."""

    def __init__(self, planner):
        """
        Args:
            planner: a `synomega.Planner` (or anything with a compatible
                `.plan(smiles, max_depth=..., time_limit=..., max_expansions=...)`).
        """
        self.planner = planner

    # ------------------------------------------------------------- single

    def score(
        self,
        smiles: str,
        *,
        max_steps: int = 5,
        time_limit: float | None = None,
        max_expansions: int | None = None,
    ) -> MoleculeReport:
        """Score one target.

        `max_steps` is the user-facing step limit: the longest route allowed.
        """
        return self.score_detailed(
            smiles,
            max_steps=max_steps,
            time_limit=time_limit,
            max_expansions=max_expansions,
        )[0]

    def score_detailed(
        self,
        smiles: str,
        *,
        max_steps: int = 5,
        time_limit: float | None = None,
        max_expansions: int | None = None,
    ) -> tuple[MoleculeReport, object | None]:
        """Score one target, also returning the underlying `SearchResult`.

        Use this when you need the route tree or search stats as well — it runs
        the search once, where calling `score()` and `plan()` separately would
        run it twice.
        """
        t0 = time.time()
        try:
            result = self.planner.plan(
                smiles,
                max_depth=max_steps,
                time_limit=time_limit,
                max_expansions=max_expansions,
            )
        except Exception as exc:
            return (
                MoleculeReport(
                    smiles=smiles,
                    solved=False,
                    bb_coverage=0.0,
                    min_steps=None,
                    min_route_depth=None,
                    num_routes=0,
                    max_steps=max_steps,
                    elapsed_s=time.time() - t0,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                None,
            )

        routes: Sequence[Route] = result.routes
        solved_routes = [r for r in routes if r.solved]

        if solved_routes:
            # "Shortest" = fewest reactions; report its depth alongside.
            best = min(solved_routes, key=lambda r: r.num_steps)
            coverage = 1.0
            min_steps = best.num_steps
            min_route_depth = best.depth
        elif routes:
            # Unsolved: report the best partial route's coverage so a near-miss
            # is distinguishable from a total failure.
            best = max(routes, key=lambda r: r.bb_coverage)
            coverage = best.bb_coverage
            min_steps = min_route_depth = None
        else:
            best = None
            coverage = 0.0
            min_steps = min_route_depth = None

        leaves = (
            [(leaf.smiles, leaf.in_stock) for leaf in best.leaves] if best else []
        )

        return (
            MoleculeReport(
                smiles=smiles,
                solved=bool(solved_routes),
                bb_coverage=coverage,
                min_steps=min_steps,
                min_route_depth=min_route_depth,
                num_routes=len(solved_routes),
                leaves=leaves,
                max_steps=max_steps,
                elapsed_s=time.time() - t0,
                expansions=result.stats.expansions,
                terminated_by=result.stats.terminated_by,
            ),
            result,
        )

    # -------------------------------------------------------------- batch

    def score_batch(
        self,
        smiles: Iterable[str],
        *,
        max_steps: int = 5,
        time_limit: float | None = None,
        max_expansions: int | None = None,
        progress: bool = True,
        on_result=None,
    ) -> BatchReport:
        """Score many targets sequentially.

        Sequential on purpose: the planner holds a GPU model and a shared
        expansion cache, both of which behave badly under naive multiprocessing.
        For large campaigns, shard the input across processes at the shell level
        (each with its own disk cache) rather than threading here.

        Args:
            on_result: optional callback invoked with each `MoleculeReport` as
                it completes — use it to stream results to disk so a long run is
                resumable.
        """
        targets = list(smiles)
        reports: list[MoleculeReport] = []

        for i, smi in enumerate(targets, 1):
            report = self.score(
                smi,
                max_steps=max_steps,
                time_limit=time_limit,
                max_expansions=max_expansions,
            )
            reports.append(report)
            if on_result is not None:
                on_result(report)
            if progress:
                solved = sum(1 for r in reports if r.solved)
                print(
                    f"\r[{i}/{len(targets)}] solve_rate={solved / i:.3f} "
                    f"last={'OK ' if report.solved else 'miss'} {smi[:40]:<40}",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
        if progress and targets:
            print(file=sys.stderr)

        return BatchReport(reports=reports, max_steps=max_steps)


__all__ = ["SynthesizabilityScorer"]
