"""Synthesizability metrics.

Two questions get conflated in the literature and this module keeps them apart:

  * **solved@N** — binary. Is there a route of depth <= N whose leaves are *all*
    purchasable? This is the standard benchmark quantity (`solve_rate` when
    averaged over a set).

  * **bb_coverage@N** — continuous. Of the leaves in the best route found, what
    fraction is purchasable? A 5-step route with 4 of 5 leaves buyable scores
    0.8 rather than 0, which orders molecules far better than a 0/1 flag when
    most targets are unsolved.

Report both. `solved` is what compares to published numbers; `bb_coverage` is
what tells you whether a near-miss was close.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MoleculeReport:
    """Synthesizability of one target."""

    smiles: str
    solved: bool
    bb_coverage: float
    #: Reactions in the shortest solved route (a convergent route has more
    #: reactions than levels, so this can exceed the depth budget).
    min_steps: int | None
    #: Longest linear sequence of the shortest solved route. THIS is what the
    #: step limit constrains -- `max_steps` is enforced on route depth.
    min_route_depth: int | None
    num_routes: int
    leaves: list[tuple[str, bool]] = field(default_factory=list)
    max_steps: int = 0
    elapsed_s: float = 0.0
    expansions: int = 0
    terminated_by: str = ""
    error: str | None = None

    @property
    def num_purchasable_leaves(self) -> int:
        return sum(1 for _, in_stock in self.leaves if in_stock)

    @property
    def num_leaves(self) -> int:
        return len(self.leaves)

    def as_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "solved": self.solved,
            "bb_coverage": round(self.bb_coverage, 4),
            "min_steps": self.min_steps,
            "min_route_depth": self.min_route_depth,
            "num_routes": self.num_routes,
            "num_leaves": self.num_leaves,
            "num_purchasable_leaves": self.num_purchasable_leaves,
            "max_steps": self.max_steps,
            "elapsed_s": round(self.elapsed_s, 3),
            "expansions": self.expansions,
            "terminated_by": self.terminated_by,
            "error": self.error,
        }

    def __repr__(self) -> str:
        if self.error:
            return f"MoleculeReport({self.smiles!r}, ERROR={self.error})"
        return (
            f"MoleculeReport({self.smiles!r}, solved={self.solved}, "
            f"bb_coverage={self.bb_coverage:.2f}, steps={self.min_steps})"
        )


@dataclass
class BatchReport:
    """Synthesizability across a set of targets."""

    reports: list[MoleculeReport]
    max_steps: int = 0

    # ------------------------------------------------------------- headline

    @property
    def n(self) -> int:
        return len(self.reports)

    @property
    def n_evaluated(self) -> int:
        """Targets that ran without error."""
        return sum(1 for r in self.reports if r.error is None)

    @property
    def solve_rate(self) -> float:
        """Fraction solved within the step limit — the headline benchmark."""
        ok = [r for r in self.reports if r.error is None]
        if not ok:
            return 0.0
        return sum(1 for r in ok if r.solved) / len(ok)

    @property
    def mean_bb_coverage(self) -> float:
        ok = [r for r in self.reports if r.error is None]
        if not ok:
            return 0.0
        return sum(r.bb_coverage for r in ok) / len(ok)

    @property
    def depth_histogram(self) -> dict[int, int]:
        """How many targets were solved at each route depth (LLS)."""
        hist: dict[int, int] = {}
        for r in self.reports:
            if r.solved and r.min_route_depth is not None:
                hist[r.min_route_depth] = hist.get(r.min_route_depth, 0) + 1
        return dict(sorted(hist.items()))

    def solve_rate_at(self, steps: int) -> float:
        """solve_rate restricted to routes of depth at most `steps`.

        Keyed on route depth (longest linear sequence), which is the quantity
        the step limit actually constrains. Only meaningful for
        `steps <= max_steps`, since deeper routes were never searched for.
        """
        ok = [r for r in self.reports if r.error is None]
        if not ok:
            return 0.0
        hits = sum(
            1 for r in ok
            if r.solved and r.min_route_depth is not None
            and r.min_route_depth <= steps
        )
        return hits / len(ok)

    @property
    def mean_elapsed_s(self) -> float:
        ok = [r for r in self.reports if r.error is None]
        if not ok:
            return 0.0
        return sum(r.elapsed_s for r in ok) / len(ok)

    # --------------------------------------------------------------- output

    def summary(self) -> dict:
        return {
            "n": self.n,
            "n_evaluated": self.n_evaluated,
            "n_errors": self.n - self.n_evaluated,
            "max_steps": self.max_steps,
            "solve_rate": round(self.solve_rate, 4),
            "mean_bb_coverage": round(self.mean_bb_coverage, 4),
            "depth_histogram": self.depth_histogram,
            "solve_rate_by_depth": {
                k: round(self.solve_rate_at(k), 4)
                for k in range(1, self.max_steps + 1)
            },
            "mean_elapsed_s": round(self.mean_elapsed_s, 3),
        }

    def to_dataframe(self):
        """pandas DataFrame of per-molecule rows (requires pandas)."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("to_dataframe() needs pandas") from exc
        return pd.DataFrame([r.as_dict() for r in self.reports])

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(
            {
                "summary": self.summary(),
                "molecules": [r.as_dict() for r in self.reports],
            },
            indent=indent,
        )

    def describe(self) -> str:
        s = self.summary()
        lines = [
            f"targets: {s['n']}  evaluated: {s['n_evaluated']}  errors: {s['n_errors']}",
            f"max_steps: {s['max_steps']}",
            f"solve_rate:       {s['solve_rate']:.4f}",
            f"mean_bb_coverage: {s['mean_bb_coverage']:.4f}",
            f"mean_elapsed_s:   {s['mean_elapsed_s']:.3f}",
            "solve_rate by max depth:",
        ]
        for k, v in s["solve_rate_by_depth"].items():
            lines.append(f"  <= {k} steps: {v:.4f}")
        if s["depth_histogram"]:
            lines.append(f"solved-route length histogram: {s['depth_histogram']}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"BatchReport(n={self.n}, solve_rate={self.solve_rate:.3f}, "
            f"mean_bb_coverage={self.mean_bb_coverage:.3f})"
        )


__all__ = ["MoleculeReport", "BatchReport"]
