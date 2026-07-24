"""The `Planner` facade — the one object most users need."""

from __future__ import annotations

from pathlib import Path

from .search.base import Budget, SearchResult
from .singlestep.base import SingleStepModel
from .singlestep.cache import CachedModel
from .stock.base import BuildingBlockSet


class Planner:
    """Ties a single-step model, a stock, and a search algorithm together."""

    def __init__(
        self,
        model: SingleStepModel,
        stock: BuildingBlockSet,
        *,
        algorithm: str = "retrostar",
        expansion_width: int = 50,
        max_depth: int = 6,
        time_limit: float = 60.0,
        max_expansions: int = 500,
        cache: bool = True,
        cache_path: str | Path | None = None,
        **algorithm_kwargs,
    ):
        """
        Args:
            model: single-step retrosynthesis backend.
            stock: purchasable building blocks.
            algorithm: "retrostar" | "bfs" | "mcts".
            expansion_width: candidate reactants requested per molecule.
            max_depth / time_limit / max_expansions: default search budget;
                overridable per `plan()` call.
            cache: wrap the model in an expansion cache (strongly recommended —
                search revisits the same molecules constantly).
            cache_path: persist the cache to SQLite so it survives the process.
        """
        from .search import get_algorithm

        self.model = (
            CachedModel(model, disk_path=cache_path) if cache else model
        )
        self.stock = stock
        self.algorithm_name = algorithm
        self.default_budget = dict(
            max_depth=max_depth,
            time_limit=time_limit,
            max_expansions=max_expansions,
        )
        self.searcher = get_algorithm(algorithm)(
            self.model,
            stock,
            expansion_width=expansion_width,
            **algorithm_kwargs,
        )

    def plan(
        self,
        target: str,
        *,
        max_depth: int | None = None,
        time_limit: float | None = None,
        max_expansions: int | None = None,
    ) -> SearchResult:
        """Search for routes to `target`. Budget args override the defaults."""
        budget = Budget(
            max_depth=(
                max_depth if max_depth is not None
                else self.default_budget["max_depth"]
            ),
            time_limit=(
                time_limit if time_limit is not None
                else self.default_budget["time_limit"]
            ),
            max_expansions=(
                max_expansions if max_expansions is not None
                else self.default_budget["max_expansions"]
            ),
        )
        return self.searcher.run(target, budget)

    @property
    def cache_hit_rate(self) -> float:
        return getattr(self.model, "hit_rate", 0.0)

    def __repr__(self) -> str:
        return (
            f"Planner(model={self.model.name}, stock={self.stock.name}, "
            f"algorithm={self.algorithm_name})"
        )


__all__ = ["Planner"]
