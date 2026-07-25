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
        plausibility=None,
        plausibility_threshold: float = 0.4,
        plausibility_kwargs: dict | None = None,
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
            plausibility: screen every single-step prediction with the dual-tower
                reaction-plausibility model. Accepts a ``PlausibilityScorer``
                instance, ``True`` (download+load the default model), or ``None``/
                ``False`` (off). Candidates whose ``reactants -> target`` scores
                below ``plausibility_threshold`` are dropped.
            plausibility_threshold: minimum plausibility to keep a candidate.
            plausibility_kwargs: extra kwargs for ``PlausibilityFilteredModel``
                (e.g. ``min_keep``, ``overfetch``, ``rerank``).
        """
        from .search import get_algorithm

        base = model
        if plausibility is not None and plausibility is not False:
            from .plausibility import PlausibilityFilteredModel, PlausibilityScorer

            scorer = plausibility
            if scorer is True:
                scorer = PlausibilityScorer.default(
                    device=getattr(model, "device", "cpu"))
            base = PlausibilityFilteredModel(
                base, scorer, threshold=plausibility_threshold,
                **(plausibility_kwargs or {}),
            )
        # Filter sits *inside* the cache, so cached expansions are already screened.
        self.model = (
            CachedModel(base, disk_path=cache_path) if cache else base
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
        exclude_target: bool = False,
    ) -> SearchResult:
        """Search for routes to `target`. Budget args override the defaults.

        `exclude_target=True` treats the target as *not* purchasable even if it
        is in the stock, so a molecule that is itself a catalogue item is not
        reported as trivially solved in zero steps.
        """
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
        return self.searcher.run(target, budget, exclude_target=exclude_target)

    @property
    def cache_hit_rate(self) -> float:
        return getattr(self.model, "hit_rate", 0.0)

    def __repr__(self) -> str:
        return (
            f"Planner(model={self.model.name}, stock={self.stock.name}, "
            f"algorithm={self.algorithm_name})"
        )


__all__ = ["Planner"]
