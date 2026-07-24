"""Search algorithm interface and shared result types."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..chem.mol import Molecule
from ..singlestep.base import SingleStepModel
from ..stock.base import BuildingBlockSet
from .graph import AndOrGraph, MolNode


@dataclass
class SearchStats:
    """What the search actually did — always report this, never guess at it."""

    expansions: int = 0            # molecules expanded (model calls)
    model_calls: int = 0           # batched calls to the single-step model
    nodes_created: int = 0
    reactions_created: int = 0
    elapsed_s: float = 0.0
    terminated_by: str = ""        # solved | depth | time | expansions | exhausted

    def as_dict(self) -> dict:
        return {
            "expansions": self.expansions,
            "model_calls": self.model_calls,
            "nodes_created": self.nodes_created,
            "reactions_created": self.reactions_created,
            "elapsed_s": round(self.elapsed_s, 3),
            "terminated_by": self.terminated_by,
        }


@dataclass
class SearchResult:
    """Outcome of one planning run."""

    target: str
    solved: bool
    graph: AndOrGraph
    stats: SearchStats
    algorithm: str = ""
    _routes: list = field(default_factory=list, repr=False)

    @property
    def routes(self) -> list:
        """Routes, best first, extracted lazily.

        When the target was solved these are complete routes. When it was not,
        this is the single best *partial* route — which is what carries the
        `bb_coverage` near-miss signal, so it must still be extracted.
        """
        if not self._routes:
            from ..route.route import extract_routes

            self._routes = extract_routes(self.graph)
        return self._routes

    @property
    def best_route(self):
        routes = self.routes
        return routes[0] if routes else None

    def __repr__(self) -> str:
        return (
            f"SearchResult({self.target!r}, solved={self.solved}, "
            f"routes={len(self.routes)}, {self.stats.expansions} expansions)"
        )


class Budget:
    """Shared stopping conditions, checked by every algorithm."""

    def __init__(
        self,
        *,
        max_depth: int = 6,
        time_limit: float | None = 60.0,
        max_expansions: int | None = 500,
    ):
        """`None` means unlimited. An explicit 0 means "already exhausted" —
        it is not treated as "no limit", which would be a surprising reading."""
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.max_expansions = max_expansions
        self._start = 0.0

    def start(self) -> None:
        self._start = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self._start

    def exhausted(self, expansions: int) -> str | None:
        """Return the reason to stop, or None to continue."""
        if self.time_limit is not None and self.elapsed >= self.time_limit:
            return "time"
        if self.max_expansions is not None and expansions >= self.max_expansions:
            return "expansions"
        return None


class SearchAlgorithm(ABC):
    """Multi-step planner over an AND-OR graph."""

    name: str = "search"

    def __init__(
        self,
        model: SingleStepModel,
        stock: BuildingBlockSet,
        *,
        expansion_width: int = 50,
    ):
        self.model = model
        self.stock = stock
        self.expansion_width = expansion_width

    @abstractmethod
    def run(self, target: str, budget: Budget) -> SearchResult:
        ...

    # ------------------------------------------------------------- helpers

    def _make_graph(self, target: str) -> AndOrGraph:
        mol = Molecule.of(target)
        return AndOrGraph.create(mol, in_stock=mol in self.stock)

    def _child_nodes(
        self, graph: AndOrGraph, reactant_smiles: tuple[str, ...], depth: int
    ) -> tuple[list[MolNode], int] | None:
        """Materialize reactant nodes. None if any component is unparseable."""
        nodes: list[MolNode] = []
        created = 0
        for smi in reactant_smiles:
            mol = Molecule.try_of(smi)
            if mol is None:
                return None
            node, is_new = graph.get_or_create(mol, depth, in_stock=mol in self.stock)
            if is_new:
                created += 1
            nodes.append(node)
        return nodes, created


__all__ = ["SearchAlgorithm", "SearchResult", "SearchStats", "Budget"]
