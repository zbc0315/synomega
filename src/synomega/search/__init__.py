"""Multi-step route planning over an AND-OR graph."""

from .base import Budget, SearchAlgorithm, SearchResult, SearchStats
from .bfs import BestFirstSearch
from .graph import AndOrGraph, MolNode, RxnNode
from .mcts import MCTS
from .retrostar import RetroStar
from .value import (
    ConstantValue,
    MolSizeValue,
    ValueFunction,
    ZeroValue,
    get_value_function,
)

ALGORITHMS = {
    "bfs": BestFirstSearch,
    "retrostar": RetroStar,
    "mcts": MCTS,
}


def get_algorithm(name: str):
    """Look up a search algorithm class by name."""
    if name not in ALGORITHMS:
        raise ValueError(
            f"unknown algorithm {name!r}; choose from {sorted(ALGORITHMS)}"
        )
    return ALGORITHMS[name]


__all__ = [
    "SearchAlgorithm",
    "SearchResult",
    "SearchStats",
    "Budget",
    "AndOrGraph",
    "MolNode",
    "RxnNode",
    "BestFirstSearch",
    "RetroStar",
    "MCTS",
    "ValueFunction",
    "ZeroValue",
    "ConstantValue",
    "MolSizeValue",
    "get_value_function",
    "ALGORITHMS",
    "get_algorithm",
]
