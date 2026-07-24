"""Cost-to-go value functions.

Retro*-style search ranks the frontier by `g + h`, where `g` is the accumulated
reaction cost and `h` estimates the remaining cost to reach purchasable
material. A value function supplies `h`.

`ZeroValue` is admissible (never overestimates), which makes the search
optimal but slow. The size-based heuristics are inadmissible but usually find
routes far sooner — the standard trade in practice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .graph import MolNode


class ValueFunction(ABC):
    """Estimate the remaining cost to make a molecule from stock."""

    name: str = "value"

    @abstractmethod
    def __call__(self, node: MolNode) -> float:
        ...

    def batch(self, nodes: list[MolNode]) -> list[float]:
        return [self(n) for n in nodes]


class ZeroValue(ValueFunction):
    """h = 0. Admissible; turns Retro* into uniform-cost search."""

    name = "zero"

    def __call__(self, node: MolNode) -> float:
        return 0.0


class ConstantValue(ValueFunction):
    """Flat penalty per unsolved molecule — a crude but effective depth bias."""

    name = "constant"

    def __init__(self, cost: float = 1.0):
        self.cost = cost

    def __call__(self, node: MolNode) -> float:
        return 0.0 if node.in_stock else self.cost


class MolSizeValue(ValueFunction):
    """Bigger molecules are assumed further from purchasable material.

    h = scale * max(0, heavy_atoms - free_atoms). The intuition: fragments at or
    below `free_atoms` heavy atoms are plausibly buyable, and cost grows with
    how much larger the molecule is.
    """

    name = "molsize"

    def __init__(self, scale: float = 0.1, free_atoms: int = 12):
        self.scale = scale
        self.free_atoms = free_atoms

    def __call__(self, node: MolNode) -> float:
        if node.in_stock:
            return 0.0
        try:
            n = node.mol.num_heavy_atoms
        except Exception:
            return 0.0
        return self.scale * max(0, n - self.free_atoms)


def get_value_function(spec: "str | ValueFunction | None") -> ValueFunction:
    if spec is None:
        return MolSizeValue()
    if isinstance(spec, ValueFunction):
        return spec
    table = {
        "zero": ZeroValue,
        "constant": ConstantValue,
        "molsize": MolSizeValue,
    }
    if spec not in table:
        raise ValueError(
            f"unknown value function {spec!r}; choose from {sorted(table)}"
        )
    return table[spec]()


__all__ = [
    "ValueFunction",
    "ZeroValue",
    "ConstantValue",
    "MolSizeValue",
    "get_value_function",
]
