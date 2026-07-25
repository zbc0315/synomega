"""Dual-tower reaction-plausibility model (mapping-free) and its single-step filter.

Screens single-step retrosynthesis predictions: for each proposed
``reactants -> target`` disconnection, a shared-encoder D-MPNN scores how likely
the reactants actually give the target, and implausible candidates are dropped.
Needs the ``gnn`` extra (torch + torch_geometric).
"""

from __future__ import annotations

from .scorer import PlausibilityScorer
from .filter import PlausibilityFilteredModel

__all__ = ["PlausibilityScorer", "PlausibilityFilteredModel"]
