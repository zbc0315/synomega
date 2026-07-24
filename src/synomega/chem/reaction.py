"""Reaction objects and reaction-SMILES helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from .mol import Molecule, canonicalize


@dataclass(frozen=True)
class Conditions:
    """Reaction conditions.

    Reserved for a future condition-prediction model (solvent / temperature /
    reagent). Nothing in synomega populates this yet; it exists so `Route`
    serialization is stable once conditions land.
    """

    solvent: str | None = None
    temperature_c: float | None = None
    reagent: str | None = None
    extra: dict = field(default_factory=dict)

    def is_empty(self) -> bool:
        return (
            self.solvent is None
            and self.temperature_c is None
            and self.reagent is None
            and not self.extra
        )


@dataclass(frozen=True)
class Reaction:
    """A single retrosynthetic step: product <- reactants."""

    product: Molecule
    reactants: tuple[Molecule, ...]
    score: float = 0.0
    template_id: int | None = None
    conditions: Conditions | None = None
    meta: dict = field(default_factory=dict, compare=False)

    @property
    def reaction_smiles(self) -> str:
        """Forward-direction reaction SMILES (reactants >> product)."""
        return f"{'.'.join(m.smiles for m in self.reactants)}>>{self.product.smiles}"

    @property
    def retro_smiles(self) -> str:
        """Retro-direction string (product >> reactants)."""
        return f"{self.product.smiles}>>{'.'.join(m.smiles for m in self.reactants)}"

    def __repr__(self) -> str:
        return f"Reaction({self.retro_smiles}, score={self.score:.4f})"


def parse_reaction_smiles(rxn: str) -> tuple[list[str], list[str], list[str]] | None:
    """Split `reactants>agents>products` into canonical component lists."""
    if ">" not in rxn:
        return None
    parts = rxn.split(">")
    if len(parts) == 2:
        left, agents, right = parts[0], "", parts[1]
    elif len(parts) == 3:
        left, agents, right = parts
    else:
        return None

    def _split(block: str) -> list[str]:
        out = []
        for piece in block.split("."):
            if not piece:
                continue
            canon = canonicalize(piece)
            if canon is not None:
                out.append(canon)
        return out

    return _split(left), _split(agents), _split(right)


def extract_product(rxn: str) -> str | None:
    """Largest product component of a reaction SMILES, canonicalized."""
    parsed = parse_reaction_smiles(rxn)
    if parsed is None:
        return None
    products = parsed[2]
    if not products:
        return None
    return max(products, key=len)


__all__ = [
    "Reaction",
    "Conditions",
    "parse_reaction_smiles",
    "extract_product",
]
