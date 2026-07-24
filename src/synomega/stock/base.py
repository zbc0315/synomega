"""Building-block stock interface.

A stock answers one question: *can I buy this molecule?* Membership is tested by
InChIKey so keys computed here match keys computed from a vendor catalogue
written by a different toolkit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..chem.mol import Molecule


class BuildingBlockSet(ABC):
    """Set of purchasable molecules."""

    name: str = "stock"

    @abstractmethod
    def __contains__(self, mol: str | Molecule) -> bool:
        """True when `mol` is purchasable."""

    @abstractmethod
    def __len__(self) -> int:
        ...

    def contains_batch(self, mols: list[str | Molecule]) -> list[bool]:
        return [m in self for m in mols]

    @staticmethod
    def _key(mol: str | Molecule) -> str | None:
        """Normalize an input to an InChIKey, or None if unparseable."""
        if isinstance(mol, Molecule):
            return mol.key
        parsed = Molecule.try_of(mol)
        return parsed.key if parsed is not None else None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} n={len(self)}>"


class EmptyStock(BuildingBlockSet):
    """Nothing is purchasable. Useful for tests and for pure-expansion runs."""

    name = "empty"

    def __contains__(self, mol) -> bool:
        return False

    def __len__(self) -> int:
        return 0


__all__ = ["BuildingBlockSet", "EmptyStock"]
