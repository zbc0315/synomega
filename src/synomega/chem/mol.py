"""Molecule handling: canonicalization, identity keys, caching.

Identity model
--------------
A molecule's *identity* is its InChIKey. We use InChIKey rather than canonical
SMILES because the building-block stock comes from an external source
(eMolecules) written by a different toolkit — InChIKey survives that round trip,
canonical SMILES does not reliably.

Both the canonical SMILES and the InChIKey are cached, so repeated lookups
during search are cheap.
"""

from __future__ import annotations

from functools import lru_cache

from rdkit import Chem, RDLogger

# RDKit is noisy about every unparseable SMILES; we handle failures explicitly.
RDLogger.DisableLog("rdApp.*")


class MoleculeError(ValueError):
    """Raised when a SMILES string cannot be parsed."""


@lru_cache(maxsize=500_000)
def canonicalize(smiles: str) -> str | None:
    """Canonical SMILES, or None if unparseable. Atom maps are stripped."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


@lru_cache(maxsize=500_000)
def inchi_key(canonical_smiles: str) -> str | None:
    """InChIKey of an already-canonical SMILES, or None on failure."""
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return None
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    return key or None


def split_components(smiles: str) -> list[str]:
    """Split a dot-separated SMILES into canonical component SMILES.

    Multi-component reactant strings ("A.B") are split so each component can be
    checked against stock and expanded independently.
    """
    parts = [p for p in smiles.split(".") if p]
    out: list[str] = []
    for part in parts:
        canon = canonicalize(part)
        if canon is not None:
            out.append(canon)
    return out


class Molecule:
    """A canonicalized molecule with a stable identity key.

    Instances are interned per canonical SMILES, so `Molecule.of(s) is
    Molecule.of(s)` holds and equality/hashing are pointer-cheap.
    """

    __slots__ = ("smiles", "_key", "_rdmol")

    _interned: dict[str, "Molecule"] = {}

    def __init__(self, canonical_smiles: str):
        self.smiles = canonical_smiles
        self._key: str | None = None
        self._rdmol = None

    @classmethod
    def of(cls, smiles: str) -> "Molecule":
        """Build (or reuse) a Molecule from an arbitrary SMILES string."""
        canon = canonicalize(smiles)
        if canon is None:
            raise MoleculeError(f"unparseable SMILES: {smiles!r}")
        hit = cls._interned.get(canon)
        if hit is None:
            hit = cls(canon)
            cls._interned[canon] = hit
        return hit

    @classmethod
    def try_of(cls, smiles: str) -> "Molecule | None":
        """Like `of`, but returns None instead of raising."""
        try:
            return cls.of(smiles)
        except MoleculeError:
            return None

    @property
    def key(self) -> str:
        """InChIKey — the identity used by the search graph and the stock."""
        if self._key is None:
            k = inchi_key(self.smiles)
            # Fall back to canonical SMILES when InChI generation fails (rare:
            # exotic valences). Still a stable key, just not cross-toolkit.
            self._key = k if k is not None else self.smiles
        return self._key

    @property
    def rdmol(self):
        if self._rdmol is None:
            self._rdmol = Chem.MolFromSmiles(self.smiles)
        return self._rdmol

    @property
    def num_heavy_atoms(self) -> int:
        return self.rdmol.GetNumHeavyAtoms()

    def __eq__(self, other) -> bool:
        return isinstance(other, Molecule) and other.smiles == self.smiles

    def __hash__(self) -> int:
        return hash(self.smiles)

    def __repr__(self) -> str:
        return f"Molecule({self.smiles!r})"

    def __str__(self) -> str:
        return self.smiles


__all__ = [
    "Molecule",
    "MoleculeError",
    "canonicalize",
    "inchi_key",
    "split_components",
]
