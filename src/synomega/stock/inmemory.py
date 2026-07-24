"""In-memory building-block stock — the default backend.

Holds a `set` of InChIKeys. At eMolecules scale (~10^7 entries) this costs
roughly 1-2 GB and gives O(1) membership, which is what search needs since it
tests stock membership on every node it touches.

Loading a raw vendor catalogue means computing 10^7 InChIKeys, which is slow
(tens of minutes single-threaded). Do it once and `save_keys()` to a plain
key-per-line file; `from_keys_file()` then loads in seconds.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable, Iterator

from ..chem.mol import Molecule
from .base import BuildingBlockSet


def _open(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return path.open()


class InMemoryStock(BuildingBlockSet):
    """A set of purchasable InChIKeys."""

    name = "inmemory"

    def __init__(self, keys: Iterable[str] | None = None, *, name: str | None = None):
        self._keys: set[str] = set(keys) if keys is not None else set()
        if name:
            self.name = name

    # ------------------------------------------------------------- building

    @classmethod
    def from_smiles(
        cls,
        smiles: Iterable[str],
        *,
        name: str | None = None,
        on_error: str = "skip",
    ) -> "InMemoryStock":
        """Build from SMILES, converting each to an InChIKey."""
        keys: set[str] = set()
        for smi in smiles:
            mol = Molecule.try_of(smi)
            if mol is None:
                if on_error == "raise":
                    raise ValueError(f"unparseable stock SMILES: {smi!r}")
                continue
            keys.add(mol.key)
        return cls(keys, name=name)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        smiles_column: int = 0,
        delimiter: str | None = None,
        skip_header: bool = False,
        name: str | None = None,
    ) -> "InMemoryStock":
        """Load a SMILES catalogue (.smi/.txt/.csv/.tsv, optionally gzipped).

        eMolecules ships as a whitespace- or tab-delimited file whose first
        column is the SMILES; the defaults match that.
        """
        path = Path(path)

        def _iter() -> Iterator[str]:
            with _open(path) as fh:
                if skip_header:
                    next(fh, None)
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    fields = line.split(delimiter) if delimiter else line.split()
                    if len(fields) <= smiles_column:
                        continue
                    yield fields[smiles_column]

        return cls.from_smiles(_iter(), name=name or path.name)

    @classmethod
    def from_keys_file(
        cls, path: str | Path, *, name: str | None = None
    ) -> "InMemoryStock":
        """Load a precomputed one-InChIKey-per-line file (fast path)."""
        path = Path(path)
        with _open(path) as fh:
            keys = {line.strip() for line in fh if line.strip()}
        return cls(keys, name=name or path.name)

    def save_keys(self, path: str | Path) -> None:
        """Persist InChIKeys so future loads skip re-parsing the catalogue."""
        path = Path(path)
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "wt") as fh:
            for key in sorted(self._keys):
                fh.write(key + "\n")

    # ------------------------------------------------------------------ api

    def add(self, mol: str | Molecule) -> bool:
        key = self._key(mol)
        if key is None:
            return False
        self._keys.add(key)
        return True

    def __contains__(self, mol: str | Molecule) -> bool:
        key = self._key(mol)
        return key is not None and key in self._keys

    def __len__(self) -> int:
        return len(self._keys)

    def __or__(self, other: "InMemoryStock") -> "InMemoryStock":
        return InMemoryStock(self._keys | other._keys,
                             name=f"{self.name}+{other.name}")


__all__ = ["InMemoryStock"]
