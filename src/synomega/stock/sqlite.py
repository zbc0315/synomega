"""SQLite-backed stock, for catalogues too large to hold in RAM.

Slower per lookup than `InMemoryStock` but bounded memory. Lookups are batched
where possible and cached in a small LRU, since search re-tests the same
molecules repeatedly.

Deliberately no Bloom-filter backend: false positives would inflate solve-rate
and quietly break comparability with published numbers, which is the main
reason for standardizing on eMolecules in the first place.
"""

from __future__ import annotations

import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

from ..chem.mol import Molecule
from .base import BuildingBlockSet


class SqliteStock(BuildingBlockSet):
    """Purchasable InChIKeys stored in a SQLite table."""

    name = "sqlite"

    def __init__(self, path: str | Path, *, cache_size: int = 200_000):
        self.path = str(path)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS stock (key TEXT PRIMARY KEY)")
        self._db.commit()
        self._cache: OrderedDict[str, bool] = OrderedDict()
        self._cache_size = cache_size
        self._len: int | None = None

    # ------------------------------------------------------------- building

    @classmethod
    def build(
        cls,
        path: str | Path,
        smiles: Iterable[str],
        *,
        batch: int = 50_000,
    ) -> "SqliteStock":
        """Create a stock database from a SMILES iterable."""
        store = cls(path)
        buf: list[tuple[str]] = []
        for smi in smiles:
            mol = Molecule.try_of(smi)
            if mol is None:
                continue
            buf.append((mol.key,))
            if len(buf) >= batch:
                store._insert(buf)
                buf.clear()
        if buf:
            store._insert(buf)
        store._db.execute("ANALYZE")
        store._db.commit()
        store._len = None
        return store

    def _insert(self, rows: list[tuple[str]]) -> None:
        self._db.executemany("INSERT OR IGNORE INTO stock (key) VALUES (?)", rows)
        self._db.commit()

    # ------------------------------------------------------------------ api

    def __contains__(self, mol: str | Molecule) -> bool:
        key = self._key(mol)
        if key is None:
            return False
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        row = self._db.execute(
            "SELECT 1 FROM stock WHERE key = ? LIMIT 1", (key,)
        ).fetchone()
        found = row is not None
        self._cache[key] = found
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return found

    def contains_batch(self, mols: list[str | Molecule]) -> list[bool]:
        keys = [self._key(m) for m in mols]
        unknown = {k for k in keys if k is not None and k not in self._cache}
        if unknown:
            todo = list(unknown)
            found: set[str] = set()
            # SQLite caps variables per statement; chunk well under the limit.
            for i in range(0, len(todo), 500):
                chunk = todo[i:i + 500]
                marks = ",".join("?" * len(chunk))
                rows = self._db.execute(
                    f"SELECT key FROM stock WHERE key IN ({marks})", chunk
                ).fetchall()
                found.update(r[0] for r in rows)
            for k in todo:
                self._cache[k] = k in found
        return [bool(k is not None and self._cache.get(k, False)) for k in keys]

    def __len__(self) -> int:
        if self._len is None:
            self._len = int(
                self._db.execute("SELECT count(*) FROM stock").fetchone()[0]
            )
        return self._len

    def close(self) -> None:
        self._db.close()


__all__ = ["SqliteStock"]
