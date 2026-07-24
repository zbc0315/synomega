"""Caching wrapper for single-step models.

Search revisits the same molecule constantly — the same intermediate shows up
in many branches, and batch synthesizability scoring re-encounters common
fragments across targets. Caching expansions is the single cheapest speedup
available.

Two tiers:
  * in-process LRU (always on)
  * optional SQLite file, so the cache survives across runs and processes
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path

from .base import Prediction, SingleStepModel


class CachedModel(SingleStepModel):
    """Memoizes `predict` results keyed by (smiles, top_k).

    A cached entry computed at top_k=N also serves any request with top_k <= N,
    truncated — so a search that varies its beam width still hits the cache.
    """

    def __init__(
        self,
        model: SingleStepModel,
        *,
        max_size: int = 100_000,
        disk_path: str | Path | None = None,
    ):
        self.model = model
        self.name = f"cached({model.name})"
        self.max_size = max_size
        self._lru: OrderedDict[str, tuple[int, list[Prediction]]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

        self._db: sqlite3.Connection | None = None
        if disk_path is not None:
            self._db = sqlite3.connect(str(disk_path), check_same_thread=False)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS expansions ("
                "  smiles TEXT PRIMARY KEY, top_k INTEGER, payload TEXT)"
            )
            self._db.commit()

    # ---------------------------------------------------------------- lookup

    def _lru_get(self, smiles: str, top_k: int) -> list[Prediction] | None:
        with self._lock:
            hit = self._lru.get(smiles)
            if hit is None:
                return None
            cached_k, preds = hit
            if cached_k < top_k:
                return None  # cached at a narrower beam; must recompute
            self._lru.move_to_end(smiles)
            return preds[:top_k]

    def _lru_put(self, smiles: str, top_k: int, preds: list[Prediction]) -> None:
        with self._lock:
            self._lru[smiles] = (top_k, preds)
            self._lru.move_to_end(smiles)
            while len(self._lru) > self.max_size:
                self._lru.popitem(last=False)

    def _disk_get(self, smiles: str, top_k: int) -> list[Prediction] | None:
        if self._db is None:
            return None
        row = self._db.execute(
            "SELECT top_k, payload FROM expansions WHERE smiles = ?", (smiles,)
        ).fetchone()
        if row is None or row[0] < top_k:
            return None
        preds = [
            Prediction(
                reactants=tuple(d["reactants"]),
                score=d["score"],
                template_id=d.get("template_id"),
            )
            for d in json.loads(row[1])
        ]
        return preds[:top_k]

    def _disk_put(self, smiles: str, top_k: int, preds: list[Prediction]) -> None:
        if self._db is None:
            return
        payload = json.dumps(
            [
                {
                    "reactants": list(p.reactants),
                    "score": p.score,
                    "template_id": p.template_id,
                }
                for p in preds
            ]
        )
        self._db.execute(
            "INSERT OR REPLACE INTO expansions (smiles, top_k, payload) VALUES (?,?,?)",
            (smiles, top_k, payload),
        )
        self._db.commit()

    # ---------------------------------------------------------------- api

    def predict(self, smiles: str, top_k: int = 50) -> list[Prediction]:
        hit = self._lru_get(smiles, top_k)
        if hit is not None:
            self.hits += 1
            return hit

        hit = self._disk_get(smiles, top_k)
        if hit is not None:
            self.hits += 1
            self._lru_put(smiles, top_k, hit)
            return hit

        self.misses += 1
        preds = self.model.predict(smiles, top_k)
        self._lru_put(smiles, top_k, preds)
        self._disk_put(smiles, top_k, preds)
        return preds

    def predict_batch(
        self, smiles: list[str], top_k: int = 50
    ) -> list[list[Prediction]]:
        results: list[list[Prediction] | None] = [None] * len(smiles)
        todo_idx: list[int] = []
        todo_smiles: list[str] = []

        for i, s in enumerate(smiles):
            hit = self._lru_get(s, top_k) or self._disk_get(s, top_k)
            if hit is not None:
                self.hits += 1
                results[i] = hit
            else:
                todo_idx.append(i)
                todo_smiles.append(s)

        if todo_smiles:
            self.misses += len(todo_smiles)
            fresh = self.model.predict_batch(todo_smiles, top_k)
            for i, s, preds in zip(todo_idx, todo_smiles, fresh):
                results[i] = preds
                self._lru_put(s, top_k, preds)
                self._disk_put(s, top_k, preds)

        return [r if r is not None else [] for r in results]

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None


__all__ = ["CachedModel"]
