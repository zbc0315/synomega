"""Multi-component evolution (MCE): grow a forward synthesis network.

Starting from a set of reactant molecules, repeatedly pick two molecules from a
growing pool, run the forward model on the pair, and add the predicted products
back to the pool. Each molecule carries:

* a **total synthesis score** — starting reactants are 1.0; a product's score is
  ``min(score of its two reactants) * (forward probability of that product)``;
* a **synthesis depth** — the depth of the molecule in the synthesis *tree*
  (``max(depth of the two reactants) + 1``), not the number of steps.

Expansion is *generational*: each round pairs the currently selectable molecules
(score >= threshold, depth < max_depth), prioritising high-score pairs, runs the
forward model in batches, and merges the products. Because a reactant's score can
improve after products were already made from it, scores are **propagated** along
the recorded reaction network (a cheap max-product relaxation, no extra model
calls), so a molecule is never left under-scored — and hence never wrongly pruned
below the threshold. All reaction edges are kept, so the result is a genuine
network, not just one best route per molecule.

Two backends, identical results, differ only in where data lives:

* ``mode="memory"`` — pool/edges/pairs held in RAM. For a handful of reactants.
* ``mode="disk"`` — pool/edges/pairs in a SQLite database under ``work_dir``, for
  many starting reactants whose intermediates do not fit in RAM.

At scale the real cost is the O(n^2) pairing; ``disk`` mode fixes *storage*, and
``frontier_width`` (top-N selectable molecules paired per round) fixes *fan-out*.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from ..chem.mol import canonicalize, inchi_key
from .base import ForwardModel

# Floats that differ by less than this are treated as equal when deciding whether
# a propagated score is an improvement — guards the relaxation against dithering.
_EPS = 1e-12
# Placeholder marks for a product node that exists but has not been scored yet;
# any real reaction edge improves on these, so relaxation always overwrites them.
_UNSET_SCORE = -1.0
_UNSET_DEPTH = 1 << 30


@dataclass
class PoolMolecule:
    """One molecule in the evolving pool."""

    smiles: str                       # canonical SMILES (identity / primary key)
    total_score: float                # starting reactants: 1.0
    depth: int                        # synthesis-tree depth; starting reactants: 0
    step_score: float | None = None   # forward prob of the best-scoring reaction
    parent_a: str | None = None       # the two reactants of the best-scoring edge
    parent_b: str | None = None
    template_id: int | None = None
    inchi_key: str | None = None      # cached for stock interop; not the pool key

    @property
    def is_source(self) -> bool:
        return self.parent_a is None


@dataclass(frozen=True)
class Edge:
    """One recorded reaction ``a + b -> product`` with its forward probability."""

    product: str
    a: str
    b: str
    step_score: float
    template_id: int | None = None

    @property
    def reaction_smiles(self) -> str:
        return f"{self.a}.{self.b}>>{self.product}"


# --------------------------------------------------------------------- stores


class EvolutionStore(ABC):
    """Storage for the pool, the reaction edges and the reacted-pair set.

    The evolution algorithm runs entirely against this interface, so the two
    backends (RAM / SQLite) produce identical results and differ only in memory.
    """

    # -- pool --------------------------------------------------------------
    @abstractmethod
    def get(self, smiles: str) -> PoolMolecule | None: ...
    @abstractmethod
    def upsert(self, mol: PoolMolecule) -> None: ...
    @abstractmethod
    def ensure_placeholder(self, smiles: str) -> None:
        """Create an unscored node if `smiles` is absent; leave it otherwise."""
    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def reactable(
        self, threshold: float, max_depth: int, width: int | None
    ) -> list[tuple[str, float, int]]:
        """`(smiles, total, depth)` for molecules that may act as a reactant.

        Sorted by total desc then smiles asc (deterministic). At most `width`.
        """
    @abstractmethod
    def iter_all(self) -> Iterator[PoolMolecule]: ...
    @abstractmethod
    def top(
        self, n: int, min_depth: int, min_score: float
    ) -> list[PoolMolecule]: ...

    # -- edges -------------------------------------------------------------
    @abstractmethod
    def add_edges(self, edges: list[Edge]) -> None: ...
    @abstractmethod
    def incoming(self, product: str) -> list[Edge]:
        """Reaction edges that produce `product`."""
    @abstractmethod
    def dependents(self, mol: str) -> list[str]:
        """Products that use `mol` as a reactant (for downstream propagation)."""
    @abstractmethod
    def iter_edges(self) -> Iterator[Edge]: ...
    @abstractmethod
    def num_edges(self) -> int: ...

    # -- reacted pairs -----------------------------------------------------
    @abstractmethod
    def reacted_contains(self, a: str, b: str) -> bool: ...
    @abstractmethod
    def add_reacted(self, pairs: Iterable[tuple[str, str]]) -> None: ...

    def close(self) -> None:  # pragma: no cover - overridden by disk backend
        pass


class InMemoryStore(EvolutionStore):
    """Everything in RAM. Fast; for a handful of starting reactants."""

    def __init__(self) -> None:
        self._pool: dict[str, PoolMolecule] = {}
        self._incoming: dict[str, list[Edge]] = {}
        self._dependents: dict[str, set[str]] = {}
        self._reacted: set[tuple[str, str]] = set()

    def get(self, smiles):
        return self._pool.get(smiles)

    def upsert(self, mol):
        self._pool[mol.smiles] = mol

    def ensure_placeholder(self, smiles):
        if smiles not in self._pool:
            self._pool[smiles] = PoolMolecule(smiles, _UNSET_SCORE, _UNSET_DEPTH)

    def __len__(self):
        return len(self._pool)

    def reactable(self, threshold, max_depth, width):
        rows = [
            (m.smiles, m.total_score, m.depth)
            for m in self._pool.values()
            if m.total_score >= threshold and m.depth < max_depth
        ]
        rows.sort(key=lambda r: (-r[1], r[0]))
        return rows if width is None else rows[:width]

    def iter_all(self):
        return iter(list(self._pool.values()))

    def top(self, n, min_depth, min_score):
        rows = [
            m for m in self._pool.values()
            if m.depth >= min_depth and m.total_score >= min_score
        ]
        rows.sort(key=lambda m: (-m.total_score, m.smiles))
        return rows[:n]

    def add_edges(self, edges):
        for e in edges:
            self._incoming.setdefault(e.product, []).append(e)
            self._dependents.setdefault(e.a, set()).add(e.product)
            self._dependents.setdefault(e.b, set()).add(e.product)

    def incoming(self, product):
        return list(self._incoming.get(product, ()))

    def dependents(self, mol):
        return list(self._dependents.get(mol, ()))

    def iter_edges(self):
        for edges in self._incoming.values():
            yield from edges

    def num_edges(self):
        return sum(len(v) for v in self._incoming.values())

    def reacted_contains(self, a, b):
        return (a, b) in self._reacted

    def add_reacted(self, pairs):
        self._reacted.update(pairs)


class SqliteStore(EvolutionStore):
    """Pool, edges and reacted pairs in a SQLite database under `work_dir`.

    Intermediates spill to disk instead of RAM. Reads for the pairing frontier
    and the relaxation are indexed; a round's products are merged in one
    transaction.
    """

    def __init__(self, work_dir: str | Path) -> None:
        import sqlite3

        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.work_dir / "evolution.sqlite"
        self._db = sqlite3.connect(str(self.path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        # Start from a clean database every run: an evolve() is a fresh expansion,
        # and stale molecules/edges/reacted from a previous run pointing at the
        # same work_dir would silently corrupt the result (and diverge from the
        # in-memory backend, which always starts empty).
        self._db.executescript(
            "DROP TABLE IF EXISTS molecules;"
            "DROP TABLE IF EXISTS edges;"
            "DROP TABLE IF EXISTS reacted;"
        )
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS molecules (
                smiles TEXT PRIMARY KEY, total REAL, depth INTEGER,
                step REAL, parent_a TEXT, parent_b TEXT,
                template_id INTEGER, inchi_key TEXT);
            CREATE INDEX IF NOT EXISTS idx_mol_total ON molecules(total DESC);
            CREATE TABLE IF NOT EXISTS edges (
                product TEXT, a TEXT, b TEXT, step REAL, template_id INTEGER);
            CREATE INDEX IF NOT EXISTS idx_edge_product ON edges(product);
            CREATE INDEX IF NOT EXISTS idx_edge_a ON edges(a);
            CREATE INDEX IF NOT EXISTS idx_edge_b ON edges(b);
            CREATE TABLE IF NOT EXISTS reacted (
                a TEXT, b TEXT, PRIMARY KEY(a, b));
            """
        )
        self._db.commit()

    @staticmethod
    def _row(r) -> PoolMolecule:
        return PoolMolecule(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7])

    def get(self, smiles):
        r = self._db.execute(
            "SELECT smiles,total,depth,step,parent_a,parent_b,template_id,inchi_key"
            " FROM molecules WHERE smiles=?",
            (smiles,),
        ).fetchone()
        return self._row(r) if r is not None else None

    def upsert(self, mol):
        self._db.execute(
            "INSERT INTO molecules"
            " (smiles,total,depth,step,parent_a,parent_b,template_id,inchi_key)"
            " VALUES (?,?,?,?,?,?,?,?)"
            " ON CONFLICT(smiles) DO UPDATE SET"
            " total=excluded.total, depth=excluded.depth, step=excluded.step,"
            " parent_a=excluded.parent_a, parent_b=excluded.parent_b,"
            " template_id=excluded.template_id, inchi_key=excluded.inchi_key",
            (mol.smiles, mol.total_score, mol.depth, mol.step_score,
             mol.parent_a, mol.parent_b, mol.template_id, mol.inchi_key),
        )
        self._db.commit()

    def ensure_placeholder(self, smiles):
        self._db.execute(
            "INSERT OR IGNORE INTO molecules (smiles,total,depth) VALUES (?,?,?)",
            (smiles, _UNSET_SCORE, _UNSET_DEPTH),
        )
        self._db.commit()

    def __len__(self):
        return int(self._db.execute("SELECT count(*) FROM molecules").fetchone()[0])

    def reactable(self, threshold, max_depth, width):
        sql = (
            "SELECT smiles,total,depth FROM molecules"
            " WHERE total>=? AND depth<? ORDER BY total DESC, smiles ASC"
        )
        params: tuple = (threshold, max_depth)
        if width is not None:
            sql += " LIMIT ?"
            params = (threshold, max_depth, width)
        return [(r[0], r[1], r[2]) for r in self._db.execute(sql, params)]

    def iter_all(self):
        cur = self._db.execute(
            "SELECT smiles,total,depth,step,parent_a,parent_b,template_id,inchi_key"
            " FROM molecules"
        )
        for r in cur:
            yield self._row(r)

    def top(self, n, min_depth, min_score):
        cur = self._db.execute(
            "SELECT smiles,total,depth,step,parent_a,parent_b,template_id,inchi_key"
            " FROM molecules WHERE depth>=? AND total>=?"
            " ORDER BY total DESC, smiles ASC LIMIT ?",
            (min_depth, min_score, n),
        )
        return [self._row(r) for r in cur]

    def add_edges(self, edges):
        self._db.executemany(
            "INSERT INTO edges (product,a,b,step,template_id) VALUES (?,?,?,?,?)",
            [(e.product, e.a, e.b, e.step_score, e.template_id) for e in edges],
        )
        self._db.commit()

    def incoming(self, product):
        cur = self._db.execute(
            "SELECT product,a,b,step,template_id FROM edges WHERE product=?",
            (product,),
        )
        return [Edge(r[0], r[1], r[2], r[3], r[4]) for r in cur]

    def dependents(self, mol):
        cur = self._db.execute(
            "SELECT DISTINCT product FROM edges WHERE a=? OR b=?", (mol, mol)
        )
        return [r[0] for r in cur]

    def iter_edges(self):
        cur = self._db.execute("SELECT product,a,b,step,template_id FROM edges")
        for r in cur:
            yield Edge(r[0], r[1], r[2], r[3], r[4])

    def num_edges(self):
        return int(self._db.execute("SELECT count(*) FROM edges").fetchone()[0])

    def reacted_contains(self, a, b):
        return self._db.execute(
            "SELECT 1 FROM reacted WHERE a=? AND b=? LIMIT 1", (a, b)
        ).fetchone() is not None

    def add_reacted(self, pairs):
        self._db.executemany(
            "INSERT OR IGNORE INTO reacted (a,b) VALUES (?,?)", list(pairs)
        )
        self._db.commit()

    def close(self):
        self._db.close()


# --------------------------------------------------------------------- result


@dataclass
class EvolutionResult:
    """Outcome of an evolution run. Backed by a store (RAM or SQLite).

    For the disk backend the store keeps an open database handle, so the queries
    below stay off-heap; call :meth:`close` when finished.
    """

    store: EvolutionStore
    stats: dict

    @property
    def num_molecules(self) -> int:
        return len(self.store)

    def molecules(self) -> Iterator[PoolMolecule]:
        return self.store.iter_all()

    def reactions(self) -> Iterator[Edge]:
        return self.store.iter_edges()

    def top(
        self, n: int = 20, *, min_depth: int = 0, min_score: float = 0.0
    ) -> list[PoolMolecule]:
        """Highest-scoring molecules, optionally restricted to real products."""
        return self.store.top(n, min_depth, min_score)

    def best_route(self, smiles: str) -> list[Edge]:
        """Reactions of the best-scoring route to `smiles`, sources last.

        Follows each molecule's best-scoring parent edge back to the starting
        reactants. Returns [] for an unknown or starting molecule.
        """
        route: list[Edge] = []
        seen: set[str] = set()
        stack = [smiles]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            mol = self.store.get(cur)
            if mol is None or mol.parent_a is None:
                continue
            route.append(
                Edge(cur, mol.parent_a, mol.parent_b, mol.step_score or 0.0,
                     mol.template_id)
            )
            stack.extend([mol.parent_a, mol.parent_b])
        return route

    def to_dict(self, *, max_molecules: int = 1000) -> dict:
        mols = self.top(max_molecules)
        return {
            "stats": self.stats,
            "num_molecules": self.num_molecules,
            # distinct reaction edges (one reactant pair can yield several); the
            # number of reactant *pairs* actually run is stats["num_reactions"].
            "num_reaction_edges": self.store.num_edges(),
            "molecules": [
                {
                    "smiles": m.smiles,
                    "total_score": m.total_score,
                    "depth": m.depth,
                    "step_score": m.step_score,
                    "parents": (
                        None if m.parent_a is None else [m.parent_a, m.parent_b]
                    ),
                    "template_id": m.template_id,
                }
                for m in mols
            ],
        }

    def to_json(self, *, indent: int = 2, max_molecules: int = 1000) -> str:
        import json

        return json.dumps(self.to_dict(max_molecules=max_molecules), indent=indent)

    def describe(self, *, n: int = 15) -> str:
        lines = [
            f"molecules: {self.num_molecules}"
            f"  pairs-run: {self.stats.get('num_reactions')}"
            f"  reaction-edges: {self.store.num_edges()}"
            f"  rounds: {self.stats.get('rounds')}"
            f"  stop: {self.stats.get('termination')}",
            f"top {n} products by total score:",
        ]
        for m in self.top(n, min_depth=1):
            line = f"  {m.total_score:.4f}  d{m.depth}  {m.smiles}"
            if m.step_score is not None:
                line += f"  (step={m.step_score:.4f})"
            lines.append(line)
        return "\n".join(lines)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "EvolutionResult":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ------------------------------------------------------------------- evolver


class MultiComponentEvolution:
    """Grow a forward synthesis network from starting reactants.

    See the module docstring for the algorithm. Construct with any
    :class:`~synomega.forward.base.ForwardModel` (tests pass a fake one).
    """

    def __init__(
        self,
        model: ForwardModel,
        *,
        max_depth: int,
        score_threshold: float,
        mode: str = "memory",
        work_dir: str | Path | None = None,
        forward_top_k: int = 5,
        allow_self_pair: bool = True,
        frontier_width: int | None = None,
        max_reactions: int | None = 2_000_000,
        max_pool_size: int | None = None,
        batch_size: int = 256,
        auto_disk_threshold: int = 200,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        if mode not in ("memory", "disk", "auto"):
            raise ValueError("mode must be 'memory', 'disk' or 'auto'")
        self.model = model
        self.max_depth = max_depth
        self.score_threshold = score_threshold
        self.mode = mode
        self.work_dir = work_dir
        self.forward_top_k = forward_top_k
        self.allow_self_pair = allow_self_pair
        self.frontier_width = frontier_width
        self.max_reactions = max_reactions
        self.max_pool_size = max_pool_size
        self.batch_size = batch_size
        self.auto_disk_threshold = auto_disk_threshold

    # ------------------------------------------------------------- backend
    def _make_store(self, n_sources: int) -> EvolutionStore:
        mode = self.mode
        if mode == "auto":
            mode = "disk" if n_sources > self.auto_disk_threshold else "memory"
        if mode == "disk":
            if self.work_dir is None:
                raise ValueError("mode='disk' requires work_dir")
            return SqliteStore(self.work_dir)
        return InMemoryStore()

    # --------------------------------------------------------------- run
    def evolve(self, reactants: list[str]) -> EvolutionResult:
        # canonicalize starting reactants, dropping unparseable ones (do not let
        # a None key poison the pool).
        sources: set[str] = set()
        for r in reactants:
            canon = canonicalize(r)
            if canon is None:
                warnings.warn(f"skipping unparseable reactant SMILES: {r!r}")
                continue
            sources.add(canon)
        if not sources:
            raise ValueError("no parseable starting reactants")

        store = self._make_store(len(sources))
        try:
            stats = self._run(store, sources)
        except BaseException:
            store.close()  # do not leak the SQLite handle on failure
            raise
        return EvolutionResult(store, stats)

    def _run(self, store: EvolutionStore, sources: set[str]) -> dict:
        self._sources = sources
        for s in sources:
            store.upsert(
                PoolMolecule(s, 1.0, 0, inchi_key=inchi_key(s))
            )

        changed = set(sources)
        rounds = 0
        n_reactions = 0
        termination = "exhausted"

        while True:
            reactable = store.reactable(
                self.score_threshold, self.max_depth, self.frontier_width
            )
            totals = {s: t for s, t, _ in reactable}
            reactable_smiles = [s for s, _, _ in reactable]
            changed_r = [s for s in reactable_smiles if s in changed]

            pairs = self._new_pairs(changed_r, reactable_smiles, totals, store)
            if not pairs:
                termination = "exhausted"
                break

            if self.max_reactions is not None:
                room = self.max_reactions - n_reactions
                if room <= 0:
                    termination = "max_reactions"
                    break
                if len(pairs) > room:
                    pairs = pairs[:room]
                    termination = "max_reactions"

            store.add_reacted(pairs)
            n_reactions += len(pairs)
            rounds += 1

            new_products = self._react_and_record(pairs, store)
            changed = self._relax(store, new_products)

            hit_cap = termination == "max_reactions"
            if (
                self.max_pool_size is not None
                and len(store) >= self.max_pool_size
            ):
                termination = "max_pool_size"
                hit_cap = True
            if hit_cap:
                break

        return {
            "rounds": rounds,
            "num_reactions": n_reactions,
            "num_molecules": len(store),
            "num_sources": len(sources),
            "termination": termination,
        }

    # ---------------------------------------------------------- pairing
    def _new_pairs(
        self,
        changed_r: list[str],
        reactable_smiles: list[str],
        totals: dict[str, float],
        store: EvolutionStore,
    ) -> list[tuple[str, str]]:
        """Not-yet-reacted pairs with at least one changed endpoint.

        Sorted by priority = min(total of the two) desc — the upper bound on the
        product's total score — so the highest-potential pairs react first.
        """
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str]] = []
        for x in changed_r:
            for y in reactable_smiles:
                a, b = (x, y) if x <= y else (y, x)
                if a == b and not self.allow_self_pair:
                    continue
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                if store.reacted_contains(a, b):
                    continue
                out.append(key)
        out.sort(key=lambda p: (-min(totals[p[0]], totals[p[1]]), p[0], p[1]))
        return out

    # -------------------------------------------------------- prediction
    def _react_and_record(
        self, pairs: list[tuple[str, str]], store: EvolutionStore
    ) -> set[str]:
        """Run the forward model on every pair; record edges + product nodes."""
        new_products: set[str] = set()
        edges: list[Edge] = []
        bs = self.batch_size
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            inputs = [f"{a}.{b}" for a, b in chunk]
            batch = self.model.predict_batch(inputs, top_k=self.forward_top_k)
            for (a, b), preds in zip(chunk, batch):
                for pred in preds:
                    prod = canonicalize(pred.product)
                    if prod is None or prod == a or prod == b:
                        continue
                    edges.append(Edge(prod, a, b, float(pred.score),
                                      pred.template_id))
                    new_products.add(prod)
        for p in new_products:
            store.ensure_placeholder(p)
        store.add_edges(edges)
        return new_products

    # -------------------------------------------------------- propagation
    def _relax(self, store: EvolutionStore, seeds: set[str]) -> set[str]:
        """Max-product relaxation over the reaction network.

        A molecule's total is ``max`` over its incoming edges of
        ``min(parent totals) * step``; its depth follows the best edge
        (``max(parent depths)+1``, ties broken to the smaller depth). Starting
        reactants are pinned at (1.0, 0). Improvements flow to dependents until
        a fixpoint. Returns the molecules that became (or stayed) selectable and
        whose score/depth improved this round — the next round's frontier.
        """
        touched: set[str] = set()
        work: deque[str] = deque(seeds)
        queued = set(seeds)
        while work:
            m = work.popleft()
            queued.discard(m)
            if m in self._sources:
                continue  # pinned at (1.0, 0)
            mol = store.get(m)
            if mol is None:
                continue

            best_total = best_depth = None
            best: Edge | None = None
            for e in store.incoming(m):
                pa = store.get(e.a)
                pb = store.get(e.b)
                if pa is None or pb is None:
                    continue
                if pa.total_score <= _UNSET_SCORE or pb.total_score <= _UNSET_SCORE:
                    continue  # parent not scored yet
                t = min(pa.total_score, pb.total_score) * e.step_score
                d = max(pa.depth, pb.depth) + 1
                # Prefer higher total; break ties by shallower depth, then by
                # (a, b) lexicographically — a pure function of the edges, so the
                # fixpoint is deterministic and identical across backends.
                if best is None:
                    better = True
                elif t > best_total + _EPS:
                    better = True
                elif abs(t - best_total) <= _EPS:
                    better = d < best_depth or (
                        d == best_depth and (e.a, e.b) < (best.a, best.b)
                    )
                else:
                    better = False
                if better:
                    best_total, best_depth, best = t, d, e
            if best is None:
                continue

            improved = (
                best_total > mol.total_score + _EPS
                or (
                    abs(best_total - mol.total_score) <= _EPS
                    and best_depth < mol.depth
                )
            )
            if not improved:
                continue
            mol.total_score = best_total
            mol.depth = best_depth
            mol.step_score = best.step_score
            mol.parent_a = best.a
            mol.parent_b = best.b
            mol.template_id = best.template_id
            if mol.inchi_key is None:
                mol.inchi_key = inchi_key(m)
            store.upsert(mol)

            if (
                mol.total_score >= self.score_threshold
                and mol.depth < self.max_depth
            ):
                touched.add(m)
            for dep in store.dependents(m):
                if dep not in queued:
                    work.append(dep)
                    queued.add(dep)
        return touched


def build_evolver(model: ForwardModel, **kwargs) -> MultiComponentEvolution:
    """Convenience constructor mirroring the other backends' factories."""
    return MultiComponentEvolution(model, **kwargs)


__all__ = [
    "PoolMolecule",
    "Edge",
    "EvolutionStore",
    "InMemoryStore",
    "SqliteStore",
    "EvolutionResult",
    "MultiComponentEvolution",
    "build_evolver",
]
