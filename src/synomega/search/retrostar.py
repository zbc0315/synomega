"""Retro*-style search (Chen et al., ICML 2020).

The difference from plain best-first is *where the cost comes from*. Retro*
maintains, for every molecule node, an estimate `V` of the total cost to
synthesize the whole target through that node, and always expands the frontier
molecule with the lowest such estimate.

Concretely:

    rn(m)  = min over reactions r of m of  [ cost(r) + sum_{c in children(r)} rn(c) ]
             (rn(m) = 0 if m is in stock; = value(m) if m is unexpanded)

    V(m)   = rn(m) + delta(m)

where `delta(m)` is the cost the rest of the tree contributes — what it takes to
finish everything *else* the target needs, given we go through `m`. Expanding
the minimum-`V` frontier node focuses effort on the disconnection that most
improves the global route, not merely the locally cheap one.

`rn` and `delta` are refreshed after each expansion by propagating upward from
the touched node. This is an honest but not heavily optimized implementation:
it recomputes ancestors rather than maintaining incremental invariants.
"""

from __future__ import annotations

import itertools
import math

from .base import Budget, SearchAlgorithm, SearchResult, SearchStats
from .graph import AndOrGraph, MolNode
from .value import ValueFunction, get_value_function

INF = float("inf")


class RetroStar(SearchAlgorithm):
    """AND-OR best-first search on total-route cost."""

    name = "retrostar"

    def __init__(
        self,
        model,
        stock,
        *,
        expansion_width: int = 50,
        value_function: "str | ValueFunction | None" = None,
        batch_size: int = 32,
    ):
        super().__init__(model, stock, expansion_width=expansion_width)
        self.value_function = get_value_function(value_function)
        self.batch_size = batch_size
        self._rn: dict[str, float] = {}

    # ------------------------------------------------------------------ run

    def run(
        self, target: str, budget: Budget, *, exclude_target: bool = False
    ) -> SearchResult:
        budget.start()
        stats = SearchStats()
        graph = self._make_graph(target, exclude_target=exclude_target)
        root = graph.root

        if root.solved:
            stats.elapsed_s = budget.elapsed
            stats.terminated_by = "solved"
            return SearchResult(target, True, graph, stats, self.name)

        root.value = self.value_function(root)
        self._rn = {root.key: root.value}
        reason = "exhausted"

        while True:
            stop = budget.exhausted(stats.expansions)
            if stop:
                reason = stop
                break

            frontier = self._select_frontier(graph, budget)
            if not frontier:
                break

            preds_batch = self.model.predict_batch(
                [n.smiles for n in frontier], self.expansion_width
            )
            stats.model_calls += 1

            for node, preds in zip(frontier, preds_batch):
                node.expanded = True
                stats.expansions += 1
                self._attach(graph, node, preds, stats, budget)

            self._refresh(graph)

            if root.solved:
                stats.elapsed_s = budget.elapsed
                stats.terminated_by = "solved"
                return SearchResult(target, True, graph, stats, self.name)

        stats.elapsed_s = budget.elapsed
        stats.terminated_by = "solved" if root.solved else reason
        return SearchResult(target, root.solved, graph, stats, self.name)

    # ------------------------------------------------------------ selection

    def _select_frontier(self, graph: AndOrGraph, budget: Budget) -> list[MolNode]:
        """Lowest-V unexpanded molecules, up to batch_size."""
        candidates: list[tuple[float, int, MolNode]] = []
        counter = itertools.count()
        for node in graph.nodes.values():
            if node.solved or node.expanded or not node.expandable:
                continue
            if node.depth >= budget.max_depth:
                node.expandable = False
                continue
            v = self._total_cost(graph, node)
            if v < INF:
                candidates.append((v, next(counter), node))
        if not candidates:
            return []
        candidates.sort(key=lambda t: t[0])
        return [n for _, _, n in candidates[: self.batch_size]]

    def _total_cost(self, graph: AndOrGraph, node: MolNode) -> float:
        """V(node) = rn(node) + delta(node)."""
        return self._rn.get(node.key, node.value) + self._delta(graph, node)

    def _delta(self, graph: AndOrGraph, node: MolNode) -> float:
        """Cost the rest of the tree adds, along the cheapest route to `node`.

        Walks up to the root; at each reaction the siblings' `rn` and the
        reaction's own cost are added. When a node has several parents we take
        the cheapest, since we only need one viable route.
        """
        if node is graph.root:
            return 0.0
        best = INF
        for rxn in node.parents:
            siblings = sum(
                self._rn.get(c.key, c.value) for c in rxn.children if c is not node
            )
            up = self._delta(graph, rxn.parent)
            if up == INF:
                continue
            best = min(best, rxn.cost + siblings + up)
        return best

    # ---------------------------------------------------------- propagation

    def _refresh(self, graph: AndOrGraph) -> None:
        """Recompute rn() for every node, bottom-up until it stops changing.

        Simple fixpoint iteration. Graphs here are small (hundreds to a few
        thousand nodes), so this is cheap next to a single model call.
        """
        for _ in range(64):
            changed = False
            for node in graph.nodes.values():
                new = self._rn_of(node)
                old = self._rn.get(node.key)
                if old is None or abs(new - old) > 1e-9:
                    self._rn[node.key] = new
                    changed = True
            if not changed:
                break

    def _rn_of(self, node: MolNode) -> float:
        if node.in_stock:
            return 0.0
        if not node.children:
            # Unexpanded: fall back to the heuristic. Expanded-but-childless
            # means a dead end.
            return node.value if not node.expanded else INF
        best = INF
        for rxn in node.children:
            total = rxn.cost
            for child in rxn.children:
                total += self._rn.get(child.key, child.value)
                if total >= INF:
                    break
            best = min(best, total)
        return best

    # -------------------------------------------------------------- attach

    def _attach(self, graph, node, preds, stats, budget) -> None:
        child_depth = node.depth + 1
        for pred in preds:
            made = self._child_nodes(graph, pred.reactants, child_depth)
            if made is None:
                continue
            children, created = made
            rxn = graph.add_reaction(
                node, children, pred.score, pred.template_id, meta=dict(pred.meta)
            )
            if rxn is None:
                continue
            stats.nodes_created += created
            stats.reactions_created += 1

            for child in children:
                if child.key not in self._rn:
                    child.value = self.value_function(child)
                    self._rn[child.key] = 0.0 if child.in_stock else child.value
                if child.in_stock and not child.solved:
                    graph.mark_solved(child)

            if rxn.solved:
                graph.mark_solved(node)


__all__ = ["RetroStar"]
