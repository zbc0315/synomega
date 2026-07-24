"""Best-first search over the AND-OR graph.

Maintains a priority queue of unexpanded molecules ordered by
`g + h` — accumulated reaction cost plus the value function's estimate of what
remains. With `ZeroValue` this is uniform-cost search; with a heuristic it is
greedy A*.

Frontier nodes are expanded in batches so GPU-backed models get real work per
call. This is the workhorse: simple, predictable, and the reference the other
algorithms are checked against.
"""

from __future__ import annotations

import heapq
import itertools

from .base import Budget, SearchAlgorithm, SearchResult, SearchStats
from .graph import MolNode
from .value import ValueFunction, get_value_function


class BestFirstSearch(SearchAlgorithm):
    """Priority-queue expansion by estimated total cost."""

    name = "bfs"

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

    def run(
        self, target: str, budget: Budget, *, exclude_target: bool = False
    ) -> SearchResult:
        budget.start()
        stats = SearchStats()
        graph = self._make_graph(target, exclude_target=exclude_target)
        root = graph.root

        if root.solved:  # target itself is purchasable
            stats.elapsed_s = budget.elapsed
            stats.terminated_by = "solved"
            return SearchResult(target, True, graph, stats, self.name)

        counter = itertools.count()          # tie-break, keeps heap stable
        heap: list[tuple[float, int, MolNode]] = []
        queued: set[str] = set()

        def push(node: MolNode, g: float) -> None:
            if node.solved or node.expanded or not node.expandable:
                return
            if node.key in queued:
                return
            if node.depth >= budget.max_depth:
                node.expandable = False
                return
            queued.add(node.key)
            node.value = self.value_function(node)
            heapq.heappush(heap, (g + node.value, next(counter), node))

        push(root, 0.0)
        reason = "exhausted"

        while heap:
            stop = budget.exhausted(stats.expansions)
            if stop:
                reason = stop
                break

            # Pull a batch of frontier nodes so the model gets real batching.
            frontier: list[MolNode] = []
            while heap and len(frontier) < self.batch_size:
                _, _, node = heapq.heappop(heap)
                queued.discard(node.key)
                if node.solved or node.expanded or not node.expandable:
                    continue
                frontier.append(node)
            if not frontier:
                continue

            preds_batch = self.model.predict_batch(
                [n.smiles for n in frontier], self.expansion_width
            )
            stats.model_calls += 1

            for node, preds in zip(frontier, preds_batch):
                node.expanded = True
                stats.expansions += 1
                self._attach(graph, node, preds, stats, budget, push)

                if root.solved:
                    stats.elapsed_s = budget.elapsed
                    stats.terminated_by = "solved"
                    return SearchResult(target, True, graph, stats, self.name)

        stats.elapsed_s = budget.elapsed
        stats.terminated_by = "solved" if root.solved else reason
        return SearchResult(target, root.solved, graph, stats, self.name)

    def _attach(self, graph, node, preds, stats, budget, push) -> None:
        """Turn predictions into reaction nodes hanging off `node`."""
        child_depth = node.depth + 1
        # g for children: cost accumulated to reach this node. Approximated by
        # depth, since a DAG node can be reached by several paths.
        base_g = float(node.depth)

        for pred in preds:
            made = self._child_nodes(graph, pred.reactants, child_depth)
            if made is None:
                continue
            children, created = made
            rxn = graph.add_reaction(
                node, children, pred.score, pred.template_id,
                meta=dict(pred.meta),
            )
            if rxn is None:  # would have created a cycle
                continue
            stats.nodes_created += created
            stats.reactions_created += 1

            if rxn.solved:
                graph.mark_solved(node)
                return

            for child in children:
                if child.in_stock:
                    graph.mark_solved(child)
                else:
                    push(child, base_g + rxn.cost)


__all__ = ["BestFirstSearch"]
