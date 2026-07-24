"""Monte-Carlo tree search (Segler et al. 2018 / AiZynthFinder style).

Four phases per iteration: select down the tree by UCT, expand one molecule,
roll out a cheap greedy playout to estimate the state's worth, back up the
reward.

MCTS explores differently from best-first: it will keep sampling a branch that
looks mediocre but uncertain. On targets where the single-step model's top-1 is
unreliable — which is the common case — that tolerance for a bad first guess is
what finds the route.

The tree here is over *states* (a set of molecules still to be made), while the
underlying AND-OR graph is shared, so molecule dedup and solution propagation
still apply.
"""

from __future__ import annotations

import math
import random

from .base import Budget, SearchAlgorithm, SearchResult, SearchStats
from .graph import AndOrGraph, MolNode


class _State:
    """A node of the MCTS tree: the frontier of molecules still unsolved."""

    __slots__ = ("mols", "parent", "children", "visits", "total_reward",
                 "untried", "terminal", "dead", "depth")

    def __init__(self, mols: list[MolNode], parent: "_State | None", depth: int):
        self.mols = mols
        self.parent = parent
        self.depth = depth
        self.children: list["_State"] = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried: list | None = None   # pending expansions, filled lazily
        self.terminal = not mols
        #: Exhausted: nothing left to try and no live children. Selection must
        #: skip these, otherwise it re-picks them forever and never expands.
        self.dead = False

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.0

    def uct(self, c: float, parent_visits: int) -> float:
        if self.visits == 0:
            return float("inf")
        explore = c * math.sqrt(math.log(max(parent_visits, 1)) / self.visits)
        return self.mean_reward + explore


class MCTS(SearchAlgorithm):
    """UCT search with greedy rollouts."""

    name = "mcts"

    def __init__(
        self,
        model,
        stock,
        *,
        expansion_width: int = 25,
        exploration: float = 1.4,
        rollout_depth: int = 3,
        seed: int | None = None,
    ):
        super().__init__(model, stock, expansion_width=expansion_width)
        self.exploration = exploration
        self.rollout_depth = rollout_depth
        self.rng = random.Random(seed)

    def run(self, target: str, budget: Budget) -> SearchResult:
        budget.start()
        stats = SearchStats()
        graph = self._make_graph(target)
        root_mol = graph.root

        if root_mol.solved:
            stats.elapsed_s = budget.elapsed
            stats.terminated_by = "solved"
            return SearchResult(target, True, graph, stats, self.name)

        root = _State([root_mol], None, depth=0)
        reason = "exhausted"

        while True:
            stop = budget.exhausted(stats.expansions)
            if stop:
                reason = stop
                break

            leaf = self._select(root)
            if leaf is None:
                break   # tree exhausted
            if leaf.terminal and not leaf.mols:
                # Solved branch already accounted for; nothing more to sample.
                leaf.dead = True
                self._backup(leaf, 1.0)
                if graph.root.solved:
                    break
                continue

            reward = self._expand_and_rollout(graph, leaf, stats, budget)
            self._backup(leaf, reward)

            if graph.root.solved:
                stats.elapsed_s = budget.elapsed
                stats.terminated_by = "solved"
                return SearchResult(target, True, graph, stats, self.name)

        stats.elapsed_s = budget.elapsed
        stats.terminated_by = "solved" if graph.root.solved else reason
        return SearchResult(target, graph.root.solved, graph, stats, self.name)

    # ------------------------------------------------------------- phases

    def _select(self, root: _State) -> "_State | None":
        """Descend by UCT to a node worth expanding.

        Returns None when the whole tree is exhausted, which ends the search
        rather than letting it spin.
        """
        node = root
        while True:
            if node.dead:
                if node.parent is None:
                    return None
                node = node.parent
                continue
            if node.terminal:
                return node
            if node.untried is None or node.untried:
                return node   # not yet fully expanded
            live = [c for c in node.children if not c.dead]
            if not live:
                # Nothing left to try here and every child is exhausted.
                node.dead = True
                if node.parent is None:
                    return None
                node = node.parent
                continue
            node = max(
                live,
                key=lambda ch: ch.uct(self.exploration, node.visits),
            )

    def _expand_and_rollout(
        self, graph: AndOrGraph, state: _State, stats: SearchStats, budget: Budget
    ) -> float:
        if state.terminal:
            return 1.0

        # Work on the shallowest unsolved molecule in this state.
        pending = [m for m in state.mols if not m.solved]
        if not pending:
            state.terminal = True
            return 1.0
        mol_node = min(pending, key=lambda m: m.depth)

        if mol_node.depth >= budget.max_depth:
            mol_node.expandable = False
            state.untried = []
            state.dead = True
            return 0.0

        if state.untried is None:
            if not mol_node.expanded:
                preds = self.model.predict(mol_node.smiles, self.expansion_width)
                stats.model_calls += 1
                stats.expansions += 1
                mol_node.expanded = True
                self._attach(graph, mol_node, preds, stats)
            state.untried = list(mol_node.children)

        if not state.untried:
            # Dead end: the molecule yielded no usable disconnection.
            if not state.children:
                state.dead = True
            return 0.0

        rxn = state.untried.pop(0)
        rest = [m for m in state.mols if m is not mol_node]
        new_mols = [c for c in list(rxn.children) + rest if not c.solved]
        child = _State(new_mols, state, state.depth + 1)
        state.children.append(child)

        if not new_mols:
            graph.mark_solved(mol_node)
            child.terminal = True
            return 1.0

        return self._rollout(graph, new_mols, stats, budget, state.depth + 1)

    def _rollout(
        self,
        graph: AndOrGraph,
        mols: list[MolNode],
        stats: SearchStats,
        budget: Budget,
        depth: int,
    ) -> float:
        """Greedy playout: keep taking the top prediction for a few steps.

        Reward is the fraction of frontier molecules that ended up purchasable —
        a partial-credit signal, which guides better than a 0/1 solved flag.
        """
        frontier = list(mols)
        for _ in range(self.rollout_depth):
            if budget.exhausted(stats.expansions):
                break
            pending = [m for m in frontier if not m.solved]
            if not pending:
                return 1.0
            node = min(pending, key=lambda m: m.depth)
            if node.depth >= budget.max_depth:
                break
            if not node.expanded:
                preds = self.model.predict(node.smiles, top_k=1)
                stats.model_calls += 1
                stats.expansions += 1
                node.expanded = True
                self._attach(graph, node, preds, stats)
            if not node.children:
                break
            rxn = node.children[0]
            frontier = [m for m in frontier if m is not node] + list(rxn.children)

        if not frontier:
            return 1.0
        solved = sum(1 for m in frontier if m.solved)
        return solved / len(frontier)

    def _backup(self, state: _State, reward: float) -> None:
        node: "_State | None" = state
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    # -------------------------------------------------------------- attach

    def _attach(self, graph, node, preds, stats) -> None:
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
                if child.in_stock and not child.solved:
                    graph.mark_solved(child)
            if rxn.solved:
                graph.mark_solved(node)


__all__ = ["MCTS"]
