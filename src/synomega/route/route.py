"""Synthesis routes: extraction from a solved graph, scoring, serialization."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..chem.reaction import Conditions
from ..search.graph import AndOrGraph, MolNode, RxnNode


@dataclass
class RouteNode:
    """One molecule in a route tree.

    A leaf has no reaction: it is either purchasable (`in_stock`) or a dead end
    that the search never resolved.
    """

    smiles: str
    in_stock: bool
    depth: int
    reaction: "RouteStep | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.reaction is None


@dataclass
class RouteStep:
    """One reaction in a route."""

    product: str
    reactants: list[RouteNode]
    score: float
    template_id: int | None = None
    #: Reserved for a future condition model; always None today.
    conditions: Conditions | None = None
    meta: dict = field(default_factory=dict)

    @property
    def reaction_smiles(self) -> str:
        return f"{'.'.join(r.smiles for r in self.reactants)}>>{self.product}"


@dataclass
class Route:
    """A synthesis tree rooted at the target."""

    root: RouteNode
    solved: bool = False

    # -------------------------------------------------------------- shape

    @property
    def leaves(self) -> list[RouteNode]:
        out: list[RouteNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                out.append(node)
            else:
                stack.extend(node.reaction.reactants)
        return out

    @property
    def steps(self) -> list[RouteStep]:
        """All reactions, deepest-last (i.e. target's own step first)."""
        out: list[RouteStep] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.reaction is not None:
                out.append(node.reaction)
                stack.extend(node.reaction.reactants)
        return out

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def depth(self) -> int:
        return max((n.depth for n in self.leaves), default=0)

    # ------------------------------------------------------------- scoring

    @property
    def bb_coverage(self) -> float:
        """Fraction of leaves that are purchasable. 1.0 means fully solved."""
        leaves = self.leaves
        if not leaves:
            return 0.0
        return sum(1 for leaf in leaves if leaf.in_stock) / len(leaves)

    @property
    def cumulative_score(self) -> float:
        """Product of per-step scores — a rough route likelihood."""
        score = 1.0
        for step in self.steps:
            score *= max(step.score, 1e-12)
        return score

    def rank_key(self) -> tuple:
        """Sort key: fully-solved first, then fewer steps, then higher score."""
        return (-self.bb_coverage, self.num_steps, -self.cumulative_score)

    # ------------------------------------------------------------- output

    def to_dict(self) -> dict:
        def _node(n: RouteNode) -> dict:
            d: dict = {
                "smiles": n.smiles,
                "in_stock": n.in_stock,
                "depth": n.depth,
                "type": "mol",
            }
            if n.reaction is not None:
                d["children"] = [
                    {
                        "type": "reaction",
                        "smiles": n.reaction.reaction_smiles,
                        "score": n.reaction.score,
                        "template_id": n.reaction.template_id,
                        "children": [_node(r) for r in n.reaction.reactants],
                    }
                ]
            return d

        return {
            "solved": self.solved,
            "num_steps": self.num_steps,
            "depth": self.depth,
            "bb_coverage": self.bb_coverage,
            "cumulative_score": self.cumulative_score,
            "tree": _node(self.root),
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent)

    def describe(self) -> str:
        """Readable step-by-step summary."""
        lines = [
            f"target: {self.root.smiles}",
            f"solved: {self.solved}  steps: {self.num_steps}  "
            f"depth: {self.depth}  bb_coverage: {self.bb_coverage:.2f}",
        ]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"  [{i}] {step.reaction_smiles}  (score={step.score:.4f})")
        missing = [leaf.smiles for leaf in self.leaves if not leaf.in_stock]
        if missing:
            lines.append(f"  not in stock: {', '.join(missing)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Route(steps={self.num_steps}, solved={self.solved}, "
            f"bb_coverage={self.bb_coverage:.2f})"
        )


# --------------------------------------------------------------- extraction


def extract_routes(graph: AndOrGraph, max_routes: int = 25) -> list[Route]:
    """Enumerate solved routes from a searched graph, best first.

    Only reactions on solved sub-trees are followed, so every returned route has
    `bb_coverage == 1.0`. If the root is unsolved, returns the single best
    partial route instead — that still carries a useful coverage number.
    """
    root = graph.root
    if not root.solved:
        partial = _best_partial(root)
        return [partial] if partial is not None else []

    routes: list[Route] = []
    for tree in _enumerate(root, max_routes=max_routes, seen=set()):
        routes.append(Route(root=tree, solved=True))
        if len(routes) >= max_routes:
            break
    routes.sort(key=lambda r: r.rank_key())
    return routes


def _enumerate(node: MolNode, max_routes: int, seen: set[str]):
    """Yield route trees rooted at `node`, cheapest reaction first."""
    if node.in_stock or not node.children:
        yield RouteNode(node.smiles, node.in_stock, node.depth)
        return

    solved_rxns = [r for r in node.children if r.solved]
    if not solved_rxns:
        yield RouteNode(node.smiles, node.in_stock, node.depth)
        return

    solved_rxns.sort(key=lambda r: r.cost)
    produced = 0
    for rxn in solved_rxns:
        if produced >= max_routes:
            return
        # Take the best sub-route per child. Enumerating the full cross product
        # explodes combinatorially and adds little — the alternatives differ
        # only deep in the tree.
        children = []
        ok = True
        for child in rxn.children:
            sub = next(_enumerate(child, 1, seen), None)
            if sub is None:
                ok = False
                break
            children.append(sub)
        if not ok:
            continue
        step = RouteStep(
            product=node.smiles,
            reactants=children,
            score=rxn.score,
            template_id=rxn.template_id,
            meta=dict(rxn.meta),
        )
        yield RouteNode(node.smiles, node.in_stock, node.depth, reaction=step)
        produced += 1


def _best_partial(root: MolNode) -> Route | None:
    """Greedy best-scoring tree through an unsolved graph.

    Used so an unsolved target still reports how far the search got, via
    `bb_coverage`.
    """
    visiting: set[str] = set()

    def build(node: MolNode) -> RouteNode:
        if node.in_stock or not node.children or node.key in visiting:
            return RouteNode(node.smiles, node.in_stock, node.depth)
        visiting.add(node.key)
        # Prefer a solved disconnection; otherwise the highest-scoring one.
        rxn = max(node.children, key=lambda r: (r.solved, r.score))
        step = RouteStep(
            product=node.smiles,
            reactants=[build(c) for c in rxn.children],
            score=rxn.score,
            template_id=rxn.template_id,
            meta=dict(rxn.meta),
        )
        visiting.discard(node.key)
        return RouteNode(node.smiles, node.in_stock, node.depth, reaction=step)

    tree = build(root)
    return Route(root=tree, solved=False)


__all__ = ["Route", "RouteNode", "RouteStep", "extract_routes"]
