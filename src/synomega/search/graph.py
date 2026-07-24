"""AND-OR search graph.

Retrosynthesis search is an AND-OR problem:

  * a **molecule** node is solved if it is in stock, OR if *any* of its
    reactions is solved                                          (OR node)
  * a **reaction** node is solved if *all* of its reactant molecules are solved
                                                                 (AND node)

Molecules are interned by InChIKey, so the structure is a DAG, not a tree: an
intermediate reached through two different branches is one node, expanded once.
That deduplication is where most of the model calls are saved.

Cycles (a molecule appearing among its own ancestors) are rejected at edge
creation time — a route that makes X from X is not a route.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..chem.mol import Molecule


class MolNode:
    """OR node — one molecule to be made."""

    __slots__ = (
        "mol", "depth", "in_stock", "solved", "expanded", "expandable",
        "parents", "children", "value",
    )

    def __init__(self, mol: Molecule, depth: int, in_stock: bool):
        self.mol = mol
        self.depth = depth
        self.in_stock = in_stock
        self.solved = in_stock
        self.expanded = False
        #: False once we know it can never be expanded (depth cap, dead end).
        self.expandable = not in_stock
        self.parents: list["RxnNode"] = []
        self.children: list["RxnNode"] = []
        #: Heuristic cost-to-go, filled in by a value function.
        self.value: float = 0.0

    @property
    def key(self) -> str:
        return self.mol.key

    @property
    def smiles(self) -> str:
        return self.mol.smiles

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def __repr__(self) -> str:
        flag = "stock" if self.in_stock else ("solved" if self.solved else "open")
        return f"MolNode({self.smiles!r}, d={self.depth}, {flag})"


class RxnNode:
    """AND node — one disconnection; every reactant must be solved."""

    __slots__ = ("parent", "children", "score", "cost", "template_id", "solved", "meta")

    def __init__(
        self,
        parent: MolNode,
        children: list[MolNode],
        score: float,
        template_id: int | None = None,
        meta: dict | None = None,
    ):
        self.parent = parent
        self.children = children
        self.score = score
        # Negative log-likelihood, so costs add along a route.
        self.cost = -math.log(max(score, 1e-12))
        self.template_id = template_id
        self.solved = False
        self.meta = meta or {}

    def update_solved(self) -> bool:
        self.solved = all(c.solved for c in self.children)
        return self.solved

    def __repr__(self) -> str:
        rhs = ".".join(c.smiles for c in self.children)
        return f"RxnNode({self.parent.smiles} <- {rhs}, score={self.score:.4f})"


@dataclass
class AndOrGraph:
    """The search graph, with molecule nodes interned by InChIKey."""

    root: MolNode
    nodes: dict[str, MolNode] = field(default_factory=dict)

    @classmethod
    def create(cls, target: Molecule, in_stock: bool) -> "AndOrGraph":
        root = MolNode(target, depth=0, in_stock=in_stock)
        return cls(root=root, nodes={root.key: root})

    # ------------------------------------------------------------ structure

    def get_or_create(
        self, mol: Molecule, depth: int, in_stock: bool
    ) -> tuple[MolNode, bool]:
        """Return (node, created). Existing nodes keep their shallowest depth."""
        hit = self.nodes.get(mol.key)
        if hit is not None:
            if depth < hit.depth:
                hit.depth = depth
            return hit, False
        node = MolNode(mol, depth=depth, in_stock=in_stock)
        self.nodes[mol.key] = node
        return node, True

    def add_reaction(
        self,
        parent: MolNode,
        reactants: list[MolNode],
        score: float,
        template_id: int | None = None,
        meta: dict | None = None,
    ) -> RxnNode | None:
        """Attach a disconnection. Returns None if it would create a cycle."""
        ancestors = self._ancestor_keys(parent)
        ancestors.add(parent.key)
        for r in reactants:
            if r.key in ancestors:
                return None

        rxn = RxnNode(parent, reactants, score, template_id, meta)
        parent.children.append(rxn)
        for r in reactants:
            r.parents.append(rxn)
        rxn.update_solved()
        return rxn

    def _ancestor_keys(self, node: MolNode) -> set[str]:
        """All molecule keys upstream of `node` (toward the target)."""
        seen: set[str] = set()
        stack = list(node.parents)
        while stack:
            rxn = stack.pop()
            p = rxn.parent
            if p.key in seen:
                continue
            seen.add(p.key)
            stack.extend(p.parents)
        return seen

    # ----------------------------------------------------------- propagation

    def propagate_solved(self, node: MolNode) -> None:
        """Push a newly-solved molecule up toward the root.

        Iterative rather than recursive: deep graphs would blow the stack.
        """
        if not node.solved:
            return
        stack = [node]
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur.key in seen:
                continue
            seen.add(cur.key)
            for rxn in cur.parents:
                if rxn.solved:
                    continue
                if rxn.update_solved():
                    parent = rxn.parent
                    if not parent.solved:
                        parent.solved = True
                        stack.append(parent)

    def mark_solved(self, node: MolNode) -> None:
        node.solved = True
        self.propagate_solved(node)

    # ---------------------------------------------------------------- stats

    @property
    def num_molecules(self) -> int:
        return len(self.nodes)

    @property
    def num_reactions(self) -> int:
        return sum(len(n.children) for n in self.nodes.values())

    @property
    def solved(self) -> bool:
        return self.root.solved

    def open_nodes(self) -> list[MolNode]:
        """Molecules that could still be expanded."""
        return [
            n for n in self.nodes.values()
            if not n.solved and not n.expanded and n.expandable
        ]


__all__ = ["MolNode", "RxnNode", "AndOrGraph"]
