"""Render the few genuine chemical structures the docs need, as SVG.

mermaid cannot draw molecules; these reaction schemes are produced by RDKit and
committed next to this script. Re-run after editing the SMILES below:

    python synomega/docs/figures/make_structures.py
"""
from __future__ import annotations

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

OUT = Path(__file__).parent


def _draw_rxn(smarts: str, name: str, w: int = 820, h: int = 240) -> None:
    rxn = AllChem.ReactionFromSmarts(smarts, useSmiles=True)
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    d.drawOptions().bondLineWidth = 2
    d.DrawReaction(rxn)
    d.FinishDrawing()
    svg = d.GetDrawingText().replace("svg:", "")
    (OUT / f"{name}.svg").write_text(svg)


# Simplification-constrained template, drawn retrosynthetically (product => two
# smaller precursors): an amide disconnection into a carboxylic acid + an amine.
# This is the "simplifying" class kept in the constrained action space.
_draw_rxn(
    "CC(=O)Nc1ccccc1>>CC(=O)O.Nc1ccccc1",
    "tpl_simplifying",
)

# Forward reaction prediction, drawn in the synthesis direction
# (reactants -> product): the same amide coupling, run forward.
_draw_rxn(
    "CC(=O)O.Nc1ccccc1>>CC(=O)Nc1ccccc1",
    "forward_demo",
)

if __name__ == "__main__":
    for p in sorted(OUT.glob("tpl_*.svg")) + sorted(OUT.glob("forward_demo.svg")):
        print("wrote:", p.name)
