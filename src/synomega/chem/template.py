"""Retrosynthesis template application.

Templates are RDChiral-style retro SMARTS written product >> reactants, so in
RDKit terms `GetReactantTemplate(0)` is the *product* pattern and RunReactants
on the product yields candidate reactant sets.

The per-match bookkeeping mirrors `ml-template-gnn/scripts/evaluate_reactants_v3.py`:
we keep every distinct reactant set a template produces (so top-K coverage is
not capped) and carry the matched product-atom indices so a reaction-center
model can break ties between matches of the same template.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import AllChem

from .mol import canonicalize


@dataclass(frozen=True)
class TemplateOutcome:
    """One reactant set produced by applying a template to a product."""

    reactants: tuple[str, ...]          # canonical SMILES, sorted
    match_atoms: tuple[int, ...] = ()   # product atom indices covered by the match

    @property
    def smiles(self) -> str:
        return ".".join(self.reactants)


@lru_cache(maxsize=50_000)
def _compiled(template_smarts: str):
    """Compile a retro template; None when RDKit rejects it."""
    try:
        rxn = AllChem.ReactionFromSmarts(template_smarts)
    except Exception:
        return None
    if rxn is None or rxn.GetNumReactantTemplates() != 1:
        return None
    return rxn


def apply_template(
    template_smarts: str,
    product_smiles: str,
    *,
    max_outcomes: int = 64,
) -> list[TemplateOutcome]:
    """Apply one retro template to a product, returning distinct reactant sets.

    Returns an empty list when the template does not compile, does not match, or
    every outcome fails sanitization.
    """
    rxn = _compiled(template_smarts)
    if rxn is None:
        return []

    product = Chem.MolFromSmiles(product_smiles)
    if product is None or product.GetNumAtoms() == 0:
        return []

    try:
        matches = product.GetSubstructMatches(rxn.GetReactantTemplate(0))
        outcome_sets = rxn.RunReactants((product,))
    except Exception:
        return []
    if not matches or not outcome_sets:
        return []

    # RunReactants emits one outcome per substructure match, in the same order.
    # When the counts disagree we cannot attribute matches to outcomes, so drop
    # the atom mapping rather than mislabel it.
    aligned = len(matches) == len(outcome_sets)

    seen: dict[tuple[str, ...], TemplateOutcome] = {}
    for i, outcome in enumerate(outcome_sets):
        if len(seen) >= max_outcomes:
            break
        parts = _sanitize_outcome(outcome)
        if parts is None:
            continue
        if parts in seen:
            continue
        seen[parts] = TemplateOutcome(
            reactants=parts,
            match_atoms=tuple(matches[i]) if aligned else (),
        )
    return list(seen.values())


def _sanitize_outcome(outcome_mols) -> tuple[str, ...] | None:
    """Sanitize an RDKit outcome tuple into sorted canonical SMILES."""
    parts: list[str] = []
    for mol in outcome_mols:
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        try:
            smi = Chem.MolToSmiles(mol)
        except Exception:
            return None
        canon = canonicalize(smi)
        if canon is None:
            return None
        parts.append(canon)
    if not parts:
        return None
    return tuple(sorted(parts))


@dataclass
class TemplateLibrary:
    """A label -> retro SMARTS mapping, as emitted by the training pipeline."""

    templates: dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path) -> "TemplateLibrary":
        """Load `label_to_template_smarts.json` ({"0": "...", ...})."""
        import json
        from pathlib import Path

        raw = json.loads(Path(path).read_text())
        return cls({int(k): str(v) for k, v in raw.items()})

    @classmethod
    def from_tsv(cls, path, label_col: str = "label",
                 smarts_col: str = "template_smarts") -> "TemplateLibrary":
        """Load a templates TSV with `label` and `template_smarts` columns."""
        import csv
        from pathlib import Path

        out: dict[int, str] = {}
        with Path(path).open() as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                out[int(row[label_col])] = row[smarts_col]
        return cls(out)

    def __len__(self) -> int:
        return len(self.templates)

    def __getitem__(self, label: int) -> str:
        return self.templates[label]

    def get(self, label: int) -> str | None:
        return self.templates.get(label)


__all__ = ["TemplateOutcome", "TemplateLibrary", "apply_template"]
