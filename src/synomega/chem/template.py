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
from itertools import permutations

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


# ------------------------------------------------------------------ forward

@dataclass(frozen=True)
class ForwardOutcome:
    """One product produced by applying a template FORWARD to reactants."""

    product: str                         # canonical SMILES, largest organic fragment
    match_atoms: tuple[int, ...] = ()    # reactant atoms covered (currently unused)

    @property
    def smiles(self) -> str:
        return self.product


def _has_radical(mol) -> bool:
    """True if any atom carries unpaired electrons.

    RDKit accepts *under*-valent atoms by assigning radical electrons (an r=0
    forward template applied to a fragment can leave a carbene/acyl-radical
    carbon such as ``[C]=O``). Such products are chemically absurd, so we drop
    them — RDKit's own sanitization only rejects *over*-valence, not this.
    """
    return any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())


def _canon_largest_product(smi: str) -> str | None:
    """Largest organic fragment, atom maps stripped, canonical (GT-product form)."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    if frags:
        m = max(frags, key=lambda f: f.GetNumHeavyAtoms())
    for atom in m.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        return Chem.MolToSmiles(m)
    except Exception:
        return None


@lru_cache(maxsize=50_000)
def _compiled_forward(retro_smarts: str):
    """Compile the FORWARD reaction from a retro template.

    The library stores retro SMARTS ``product >> reactants``; reversing it around
    ``>>`` gives the forward reaction ``reactants >> product``. Unlike
    :func:`_compiled`, the reactant side may hold two or more templates (an
    intermolecular reaction), so we do NOT require a single reactant template.
    None when RDKit rejects the reversed SMARTS.

    The reversal (swap around ``>>``, strip one outer paren layer) mirrors
    ``evaluate_products.py``; keep it byte-for-byte.
    """
    if ">>" not in retro_smarts:
        return None
    lhs, rhs = retro_smarts.split(">>", 1)          # lhs = product, rhs = reactants
    fwd = rhs.strip("()") + ">>" + lhs.strip("()")
    try:
        rxn = AllChem.ReactionFromSmarts(fwd)
    except Exception:
        return None
    if rxn is None or rxn.GetNumReactantTemplates() == 0:
        return None
    return rxn


def apply_template_forward(
    retro_smarts: str,
    reactants_smiles: str,
    *,
    max_outcomes: int = 64,
) -> list[ForwardOutcome]:
    """Apply one template FORWARD to a reactant set, returning distinct products.

    Reverses the retro template, assigns the reactant fragments to the template's
    reactant slots (every ordered assignment, so intermolecular templates fire on
    all pairings), runs RDKit forward, and keeps every distinct product that
    sanitizes and is radical-free. Returns an empty list when the template does
    not compile, arity cannot be satisfied, or nothing survives.
    """
    rxn = _compiled_forward(retro_smarts)
    if rxn is None:
        return []
    mol = Chem.MolFromSmiles(reactants_smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return []
    frags = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    if not frags or len(frags) > 8:              # >8 fragments: pathological, skip
        return []
    nslots = rxn.GetNumReactantTemplates()
    if len(frags) < nslots:
        return []

    seen: dict[str, ForwardOutcome] = {}
    for asg in permutations(range(len(frags)), nslots):
        try:
            outcomes = rxn.RunReactants(tuple(frags[i] for i in asg), maxProducts=1000)
        except Exception:
            continue
        for outcome in outcomes:
            for pmol in outcome:
                if pmol is None:
                    continue
                try:
                    Chem.SanitizeMol(pmol)
                    if _has_radical(pmol):
                        continue
                    canon = _canon_largest_product(Chem.MolToSmiles(pmol))
                except Exception:
                    canon = None
                if canon and canon not in seen:
                    seen[canon] = ForwardOutcome(product=canon)
                    if len(seen) >= max_outcomes:
                        return list(seen.values())
    return list(seen.values())


# --------------------------------------------------------------- library load

def load_template_library(
    run_dir,
    *,
    checkpoint_name: str = "best.pt",
    templates_path=None,
):
    """Locate a checkpoint and its label -> retro SMARTS map in a run directory.

    Shared discovery for both the retro and forward template-GNN backends.
    Returns ``(checkpoint_path, TemplateLibrary)``. The template map is searched
    at ``templates_path``, then ``<run_dir>/label_to_template_smarts.json``, then
    the processed-data dir recorded in the checkpoint's ``config.yaml``.
    """
    from pathlib import Path

    run_dir = Path(run_dir)
    ckpt = run_dir / checkpoint_name
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt}")

    candidates: list[Path] = []
    if templates_path is not None:
        candidates.append(Path(templates_path))
    candidates.append(run_dir / "label_to_template_smarts.json")

    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text())
            processed = Path(cfg["data"]["processed_dir"])
            candidates.append(processed / "label_to_template_smarts.json")
        except Exception:
            pass

    for cand in candidates:
        if cand.exists():
            lib = (
                TemplateLibrary.from_tsv(cand)
                if cand.suffix == ".tsv"
                else TemplateLibrary.from_json(cand)
            )
            return ckpt, lib

    raise FileNotFoundError(
        "could not locate a template map (label_to_template_smarts.json). "
        f"Tried: {', '.join(str(c) for c in candidates)}. "
        "Pass templates_path= explicitly."
    )


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


__all__ = [
    "TemplateOutcome",
    "TemplateLibrary",
    "apply_template",
    "ForwardOutcome",
    "apply_template_forward",
    "load_template_library",
]
