"""Atom/bond featurization for the D-MPNN template classifier.

⚠️  This is a VERBATIM mirror of `ml-template-gnn/src/template_gnn/featurize.py`.
It is vendored so synomega can run inference from a checkpoint without depending
on the training repo. It MUST stay bit-identical to whatever produced the
checkpoint you load — any drift silently corrupts predictions rather than
raising, so `synomega.singlestep.template_gnn` asserts ATOM_FDIM against the
checkpoint's first-layer weight shape.

Atom features (45 dim), bond features (12 dim). See the training repo for the
full field-by-field breakdown.
"""

from __future__ import annotations

import math

from rdkit import Chem
from rdkit.Chem import AllChem, rdchem

ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53]
DEGREE_LIST = list(range(6))
CHARGE_LIST = [-2, -1, 0, 1, 2]
CHIRALITY_LIST = [
    rdchem.ChiralType.CHI_UNSPECIFIED,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    rdchem.ChiralType.CHI_OTHER,
]
NUM_H_LIST = list(range(5))
HYBRIDIZATION_LIST = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

ATOM_FDIM = (
    len(ATOM_LIST) + 1
    + len(DEGREE_LIST) + 1
    + len(CHARGE_LIST) + 1
    + len(CHIRALITY_LIST)
    + len(NUM_H_LIST) + 1
    + len(HYBRIDIZATION_LIST) + 1
    + 1  # is_aromatic
    + 1  # is_in_ring
    + 1  # mass
    + 1  # chi_pauling / 4
    + 1  # q_gasteiger (masked)
    + 1  # q_valid
)

PAULING_EN = {
    1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
    19: 0.82, 20: 1.00, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65,
    33: 2.18, 34: 2.55, 35: 2.96, 47: 1.93, 50: 1.96, 53: 2.66,
    78: 2.28, 79: 2.54, 80: 2.00,
}

BOND_TYPE_LIST = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]
STEREO_LIST = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOCIS,
    rdchem.BondStereo.STEREOTRANS,
]

BOND_FDIM = len(BOND_TYPE_LIST) + 1 + 1 + len(STEREO_LIST)


def _one_hot(value, choices, allow_other: bool = True) -> list[int]:
    feat = [0] * (len(choices) + (1 if allow_other else 0))
    try:
        idx = choices.index(value)
    except ValueError:
        if not allow_other:
            return feat
        idx = len(choices)
    feat[idx] = 1
    return feat


def compute_gasteiger(mol: Chem.Mol) -> bool:
    """Annotate Gasteiger charges in place; False when RDKit refuses."""
    try:
        AllChem.ComputeGasteigerCharges(mol)
        return True
    except Exception:
        return False


def atom_features(atom: Chem.Atom) -> list[float]:
    z = atom.GetAtomicNum()
    chi = PAULING_EN.get(z, 2.0) / 4.0

    q = 0.0
    q_valid = 0
    if atom.HasProp("_GasteigerCharge"):
        raw = atom.GetDoubleProp("_GasteigerCharge")
        if math.isfinite(raw):
            q = raw
            q_valid = 1

    return (
        _one_hot(z, ATOM_LIST)
        + _one_hot(atom.GetTotalDegree(), DEGREE_LIST)
        + _one_hot(atom.GetFormalCharge(), CHARGE_LIST)
        + _one_hot(atom.GetChiralTag(), CHIRALITY_LIST, allow_other=False)
        + _one_hot(atom.GetTotalNumHs(), NUM_H_LIST)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATION_LIST)
        + [1 if atom.GetIsAromatic() else 0]
        + [1 if atom.IsInRing() else 0]
        + [atom.GetMass() * 0.01]
        + [chi, q, float(q_valid)]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    return (
        _one_hot(bond.GetBondType(), BOND_TYPE_LIST, allow_other=False)
        + [1 if bond.GetIsConjugated() else 0]
        + [1 if bond.IsInRing() else 0]
        + _one_hot(bond.GetStereo(), STEREO_LIST, allow_other=False)
    )


def largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) == 1:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


__all__ = [
    "ATOM_FDIM",
    "BOND_FDIM",
    "atom_features",
    "bond_features",
    "compute_gasteiger",
    "largest_fragment",
]
