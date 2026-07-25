"""Single-molecule graph featurization for the dual-tower plausibility model.

These atom/bond features are byte-for-byte identical to the ones the model was
trained with (the CGR classifier's per-side features), so a checkpoint trained in
the ``reaction-plausibility`` project loads and scores correctly here. A reactant
set ``"A.B"`` is parsed as one disconnected graph and the product as another — no
atom-map numbers are read, so nothing needs a reactant<->product correspondence.
"""

from __future__ import annotations

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53]
DEGREE_LIST = list(range(6))
CHARGE_LIST = [-2, -1, 0, 1, 2]
NUM_H_LIST = list(range(5))
HYB_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

_ATOM_FDIM = (
    len(ATOM_LIST) + 1
    + len(DEGREE_LIST) + 1
    + len(CHARGE_LIST) + 1
    + len(NUM_H_LIST) + 1
    + len(HYB_LIST) + 1
    + 1  # aromatic
    + 1  # in ring
)
_BOND_FDIM = (len(BOND_TYPES) + 1) + 1 + 1 + 1  # type one-hot(+none) + conj + ring + present

ATOM_FDIM = _ATOM_FDIM
BOND_FDIM = _BOND_FDIM


def _one_hot(value, choices, extra=True):
    feat = [0.0] * (len(choices) + (1 if extra else 0))
    try:
        feat[choices.index(value)] = 1.0
    except ValueError:
        if extra:
            feat[-1] = 1.0
    return feat


def atom_features(atom) -> list[float]:
    return (
        _one_hot(atom.GetAtomicNum(), ATOM_LIST)
        + _one_hot(atom.GetTotalDegree(), DEGREE_LIST)
        + _one_hot(atom.GetFormalCharge(), CHARGE_LIST)
        + _one_hot(atom.GetTotalNumHs(), NUM_H_LIST)
        + _one_hot(atom.GetHybridization(), HYB_LIST)
        + [1.0 if atom.GetIsAromatic() else 0.0]
        + [1.0 if atom.IsInRing() else 0.0]
    )


def bond_features(bond) -> list[float]:
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)
        + [1.0 if bond.GetIsConjugated() else 0.0]
        + [1.0 if bond.IsInRing() else 0.0]
        + [1.0]
    )


def mol_to_graph(smiles: str):
    """SMILES (maps ignored) -> a PyG ``Data`` with paired directed edges, or None."""
    import torch
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_feats = [atom_features(a) for a in mol.GetAtoms()]
    if not atom_feats:
        return None
    bond_index, bond_feats = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        bond_index.append((i, j)); bond_feats.append(f)   # keep the two
        bond_index.append((j, i)); bond_feats.append(f)    # directions adjacent
    x = torch.tensor(atom_feats, dtype=torch.float)
    if bond_index:
        edge_index = torch.tensor(bond_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(bond_feats, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FDIM), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


__all__ = ["mol_to_graph", "atom_features", "bond_features", "ATOM_FDIM", "BOND_FDIM"]
