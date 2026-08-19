"""Forward reaction prediction: template application, radical filter, model.

The RDKit-only tests run everywhere. The GNN regression test needs the forward
checkpoint and the r20 mapped corpus (server-only), so it is skipped unless both
are provided via the env vars below.
"""

from __future__ import annotations

import os

import pytest
from rdkit import Chem

from synomega.chem.template import (
    ForwardOutcome,
    _has_radical,
    apply_template_forward,
)
from synomega.forward import ForwardModel, ForwardPrediction


# Amide-bond disconnection, written retro (product >> reactants). Applied forward
# it couples a carboxylic acid and an amine into an amide.
AMIDE_RETRO = "[C:1](=[O:2])[NH1:3]>>[C:1](=[O:2])[OH].[NH2:3]"


def _canon(smi: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


# ------------------------------------------------------------ apply forward

def test_apply_forward_amide_coupling():
    outcomes = apply_template_forward(AMIDE_RETRO, "CC(=O)O.CN")
    products = {o.product for o in outcomes}
    assert _canon("CC(=O)NC") in products
    assert all(isinstance(o, ForwardOutcome) for o in outcomes)


def test_apply_forward_products_are_radical_free():
    for o in apply_template_forward(AMIDE_RETRO, "CC(=O)O.CN"):
        mol = Chem.MolFromSmiles(o.product)
        assert mol is not None
        assert not _has_radical(mol)


def test_apply_forward_arity_too_few_fragments():
    # two reactant slots, only one reactant fragment -> nothing fires
    assert apply_template_forward(AMIDE_RETRO, "CC(=O)O") == []


def test_apply_forward_bad_template_returns_empty():
    assert apply_template_forward("not a template", "CC(=O)O.CN") == []
    assert apply_template_forward("no-arrow-here", "CN") == []


def test_apply_forward_pathological_fragment_count_skipped():
    many = ".".join(["C"] * 9)  # 9 fragments > 8 guard
    assert apply_template_forward(AMIDE_RETRO, many) == []


# --------------------------------------------------------------- radical util

def test_has_radical():
    assert _has_radical(Chem.MolFromSmiles("[CH3]"))       # methyl radical
    assert _has_radical(Chem.MolFromSmiles("CC(=O)[C]=O"))  # acyl-radical artifact
    assert not _has_radical(Chem.MolFromSmiles("CCO"))
    assert not _has_radical(Chem.MolFromSmiles("CC(=O)NC"))


# ----------------------------------------------------------- ForwardPrediction

def test_forward_prediction_dataclass():
    p = ForwardPrediction(product="CCO", score=0.8, template_id=3)
    assert p.smiles == "CCO"
    assert "0.8000" in repr(p)
    with pytest.raises(Exception):
        p.product = "CC"  # frozen


def test_forward_model_is_not_single_step_model():
    # a forward model must NOT satisfy the retro SingleStepModel contract, so the
    # planner cannot silently accept it.
    from synomega.singlestep import SingleStepModel

    assert not issubclass(ForwardModel, SingleStepModel)


# ------------------------------------------------------ GNN numerical regression

@pytest.mark.gnn
@pytest.mark.slow
def test_forward_model_numerical_regression():
    """Forward model reproduces product top-1 on a test-split sample.

    Enable by pointing at the assets (server):
      SYNOMEGA_FORWARD_MODEL=/.../runs/r20_forward
      SYNOMEGA_FORWARD_EVAL_CSV=/.../data_tmp/r20_mapped.csv
    """
    run_dir = os.environ.get("SYNOMEGA_FORWARD_MODEL", "").strip()
    csv_path = os.environ.get("SYNOMEGA_FORWARD_EVAL_CSV", "").strip()
    if not run_dir or not csv_path or not os.path.exists(csv_path):
        pytest.skip("forward model / eval csv not provided")

    pytest.importorskip("torch")
    from synomega.forward import ForwardTemplateGNN

    model = ForwardTemplateGNN.from_pretrained(run_dir, topk_templates=10)

    reactants, gold = [], []
    with open(csv_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3:
                continue
            try:
                rid = int(parts[0])
            except ValueError:
                continue
            if rid % 20 != 0:                     # test split
                continue
            seg = ",".join(parts[1:-1]).split(">")
            if len(seg) < 2 or not seg[0] or not seg[-1]:
                continue
            gp = Chem.MolFromSmiles(seg[-1])
            if gp is None:
                continue
            frs = Chem.GetMolFrags(gp, asMols=True, sanitizeFrags=False)
            gp = max(frs, key=lambda f: f.GetNumHeavyAtoms()) if frs else gp
            for a in gp.GetAtoms():
                a.SetAtomMapNum(0)
            reactants.append(seg[0])
            gold.append(Chem.MolToSmiles(gp))
            if len(reactants) >= 1000:
                break

    preds = model.predict_batch(reactants, top_k=1)
    hit = sum(1 for pr, g in zip(preds, gold) if pr and pr[0].product == g)
    product_top1 = hit / len(reactants)
    assert product_top1 >= 0.55, f"product top-1 {product_top1:.3f} < 0.55"
