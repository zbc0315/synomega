"""Molecule identity, canonicalization, and template application."""

from __future__ import annotations

from synomega.chem import Molecule, apply_template, canonicalize, split_components
from synomega.chem.reaction import extract_product, parse_reaction_smiles


def test_canonicalization_is_representation_independent():
    # Same molecule, three ways of writing it.
    a = Molecule.of("OCC")
    b = Molecule.of("CCO")
    c = Molecule.of("C(C)O")
    assert a.smiles == b.smiles == c.smiles
    assert a is b is c  # interned
    assert a.key == b.key


def test_atom_maps_are_stripped():
    mapped = Molecule.of("[CH3:1][CH2:2][OH:3]")
    plain = Molecule.of("CCO")
    assert mapped.smiles == plain.smiles


def test_bad_smiles_returns_none():
    assert canonicalize("not a smiles!!") is None
    assert Molecule.try_of("not a smiles!!") is None


def test_inchikey_is_stable_and_shaped():
    key = Molecule.of("CC(=O)Nc1ccccc1").key
    assert len(key) == 27 and key.count("-") == 2


def test_split_components():
    parts = split_components("CCO.CC(=O)O")
    assert set(parts) == {"CCO", "CC(=O)O"}


def test_parse_reaction_smiles():
    left, agents, right = parse_reaction_smiles("CCO.CC(=O)O>>CCOC(C)=O")
    assert set(left) == {"CCO", "CC(=O)O"}
    assert agents == []
    assert right == ["CCOC(C)=O"]
    assert parse_reaction_smiles("no arrow here") is None


def test_extract_product_takes_largest():
    assert extract_product("A.B>>CCO.CC(=O)Nc1ccccc1") == "CC(=O)Nc1ccccc1"


def test_apply_template_amide_disconnection():
    # Retro template: amide C-N bond -> acid + amine.
    smarts = "[C;H0;D3;+0:1](=[O;D1])-[NH;D2;+0:2]>>[C;H0;D3;+0:1](=[O;D1])-[OH].[NH2;D1;+0:2]"
    outcomes = apply_template(smarts, "CC(=O)Nc1ccccc1")
    assert outcomes, "amide template should match acetanilide"
    produced = {o.reactants for o in outcomes}
    assert any("Nc1ccccc1" in r for r in produced)


def test_apply_template_no_match_is_empty():
    smarts = "[C;H0;D3;+0:1](=[O;D1])-[NH;D2;+0:2]>>[C;H0;D3;+0:1](=[O;D1])-[OH].[NH2;D1;+0:2]"
    assert apply_template(smarts, "CCO") == []


def test_apply_template_rejects_garbage_smarts():
    assert apply_template("this is not smarts", "CCO") == []
