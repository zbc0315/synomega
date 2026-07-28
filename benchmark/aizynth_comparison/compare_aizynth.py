#!/usr/bin/env python3
"""SynOmega vs AiZynthFinder on all 1000 ChEMBL targets (budget-aligned).

Reproduces every AiZynthFinder-comparison number in the paper (solved rate,
per-target agreement, and the wall-clock efficiency contrast) from:
  - ./aizynth_out.json.gz                      (AiZynthFinder, depth/width/budget aligned)
  - ../efficiency_coverage/original.csv        (SynOmega, unconstrained model)
  - ../efficiency_coverage/simplify.csv        (SynOmega, simplifying-template constrained)

Targets are matched by canonical SMILES. AiZynthFinder's `number_of_nodes` (MCTS
tree size) and SynOmega's `expansions` (single-step expansions) count different
quantities and are NOT compared; we contrast wall-clock time and solved rate only.

Run:  python compare_aizynth.py
"""
import gzip
import json
import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))


def canon(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else smi


def load_synomega(name):
    path = os.path.join(HERE, "..", "efficiency_coverage", name)
    rows = list(csv.DictReader(open(path)))
    solved = {canon(r["smiles"]): (r["solved"] in ("True", "1", "true")) for r in rows}
    sec = {canon(r["smiles"]): float(r["sec"]) for r in rows if r["sec"] not in ("", "nan")}
    return solved, sec


def main():
    az = json.load(gzip.open(os.path.join(HERE, "aizynth_out.json.gz")))["data"]
    az_solved = {canon(r["target"]): bool(r["is_solved"]) for r in az}
    az_time = [r["search_time"] for r in az]

    orig_solved, orig_sec = load_synomega("original.csv")
    simp_solved, simp_sec = load_synomega("simplify.csv")

    # SynOmega and AiZynth both cover all 1000 targets; use the shared set
    common = [k for k in az_solved if k in simp_solved and k in orig_solved]
    n = len(common)
    orig_sec_all = [orig_sec[k] for k in common if k in orig_sec]
    simp_sec_all = [simp_sec[k] for k in common if k in simp_sec]
    print(f"Shared target set (AiZynth run): n = {n}\n")

    print(f"Solved rate (on the shared {n})")
    print(f"  AiZynthFinder        {sum(az_solved[k] for k in common)}/{n} = "
          f"{100 * sum(az_solved[k] for k in common) / n:.1f}%")
    print(f"  SynOmega (original)  {sum(orig_solved[k] for k in common)}/{n} = "
          f"{100 * sum(orig_solved[k] for k in common) / n:.1f}%")
    print(f"  SynOmega (simplify)  {sum(simp_solved[k] for k in common)}/{n} = "
          f"{100 * sum(simp_solved[k] for k in common) / n:.1f}%\n")

    print("Median / mean search time per target (s)")
    print(f"  AiZynthFinder        median {st.median(az_time):.2f}   mean {st.mean(az_time):.2f}")
    print(f"  SynOmega (original)  median {st.median(orig_sec_all):.2f}   mean {st.mean(orig_sec_all):.2f}")
    print(f"  SynOmega (simplify)  median {st.median(simp_sec_all):.2f}   mean {st.mean(simp_sec_all):.2f}")
    print(f"  -> AiZynth median is {st.median(az_time) / st.median(simp_sec_all):.0f}x the simplify "
          f"model and {st.median(az_time) / st.median(orig_sec_all):.0f}x the original model\n")

    # per-target agreement: SynOmega (simplify) vs AiZynthFinder
    common = [k for k in simp_solved if k in az_solved]
    both = sum(simp_solved[k] and az_solved[k] for k in common)
    syn = sum(simp_solved[k] and not az_solved[k] for k in common)
    azo = sum(not simp_solved[k] and az_solved[k] for k in common)
    nei = sum(not simp_solved[k] and not az_solved[k] for k in common)
    print("Per-target agreement, SynOmega (simplify) vs AiZynthFinder")
    print(f"  both={both}  SynOmega-only={syn}  AiZynthFinder-only={azo}  neither={nei}")


if __name__ == "__main__":
    main()
