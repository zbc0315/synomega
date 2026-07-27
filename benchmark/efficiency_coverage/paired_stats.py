#!/usr/bin/env python3
"""Wilcoxon signed-rank test on paired node expansions, simplify vs full.

On the targets solved by BOTH models, is the per-target reduction in node
expansions under the simplification constraint statistically significant?
Pairs by canonical SMILES. Run: python paired_stats.py
"""
import csv
import os
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))


def canon(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else smi


def load(name):
    rows = list(csv.DictReader(open(os.path.join(HERE, name))))
    return {canon(r["smiles"]): r for r in rows}


def main():
    full = load("full.csv")
    simp = load("simplify.csv")
    solved_both = [k for k in full
                   if k in simp
                   and full[k]["solved"] in ("True", "1", "true")
                   and simp[k]["solved"] in ("True", "1", "true")]
    f_exp = [float(full[k]["expansions"]) for k in solved_both]
    s_exp = [float(simp[k]["expansions"]) for k in solved_both]
    n = len(solved_both)
    mean_f = sum(f_exp) / n
    mean_s = sum(s_exp) / n
    fewer = sum(s < f for s, f in zip(s_exp, f_exp))
    more = sum(s > f for s, f in zip(s_exp, f_exp))
    tie = n - fewer - more

    stat, p = wilcoxon(f_exp, s_exp)  # two-sided; H0: no shift
    stat_g, p_g = wilcoxon(f_exp, s_exp, alternative="greater")  # full > simplify

    print(f"Paired on solved-by-both targets: n = {n}")
    print(f"  mean expansions  full {mean_f:.1f}  simplify {mean_s:.1f}  "
          f"({100 * (mean_f - mean_s) / mean_f:.0f}% fewer)")
    print(f"  per-target: simplify fewer on {fewer}, more on {more}, tie {tie}")
    print(f"  Wilcoxon signed-rank (two-sided):  W={stat:.0f}  p={p:.2e}")
    print(f"  Wilcoxon (one-sided, full > simplify): W={stat_g:.0f}  p={p_g:.2e}")


if __name__ == "__main__":
    main()
