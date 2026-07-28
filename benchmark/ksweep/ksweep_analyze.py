#!/usr/bin/env python3
"""k-sweep analysis: the original model @ expansion width 50 is the per-molecule
gold standard; the simplify model is evaluated at k = 3..10. For each k we report
the efficiency (mean expansions, median wall-clock) and the scoring accuracy relative
to the gold (bb-coverage MAE/RMSE, Spearman/Pearson, and solved agreement).

Reads gold_original_k50.csv and simplify_k03.csv..simplify_k10.csv in this directory
(pulled from the server run). Per-molecule scores/times are kept in those CSVs for
deeper analysis. Run: python ksweep_analyze.py
"""
import csv
import os
import statistics as st
from scipy.stats import spearmanr, pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    rows = list(csv.DictReader(open(os.path.join(HERE, name))))
    return {r["smiles"]: r for r in rows}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def solved(r):
    return r["solved"] in ("1", "True", "true")


def sweep_table(title, prefix, gold, gkeys):
    print(f"\n=== {title} (accuracy vs gold = original @ k=50) ===")
    print(f"{'k':>3} {'mean_exp':>9} {'med_time':>9} {'solved%':>8} "
          f"{'MAE':>7} {'RMSE':>7} {'Spearman':>9} {'Pearson':>8} {'solve_agree':>11}")
    for k in range(3, 11):
        name = f"{prefix}_k{k:02d}.csv"
        if not os.path.exists(os.path.join(HERE, name)):
            print(f"{k:>3}  (pending: {name})")
            continue
        s = load(name)
        keys = [x for x in gkeys if x in s and num(s[x]["bb_coverage"]) is not None]
        gv = [num(gold[x]["synscore"]) for x in keys]
        sv = [num(s[x]["synscore"]) for x in keys]
        exp = [num(s[x]["expansions"]) for x in keys if num(s[x]["expansions"]) is not None]
        sec = [num(s[x]["sec"]) for x in keys if num(s[x]["sec"]) is not None]
        err = [a - b for a, b in zip(sv, gv)]
        mae = sum(abs(e) for e in err) / len(err)
        rmse = (sum(e * e for e in err) / len(err)) ** 0.5
        rho = spearmanr(sv, gv).statistic
        r = pearsonr(sv, gv).statistic
        solve_agree = sum(solved(s[x]) == solved(gold[x]) for x in keys) / len(keys)
        solved_pct = 100 * sum(solved(s[x]) for x in keys) / len(keys)
        print(f"{k:>3} {st.mean(exp):>9.1f} {st.median(sec):>9.3f} {solved_pct:>8.1f} "
              f"{mae:>7.4f} {rmse:>7.4f} {rho:>9.3f} {r:>8.3f} {solve_agree:>11.3f}")


def main():
    gold = load("gold_original_k50.csv")
    gkeys = [k for k in gold if num(gold[k]["bb_coverage"]) is not None]
    ge = [num(gold[x]["expansions"]) for x in gkeys if num(gold[x]["expansions"]) is not None]
    gt = [num(gold[x]["sec"]) for x in gkeys if num(gold[x]["sec"]) is not None]
    gsolved = 100 * sum(solved(gold[x]) for x in gkeys) / len(gkeys)
    print(f"gold = original model @ k=50, n={len(gkeys)} scored | "
          f"mean_exp {st.mean(ge):.1f}  med_time {st.median(gt):.3f}s  solved {gsolved:.1f}%")

    sweep_table("SIMPLIFY sweep", "simplify", gold, gkeys)
    sweep_table("ORIGINAL sweep", "original", gold, gkeys)


if __name__ == "__main__":
    main()
