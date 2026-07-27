#!/usr/bin/env python3
"""Bootstrap 95% CI for the Spearman rho of the SynOmega score vs SAscore /
SCScore / RAscore over the 1000-target set.

Deterministic (fixed seed) so the reported CI is reproducible. Uses the same
merged data as analyze_scores.py. Run: python bootstrap_ci.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 10000
SEED = 20260727


def rho_ci(x, y, rng):
    n = len(x)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boots[i] = spearmanr(x[idx], y[idx]).statistic
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return lo, hi


def main():
    syn = pd.read_csv(os.path.join(HERE, "syn_targets.csv"))
    base = pd.read_csv(os.path.join(HERE, "base_targets.csv"))
    ra = pd.read_csv(os.path.join(HERE, "rascore_targets.csv"))
    df = syn.merge(base, on="smiles").merge(ra, on="smiles", how="left")
    for c in ["synscore", "sascore", "scscore", "rascore"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["synscore", "sascore", "scscore", "rascore"])
    # SynOmega score = 1/(U+1)**U (U = non-purchasable starting materials); read from
    # the `synscore` column. Unlike bb_coverage this depends on the absolute count U,
    # so the correlations differ from the bb_coverage ones.
    df["score"] = df["synscore"]
    rng = np.random.default_rng(SEED)
    print(f"n = {len(df)}  (bootstrap {N_BOOT}x, seed {SEED})")
    print(f"{'SynOmega vs':10}  rho      95% CI")
    for name, col in [("SAscore", "sascore"), ("RAscore", "rascore"), ("SCScore", "scscore")]:
        x = df["score"].to_numpy()
        y = df[col].to_numpy()
        rho = spearmanr(x, y).statistic
        lo, hi = rho_ci(x, y, rng)
        print(f"{name:10}  {rho:+.3f}   [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
