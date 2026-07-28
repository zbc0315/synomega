"""Merge score CSVs and analyze bb-coverage vs SAscore/SCScore/RAscore.

Correlations, and the key claim: on the solved@N=0 subset (where binary
solvability is constant 0), does the continuous bb-coverage still vary and
correlate with an independent complexity score?
"""
import argparse, sys
import pandas as pd
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--rascore", default=None)
    args = ap.parse_args()

    syn = pd.read_csv(args.syn)
    base = pd.read_csv(args.base)
    df = syn.merge(base, on="smiles", how="inner")
    if args.rascore:
        ra = pd.read_csv(args.rascore)
        df = df.merge(ra, on="smiles", how="left")
    for c in ["solved", "bb_coverage", "sascore", "scscore", "rascore"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["bb_coverage", "sascore", "scscore"])
    n = len(df)
    print(f"molecules with all scores: {n}")
    print(f"solved@5 rate: {df['solved'].mean():.3f}  "
          f"bb_coverage mean {df['bb_coverage'].mean():.3f}")

    # higher bb_coverage/rascore = easier; higher SA/SC = harder -> expect negative corr
    scores = ["sascore", "scscore"] + (["rascore"] if "rascore" in df.columns else [])
    print("\n=== Spearman correlation (entire set) ===")
    print(f"{'pair':28} rho      p")
    pairs = [("bb_coverage", s) for s in scores] + \
            [("solved", s) for s in scores] + \
            [("sascore", "scscore")]
    if "rascore" in df.columns:
        pairs += [("bb_coverage", "rascore"), ("sascore", "rascore")]
    for a, b in pairs:
        sub = df.dropna(subset=[a, b])
        rho, p = spearmanr(sub[a], sub[b])
        print(f"{a+' vs '+b:28} {rho:+.3f}   {p:.1e}")

    # KEY: on solved==0 subset, does bb_coverage still vary & correlate?
    z = df[df["solved"] == 0]
    print(f"\n=== solved@5 == 0 subset (binary collapses): n={len(z)} ===")
    print(f"  bb_coverage: mean {z['bb_coverage'].mean():.3f}  std {z['bb_coverage'].std():.3f}  "
          f"range [{z['bb_coverage'].min():.2f}, {z['bb_coverage'].max():.2f}]")
    for s in scores:
        sub = z.dropna(subset=["bb_coverage", s])
        if len(sub) > 10:
            rho, p = spearmanr(sub["bb_coverage"], sub[s])
            print(f"  bb_coverage vs {s} (within solved=0): rho={rho:+.3f} p={p:.1e} n={len(sub)}")

    # divergence: molecules where bb_coverage and SAscore disagree most
    d = df.copy()
    d["sa_rank"] = d["sascore"].rank(pct=True)          # high = hard
    d["bb_rank"] = (1 - d["bb_coverage"]).rank(pct=True)  # high = hard (low coverage)
    d["gap"] = (d["sa_rank"] - d["bb_rank"]).abs()
    print("\n=== top divergence (SAscore vs bb-coverage) ===")
    for _, r in d.sort_values("gap", ascending=False).head(6).iterrows():
        print(f"  SA={r['sascore']:.2f} bbcov={r['bb_coverage']:.2f} solved={int(r['solved'])}  {r['smiles'][:70]}")


if __name__ == "__main__":
    main()
