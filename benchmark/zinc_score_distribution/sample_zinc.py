#!/usr/bin/env python3
"""Draw a reproducible random sample of SMILES from the ZINC building-block library
(the same purchasable set used as SynOmega's stock) for the Figure 1 (b-d) score
distributions. Source: zinc15_filtered.csv (one 'smiles' column, ~10.6M purchasable
ZINC15 molecules -- the SMILES the InChIKey stock was built from). Run on the server:

    python sample_zinc.py --src ~/zinc15_filtered.csv --n 20000 --out zinc_sample_20k.smi

The sample is then scored with score_baselines.py (SAscore + SCScore) and
score_rascore.py (RAscore); see README.md.
"""
import argparse
import pandas as pd

SEED = 20260727  # fixed seed -> the exact sample is reproducible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.src)
    s = df.sample(n=a.n, random_state=SEED)["smiles"]
    s.to_csv(a.out, index=False, header=False)
    print(f"wrote {len(s)} SMILES -> {a.out} (seed {SEED})")


if __name__ == "__main__":
    main()
