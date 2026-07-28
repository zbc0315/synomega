#!/usr/bin/env python3
"""Full analysis of the ChEMBL-target re-run: efficiency (simplify vs original),
score correlations (+ bootstrap CI), and paired Wilcoxon. AiZynth agreement is
added by analyze_aizynth() once aizynth_200.json.gz is present.

Reads the staged CSVs in this directory. Run: python analyze_chembl.py
"""
import csv
import os
import statistics as st
import numpy as np
from scipy.stats import spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260727


_PATHS = {
    "original_1000.csv": "../efficiency_coverage/original.csv",
    "simplify_1000.csv": "../efficiency_coverage/simplify.csv",
    "base_1000.csv": "../synthesizability_baselines/base_targets.csv",
    "rascore_1000.csv": "../synthesizability_baselines/rascore_targets.csv",
    "aizynth_200.json.gz": "../aizynth_comparison/aizynth_out.json.gz",
}


def load(name):
    return list(csv.DictReader(open(os.path.join(HERE, _PATHS[name]))))


def is_solved(r):
    return r["solved"] in ("1", "True", "true")


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def efficiency(n_label="1000"):
    orig = load("original_1000.csv")            # original model on all 1000
    simp = load("simplify_1000.csv")            # simplify model on all 1000
    omap = {r["smiles"]: r for r in orig}
    smap = {r["smiles"]: r for r in simp}
    common = [s for s in omap if s in smap]
    N = len(common)

    n_orig = sum(is_solved(omap[s]) for s in common)
    n_simp = sum(is_solved(smap[s]) for s in common)
    both = [s for s in common if is_solved(omap[s]) and is_solved(smap[s])]
    uniq_orig = sum(is_solved(omap[s]) and not is_solved(smap[s]) for s in common)
    uniq_simp = sum(is_solved(smap[s]) and not is_solved(omap[s]) for s in common)

    oexp_all = [num(omap[s]["expansions"]) for s in common if num(omap[s]["expansions"]) is not None]
    sexp_all = [num(smap[s]["expansions"]) for s in common if num(smap[s]["expansions"]) is not None]
    oexp_b = [num(omap[s]["expansions"]) for s in both]
    sexp_b = [num(smap[s]["expansions"]) for s in both]
    fewer = sum(sx < ox for ox, sx in zip(oexp_b, sexp_b))
    more = sum(sx > ox for ox, sx in zip(oexp_b, sexp_b))
    W, p = wilcoxon(oexp_b, sexp_b, alternative="greater")

    ocov = st.mean(num(omap[s]["bb_coverage"]) for s in common if num(omap[s]["bb_coverage"]) is not None)
    scov = st.mean(num(smap[s]["bb_coverage"]) for s in common if num(smap[s]["bb_coverage"]) is not None)
    otime = st.median(num(omap[s]["sec"]) for s in common)
    stime = st.median(num(smap[s]["sec"]) for s in common)

    # McNemar for solve-rate difference (discordant pairs)
    from scipy.stats import binomtest
    mcp = binomtest(min(uniq_orig, uniq_simp), uniq_orig + uniq_simp, 0.5).pvalue
    print(f"=== EFFICIENCY (all {N} targets) ===")
    print(f"solved: original {n_orig}/{N} ({100*n_orig/N:.1f}%)  simplify {n_simp}/{N} ({100*n_simp/N:.1f}%)")
    print(f"solved by both: {len(both)}   unique original: {uniq_orig}   unique simplify: {uniq_simp}"
          f"   (McNemar p={mcp:.4f})")
    print(f"mean expansions (all {N}): original {st.mean(oexp_all):.1f}  simplify {st.mean(sexp_all):.1f}"
          f"  ({100*(st.mean(oexp_all)-st.mean(sexp_all))/st.mean(oexp_all):.0f}% fewer)")
    print(f"mean expansions (solved-by-both n={len(both)}): original {st.mean(oexp_b):.1f}  "
          f"simplify {st.mean(sexp_b):.1f}  ({100*(st.mean(oexp_b)-st.mean(sexp_b))/st.mean(oexp_b):.0f}% fewer)")
    print(f"per-target: simplify fewer {fewer}, more {more}, tie {len(both)-fewer-more}")
    print(f"Wilcoxon (original>simplify, one-sided): W={W:.0f} p={p:.2e}")
    print(f"mean bb-coverage: original {ocov:.3f}  simplify {scov:.3f}")
    print(f"median time/target: original {otime:.2f}s  simplify {stime:.2f}s")


def aizynth():
    import gzip
    import json
    from rdkit import Chem
    canon = lambda s: (Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else s)
    az = json.load(gzip.open(os.path.join(HERE, _PATHS["aizynth_200.json.gz"])))["data"]
    azs = {canon(r["target"]): bool(r["is_solved"]) for r in az}
    azt = st.median(r["search_time"] for r in az)
    orig = {canon(r["smiles"]): is_solved(r) for r in load("original_1000.csv")}
    simp = {canon(r["smiles"]): is_solved(r) for r in load("simplify_1000.csv")}
    common = [k for k in azs if k in simp]
    n = len(common)
    n_az = sum(azs[k] for k in common)
    n_orig = sum(orig[k] for k in common if k in orig)
    n_simp = sum(simp[k] for k in common)
    both = sum(simp[k] and azs[k] for k in common)
    so = sum(simp[k] and not azs[k] for k in common)
    ao = sum(not simp[k] and azs[k] for k in common)
    nei = sum(not simp[k] and not azs[k] for k in common)
    print(f"\n=== AiZynthFinder (budget-aligned, n={n}) ===")
    print(f"solved: AiZynth {n_az}/{n} ({100*n_az/n:.1f}%)  original {n_orig}/{n} ({100*n_orig/n:.1f}%)  "
          f"simplify {n_simp}/{n} ({100*n_simp/n:.1f}%)")
    print(f"AiZynth median search_time: {azt:.1f}s")
    print(f"agreement (simplify vs AiZynth): both {both}  simplify-only {so}  AiZynth-only {ao}  neither {nei}")


def correlations():
    syn = {r["smiles"]: num(r["bb_coverage"]) for r in load("original_1000.csv")}
    solv = {r["smiles"]: is_solved(r) for r in load("original_1000.csv")}
    base = load("base_1000.csv")
    ra = {r["smiles"]: num(r["rascore"]) for r in load("rascore_1000.csv")}
    rows = []
    for r in base:
        s = r["smiles"]
        if s in syn and syn[s] is not None:
            rows.append((syn[s], num(r["sascore"]), num(r["scscore"]), ra.get(s)))
    print("\n=== SCORE CORRELATIONS (1000 targets) ===")
    print(f"n with all scores: {sum(1 for r in rows if None not in r)}")
    print(f"SynOmega solve rate: {100*sum(solv.values())/len(solv):.1f}%")
    rng = np.random.default_rng(SEED)
    names = [("SAscore", 1), ("RAscore", 3), ("SCScore", 2)]
    for name, idx in names:
        pairs = [(r[0], r[idx]) for r in rows if r[idx] is not None]
        x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
        rho = spearmanr(x, y).statistic
        boots = np.empty(10000)
        n = len(x)
        for i in range(10000):
            idxs = rng.integers(0, n, n)
            boots[i] = spearmanr(x[idxs], y[idxs]).statistic
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  SynOmega vs {name}: rho={rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  n={n}")

    # cost medians
    sa_sec = [num(r["sa_sec"]) for r in base if num(r["sa_sec"])]
    sc_sec = [num(r["sc_sec"]) for r in base if num(r["sc_sec"])]
    ra_sec = [num(r["ra_sec"]) for r in load("rascore_1000.csv") if num(r["ra_sec"])]
    syn_sec = [num(r["sec"]) for r in load("original_1000.csv") if num(r["sec"]) is not None and r["status"] == "ok"]
    print(f"\n=== COST (median per target) ===")
    print(f"SynOmega {st.median(syn_sec)*1000:.0f} ms | SAscore {st.median(sa_sec)*1000:.2f} ms | "
          f"SCScore {st.median(sc_sec)*1000:.2f} ms | RAscore {st.median(ra_sec)*1000:.1f} ms")


if __name__ == "__main__":
    efficiency()
    aizynth()
    correlations()
