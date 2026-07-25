"""Calibrate the single-step plausibility-filter threshold on held-out reactions.

For each held-out REAL reaction (product P, true reactants R_true), run the
single-step model on P and score every candidate with the dual-tower plausibility
model. Two signals drive the threshold:
  * true-disconnection scores  -> the threshold must not delete real chemistry
  * all-candidate scores        -> how much the threshold prunes overall
Report a sweep of (false-delete rate on true disconnections, overall prune rate)
so a threshold with low false-delete and useful pruning can be picked.
"""
import argparse, os
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")


def canon_set(smiles: str) -> str:
    parts = []
    for comp in smiles.split("."):
        m = Chem.MolFromSmiles(comp)
        if m is None:
            return ""
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
        parts.append(Chem.MolToSmiles(m))
    return ".".join(sorted(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/dataset_filtered/test.parquet")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--plaus-ckpt", required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.environ["SYNOMEGA_PLAUSIBILITY_MODEL"] = args.plaus_ckpt

    from synomega.singlestep import TemplateGNN
    from synomega.plausibility import PlausibilityScorer
    ss = TemplateGNN.from_pretrained(args.run_dir, device=args.device)
    sc = PlausibilityScorer.default(device=args.device)

    df = pd.read_parquet(args.test, columns=["reactants", "product", "label"])
    df = df[df["label"] == 1].sample(args.n, random_state=0)
    targets, truths = [], []
    for r, p in zip(df["reactants"], df["product"]):
        cp = canon_set(p); cr = canon_set(r)
        if cp and cr:
            targets.append(cp); truths.append(cr)
    print(f"held-out reactions: {len(targets)}", flush=True)

    preds = ss.predict_batch(targets, top_k=args.top_k)
    # flatten candidates for one plausibility pass
    reactions, spans = [], []
    for tgt, plist in zip(targets, preds):
        s = len(reactions)
        for p in plist:
            reactions.append((p.smiles, tgt))
        spans.append((s, len(reactions)))
    scores = sc.score_reactions(reactions)

    all_scores, true_scores = [], []
    found = 0
    for (a, b), plist, truth in zip(spans, preds, truths):
        cand_scores = scores[a:b]
        all_scores.extend(cand_scores)
        matched = None
        for p, s in zip(plist, cand_scores):
            if canon_set(p.smiles) == truth:
                matched = s if matched is None else max(matched, s)
        if matched is not None:
            true_scores.append(matched); found += 1
    print(f"candidates total: {len(all_scores):,}  "
          f"true disconnection recovered in candidates: {found}/{len(targets)}", flush=True)

    def pct(xs, q):
        xs = sorted(xs)
        return xs[int(q / 100 * (len(xs) - 1))] if xs else float("nan")
    print(f"\ntrue-disconnection plaus percentiles: "
          f"p1={pct(true_scores,1):.3f} p2={pct(true_scores,2):.3f} "
          f"p5={pct(true_scores,5):.3f} p10={pct(true_scores,10):.3f} "
          f"p25={pct(true_scores,25):.3f} p50={pct(true_scores,50):.3f}")

    print("\nthr   false-delete(true)   prune-rate(all)   avg-kept/target")
    n_t = len(true_scores); n_a = len(all_scores); n_tgt = len(targets)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        fd = sum(1 for s in true_scores if s < t) / max(1, n_t)
        pr = sum(1 for s in all_scores if s < t) / max(1, n_a)
        kept = sum(1 for s in all_scores if s >= t) / max(1, n_tgt)
        print(f"{t:.1f}   {fd*100:6.2f}%              {pr*100:6.2f}%          {kept:5.1f}")


if __name__ == "__main__":
    main()
