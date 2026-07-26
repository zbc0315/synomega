"""Single-step top-k accuracy vs plausibility-filter threshold (r20_center test).

top-k = the true template (test label) is among the first k surviving candidates.
The filter deletes wrong candidates without re-ordering, so for k>1 it can PROMOTE
the true disconnection into top-k by removing wrong candidates ranked above it.
One predict + one plausibility scoring per product is reused across all k and
thresholds. Also reports runtime.
"""
import argparse, os, time, json
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

KS = [1, 3, 5, 10]


def strip(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--plaus-ckpt", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--thresholds", default="0.3,0.4,0.5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.environ["SYNOMEGA_PLAUSIBILITY_MODEL"] = args.plaus_ckpt
    THR = [float(x) for x in args.thresholds.split(",")]

    from synomega.singlestep import TemplateGNN
    from synomega.plausibility import PlausibilityScorer
    model = TemplateGNN.from_pretrained(args.run_dir, device=args.device)
    scorer = PlausibilityScorer.default(device=args.device)

    df = pd.read_parquet(args.test, columns=["product_smiles", "label"]).sample(
        args.n, random_state=0)
    targets, labels = [], []
    for p, lab in zip(df["product_smiles"], df["label"]):
        c = strip(p)
        if c:
            targets.append(c); labels.append(int(lab))
    for m in targets[:3]:
        scorer.score_reactions([(x.smiles, m) for x in model.predict(m, top_k=args.top_k)])

    configs = ["base"] + [f"thr={t}" for t in THR]
    hit = {c: {k: 0 for k in KS} for c in configs}
    t_pred = t_score = 0.0
    n = 0
    for tgt, lab in zip(targets, labels):
        t0 = time.perf_counter(); preds = model.predict(tgt, top_k=args.top_k)
        t_pred += time.perf_counter() - t0
        if not preds:
            continue
        n += 1
        t1 = time.perf_counter()
        scores = scorer.score_reactions([(p.smiles, tgt) for p in preds])
        t_score += time.perf_counter() - t1
        ordered = {
            "base": preds,
            **{f"thr={t}": ([p for p, s in zip(preds, scores) if s >= t] or [preds[0]])
               for t in THR},
        }
        for c, lst in ordered.items():
            tids = [p.template_id for p in lst]
            for k in KS:
                if lab in tids[:k]:
                    hit[c][k] += 1

    lines = []
    lines.append(f"single-step top-k vs plausibility filter (r20_center test, "
                 f"n={n}, top_k={args.top_k}, {args.device})")
    header = "config      " + "".join(f"top-{k:<7}" for k in KS)
    lines.append(header)
    base = hit["base"]
    for c in configs:
        row = f"{c:<11} " + "".join(f"{hit[c][k]/n:.4f}      " for k in KS)
        if c != "base":
            row += "  Δ " + " ".join(f"{100*(hit[c][k]-base[k])/n:+.2f}" for k in KS) + " pp"
        lines.append(row)
    lines.append("")
    lines.append(f"runtime ({args.device}): predict {1000*t_pred/n:.1f} ms/mol  "
                 f"+scoring {1000*t_score/n:.1f} ms  = {1000*(t_pred+t_score)/n:.1f} ms "
                 f"(x{(t_pred+t_score)/t_pred:.2f})")
    out = "\n".join(lines)
    print("\n" + out)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out + "\n")
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
