"""Single-step top-1 accuracy and runtime, with plausibility filtering at several
thresholds, on the held-out r20_center test set.

top-1 = the model's #1 prediction uses the true template (test label). The filter
only deletes candidates, so it changes top-1 only when the original #1 is dropped
(a wrong #1 dropped -> possible gain; a correct #1 dropped -> a loss). One predict
+ one plausibility scoring per product is reused across all thresholds.
"""
import argparse, os, time, statistics as st
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")


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
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--thresholds", default="0.3,0.4,0.5")
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

    # warmup
    for m in targets[:3]:
        scorer.score_reactions([(x.smiles, m) for x in model.predict(m, top_k=args.top_k)])

    base_hit = 0
    filt_hit = {t: 0 for t in THR}
    filt_lost = {t: 0 for t in THR}   # correct #1 deleted
    filt_gain = {t: 0 for t in THR}   # wrong #1 deleted -> correct surfaced
    t_pred = t_score = 0.0
    n = 0
    for tgt, lab in zip(targets, labels):
        t0 = time.perf_counter()
        preds = model.predict(tgt, top_k=args.top_k)
        t_pred += time.perf_counter() - t0
        if not preds:
            continue
        n += 1
        t1 = time.perf_counter()
        scores = scorer.score_reactions([(p.smiles, tgt) for p in preds])
        t_score += time.perf_counter() - t1
        base_ok = preds[0].template_id == lab
        base_hit += base_ok
        for t in THR:
            kept = [p for p, s in zip(preds, scores) if s >= t] or [preds[0]]
            f_ok = kept[0].template_id == lab
            filt_hit[t] += f_ok
            if base_ok and not f_ok:
                filt_lost[t] += 1
            if not base_ok and f_ok:
                filt_gain[t] += 1

    print(f"\n=========  SINGLE-STEP TOP-1 (r20_center test, n={n}, top_k={args.top_k}, {args.device})  =========")
    print(f"  no filter        : top-1 = {base_hit/n:.4f}  ({base_hit}/{n})")
    for t in THR:
        print(f"  filter thr={t:<4} : top-1 = {filt_hit[t]/n:.4f}  ({filt_hit[t]}/{n})  "
              f"[correct-#1 deleted: {filt_lost[t]}, wrong-#1 fixed: {filt_gain[t]}]  "
              f"delta {100*(filt_hit[t]-base_hit)/n:+.2f} pp")
    print(f"\n=========  RUNTIME ({args.device})  =========")
    print(f"  predict only (no filter) : {1000*t_pred/n:7.1f} ms/mol")
    print(f"  + plausibility scoring   : {1000*t_score/n:7.1f} ms/mol")
    print(f"  total with filter        : {1000*(t_pred+t_score)/n:7.1f} ms/mol  "
          f"(x{(t_pred+t_score)/t_pred:.2f})")


if __name__ == "__main__":
    main()
