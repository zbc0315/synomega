"""SynOmega synthesizability (solved@N + bb-coverage@N) for a SMILES list."""
import argparse, time, csv, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stock", required=True)
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--max-expansions", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from synomega.singlestep import TemplateGNN
    from synomega.stock import InMemoryStock
    from synomega.planner import Planner
    from synomega.synthesizability import SynthesizabilityScorer

    model = TemplateGNN.from_pretrained(args.run_dir, device=args.device)
    stock = InMemoryStock.from_keys_file(args.stock)
    planner = Planner(model, stock, algorithm="retrostar",
                      time_limit=args.time_limit, max_expansions=args.max_expansions,
                      max_depth=args.max_steps)
    scorer = SynthesizabilityScorer(planner)

    smis = [l.strip().split()[0] for l in open(args.smiles) if l.strip()]
    print(f"scoring {len(smis)} molecules...", flush=True)
    t0 = time.time()
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "solved", "bb_coverage", "min_steps", "sec"])
        for i, smi in enumerate(smis, 1):
            ts = time.time()
            try:
                r = scorer.score(smi, max_steps=args.max_steps)
                w.writerow([smi, int(bool(r.solved)), round(float(r.bb_coverage), 4),
                            getattr(r, "min_steps", "") or "", round(time.time()-ts, 2)])
            except Exception as e:
                w.writerow([smi, "", "", "", "ERR:" + type(e).__name__])
            if i % 50 == 0:
                f.flush()
                print(f"  {i}/{len(smis)}  ({(time.time()-t0)/i:.1f}s/mol avg)", flush=True)
    print(f"done in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
