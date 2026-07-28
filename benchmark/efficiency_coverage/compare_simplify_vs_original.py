"""Efficiency vs coverage comparison for the simplifying-template constrained
single-step model against the original model, on the SAME target set with the SAME
search budget.

Per molecule (via SynthesizabilityScorer.score): solved, bb_coverage, min_steps,
min_route_depth (LLS), expansions, wall-clock sec, terminated status. A hard
per-molecule wall-clock cap (SIGALRM) bounds retro*'s occasional failure to honour
time_limit on pathological targets; capped molecules are recorded status=TIMEOUT.

Run once per model (original=run_r20, simplify=run_simplify) on the same --smiles,
then diff the two CSVs.

Usage:
    python compare_simplify_vs_original.py --run-dir <dir> --stock <keys> \
        --smiles subset.smi --out original.csv \
        --time-limit 8 --max-expansions 100 --hard-timeout 60 --device cuda:0
"""
import argparse, csv, signal, time


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stock", required=True)
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--time-limit", type=float, default=8.0)
    ap.add_argument("--max-expansions", type=int, default=100)
    ap.add_argument("--expansion-width", type=int, default=10)
    ap.add_argument("--hard-timeout", type=int, default=60)
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
                      max_depth=args.max_steps, expansion_width=args.expansion_width)
    scorer = SynthesizabilityScorer(planner)

    smis = [l.strip().split()[0] for l in open(args.smiles) if l.strip()]
    signal.signal(signal.SIGALRM, _alarm)
    print(f"scoring {len(smis)} molecules with {args.run_dir}", flush=True)
    t0 = time.time()
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "solved", "bb_coverage", "num_leaves",
                    "num_purchasable", "u", "synscore", "min_steps",
                    "min_route_depth", "expansions", "sec", "status"])
        for i, smi in enumerate(smis, 1):
            ts = time.time()
            signal.alarm(args.hard_timeout)
            try:
                r = scorer.score(smi, max_steps=args.max_steps)
                signal.alarm(0)
                w.writerow([smi, int(bool(r.solved)), round(float(r.bb_coverage), 4),
                            r.num_leaves, r.num_purchasable_leaves,
                            r.num_unpurchasable_leaves, round(float(r.score), 6),
                            getattr(r, "min_steps", "") or "",
                            getattr(r, "min_route_depth", "") or "",
                            getattr(r, "expansions", ""), round(time.time() - ts, 2), "ok"])
            except _Timeout:
                w.writerow([smi, "", "", "", "", "", "", "", "", "", round(time.time() - ts, 2), "TIMEOUT"])
            except Exception as e:
                signal.alarm(0)
                w.writerow([smi, "", "", "", "", "", "", "", "", "",
                            round(time.time() - ts, 2), "ERR:" + type(e).__name__])
            if i % 25 == 0:
                f.flush()
                print(f"  {i}/{len(smis)}  ({(time.time()-t0)/i:.1f}s/mol avg)", flush=True)
    print(f"done in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
