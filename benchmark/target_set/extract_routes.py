#!/usr/bin/env python3
"""Extract the original-model and simplify-model best routes for a list of targets,
dumping each route tree to JSONL. Used to find illustrative examples where the two
models reach the same building blocks by different routes (e.g. the original model
detouring through a protecting-group step). Run per shard on the server.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original-run", required=True)
    ap.add_argument("--simp-run", required=True)
    ap.add_argument("--stock", required=True)
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from synomega.singlestep import TemplateGNN
    from synomega.stock import InMemoryStock
    from synomega.planner import Planner
    from synomega.synthesizability import SynthesizabilityScorer

    stock = InMemoryStock.from_keys_file(args.stock)

    def scorer(run):
        m = TemplateGNN.from_pretrained(run, device=args.device)
        p = Planner(m, stock, algorithm="retrostar", time_limit=8,
                    max_expansions=100, max_depth=5, expansion_width=10)
        return SynthesizabilityScorer(p)

    s_orig, s_simp = scorer(args.original_run), scorer(args.simp_run)
    smis = [l.strip().split()[0] for l in open(args.smiles) if l.strip()]
    with open(args.out, "w") as f:
        for smi in smis:
            rec = {"smiles": smi}
            try:
                for tag, sc in (("original", s_orig), ("simplify", s_simp)):
                    rep, res = sc.score_detailed(smi, max_steps=5)
                    br = res.best_route if res is not None else None
                    rec[tag] = {
                        "solved": bool(rep.solved),
                        "leaves": sorted(lf.smiles for lf in br.leaves if lf.in_stock) if br else [],
                        "route": br.to_dict() if br else None,
                    }
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"
            f.write(json.dumps(rec) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
