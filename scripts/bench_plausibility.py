"""Efficiency impact of the plausibility filter on single-step and multi-step.

Same molecules, two configs (filter off / on). Reports single-step per-call
latency and multi-step wall-clock + search expansions + solve rate, so the cost
of screening is separated from its effect on the search itself.
"""
import argparse, os, time, statistics as st
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")


def canon(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stock", required=True)
    ap.add_argument("--plaus-ckpt", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--n-single", type=int, default=40)
    ap.add_argument("--n-multi", type=int, default=25)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--max-expansions", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.environ["SYNOMEGA_PLAUSIBILITY_MODEL"] = args.plaus_ckpt

    from synomega.singlestep import TemplateGNN
    from synomega.stock import InMemoryStock
    from synomega.planner import Planner
    from synomega.plausibility import PlausibilityScorer, PlausibilityFilteredModel

    model = TemplateGNN.from_pretrained(args.run_dir, device=args.device)
    stock = InMemoryStock.from_keys_file(args.stock)
    scorer = PlausibilityScorer.default(device=args.device)
    filt = PlausibilityFilteredModel(model, scorer, threshold=0.4)

    df = pd.read_parquet(args.test, columns=["product", "label"])
    df = df[df["label"] == 1].sample(args.n_single + args.n_multi + 20, random_state=1)
    mols, seen = [], set()
    for p in df["product"]:
        c = canon(p)
        if c and c not in seen:
            seen.add(c); mols.append(c)
    single_mols = mols[:args.n_single]
    multi_mols = mols[:args.n_multi]

    # ---- warmup (load + JIT + first template applications) ----
    for m in mols[:3]:
        model.predict(m, top_k=args.top_k)
        filt.predict(m, top_k=args.top_k)

    # ---- single-step latency ----
    def timeit(fn, xs):
        ts = []
        for x in xs:
            t0 = time.perf_counter(); fn(x); ts.append((time.perf_counter() - t0) * 1000)
        return ts
    base_ss = timeit(lambda m: model.predict(m, top_k=args.top_k), single_mols)
    filt_ss = timeit(lambda m: filt.predict(m, top_k=args.top_k), single_mols)
    # avg surviving candidates
    n_base = st.mean(len(model.predict(m, top_k=args.top_k)) for m in single_mols[:10])
    n_filt = st.mean(len(filt.predict(m, top_k=args.top_k)) for m in single_mols[:10])

    print("\n================  SINGLE-STEP (top_k=%d, CPU)  ================" % args.top_k)
    print(f"molecules: {len(single_mols)}")
    print(f"  filter OFF : median {st.median(base_ss):7.1f} ms   mean {st.mean(base_ss):7.1f} ms")
    print(f"  filter ON  : median {st.median(filt_ss):7.1f} ms   mean {st.mean(filt_ss):7.1f} ms")
    print(f"  overhead   : x{st.mean(filt_ss)/st.mean(base_ss):.2f}  "
          f"(+{st.mean(filt_ss)-st.mean(base_ss):.1f} ms/call)")
    print(f"  avg candidates: {n_base:.1f} -> {n_filt:.1f} after filtering")

    # ---- multi-step planning ----
    def bench_plan(use_plaus):
        p = Planner(model, stock, algorithm="retrostar",
                    plausibility=(scorer if use_plaus else None),
                    plausibility_threshold=0.4, expansion_width=args.top_k)
        rows = []
        for m in multi_mols:
            t0 = time.perf_counter()
            r = p.plan(m, max_depth=args.max_depth,
                       max_expansions=args.max_expansions, time_limit=120.0)
            dt = (time.perf_counter() - t0) * 1000
            exp = getattr(r.stats, "expansions", None)
            rows.append((dt, exp, bool(r.solved)))
        return rows

    base_ms = bench_plan(False)
    filt_ms = bench_plan(True)
    def agg(rows):
        return (st.median([r[0] for r in rows]), st.mean([r[0] for r in rows]),
                st.mean([r[1] for r in rows if r[1] is not None]),
                sum(r[2] for r in rows))
    b = agg(base_ms); f = agg(filt_ms)
    print("\n================  MULTI-STEP (retrostar, depth<=%d, max_exp=%d)  ================"
          % (args.max_depth, args.max_expansions))
    print(f"molecules: {len(multi_mols)}")
    print(f"  filter OFF : median {b[0]:8.0f} ms   mean {b[1]:8.0f} ms   "
          f"avg expansions {b[2]:6.1f}   solved {b[3]}/{len(multi_mols)}")
    print(f"  filter ON  : median {f[0]:8.0f} ms   mean {f[1]:8.0f} ms   "
          f"avg expansions {f[2]:6.1f}   solved {f[3]}/{len(multi_mols)}")
    print(f"  wall-clock overhead : x{f[1]/b[1]:.2f}   "
          f"expansions ratio : x{f[2]/b[2]:.2f}")


if __name__ == "__main__":
    main()
