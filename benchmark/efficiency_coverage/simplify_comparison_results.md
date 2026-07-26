# Simplification-constrained vs unconstrained model: efficiency-coverage comparison

Matched-budget comparison of the constrained ("simplify") single-step model against
the unconstrained ("full") model, on the same 200 drug-like targets
(`subset200.smi`), same building-block set (ZINC), same retro* budget.

- Budget: expansion width **k=10** (top-10 candidate reactant sets per node
  expansion), time limit 8 s, <=100 node expansions, depth <=5, 60 s hard
  wall-clock cap per molecule.
- Run single-process, one model per GPU.
- Script: `compare_simplify_vs_full.py`. Per-molecule results: `full.csv`, `simplify.csv`.

## Aggregate (n=200)

| Metric | full | simplify |
|---|---|---|
| Solved rate | 72.5% (145/200) | 72.0% (144/200) |
| Solved by this model only | 18 | 17 |
| Mean bb-coverage | 0.865 | 0.822 |
| Mean expansions (all) | 52.5 | **45.1** |
| Mean expansions (solved by both, n=127) | 28.3 | **19.7** |
| Median time / target | 1.27 s | **0.59 s** |

## Paired (solved by both, n=127)

- Expansions: full mean 28.3 -> simplify mean **19.7** (**-30%**).
- Per molecule: simplify uses **fewer** expansions on **53**, **more** on **24**.

## Conclusion

The simplification constraint cuts node expansions ~30% on jointly-solved targets
and roughly halves the median search time, **at matched solvability**: solved rate
tied (144 vs 145), unique solves even (17 vs 18), mean bb-coverage marginally lower
(0.822 vs 0.865). The constrained action space concentrates a limited budget on
productive (fragmenting) disconnections; the broader action space of the full model
disperses the same budget across non-simplifying branches.

## External baseline: AiZynthFinder

On the same 200 targets and ZINC stock, AiZynthFinder (public USPTO policy, default
settings) solved **73/200 (36.5%)**, vs 72.0% (simplify) / 72.5% (full) for
SynOmega -- roughly twice the solve rate on this set. Tool-level comparison at each
planner's standard operating point (different single-step models/settings), not a
controlled ablation. Data: `../aizynth_comparison/aizynth_out.json.gz`.
