# Simplifying-constrained vs original model: efficiency-coverage comparison

> This is **Figure 3(d)** in the manuscript and **SI Tables S6/S7**.

Matched-budget comparison of the constrained ("simplify") single-step model against
the unconstrained ("original") model, on all **1000 ChEMBL drug molecules**
(`../target_set/targets.smi`), same building-block set (ZINC), same retro* budget.

- Budget: expansion width **k=10** (top-10 candidate reactant sets per node
  expansion), time limit 8 s, <=100 node expansions, depth <=5, 60 s hard
  wall-clock cap per molecule.
- Run single-process, sharded across GPUs.
- Script: `compare_simplify_vs_original.py`. Per-molecule results: `original.csv`,
  `simplify.csv`. Stats: `python paired_stats.py`.

## Aggregate (n=1000)

| Metric | original | simplify |
|---|---|---|
| Solved rate | 81.8% (818/1000) | **85.1% (851/1000)** |
| Solved by this model only | 35 | **68** |
| Mean bb-coverage | 0.919 | 0.919 |
| Mean expansions (all) | 38.8 | **32.0** |
| Mean expansions (solved by both, n=783) | 22.0 | **15.7** |
| Median time / target | 0.49 s | **0.32 s** |

## Paired (solved by both, n=783)

- Expansions: original mean 22.0 -> simplify mean **15.7** (**~30% fewer**).
- Per molecule: simplify uses **fewer** expansions on **283**, **more** on **136** (tie 364).
- Wilcoxon signed-rank (two-sided): **W = 20881, p = 8.6e-21** (one-sided
  original > simplify: p = 4.3e-21).
- Solve-rate difference (McNemar exact on discordant 68 simplify-only vs 35
  original-only): **p = 0.0015**.

Over all 1000 targets mean expansions drop 38.8 -> 32.0 (18% fewer).

## Conclusion

On out-of-distribution ChEMBL drug molecules the simplifying constraint cuts node
expansions ~30% on jointly-solved targets and lowers the median search time by about a third,
and modestly raises solvability (85.1% vs 81.8%, +3.3 pp) at identical mean
bb-coverage (0.919). The constrained action space concentrates a limited budget on
productive simplifying disconnections; the broader action space of the original model
disperses the same budget across non-simplifying branches.

## External baseline: AiZynthFinder

On all 1000 targets and the same ZINC stock, AiZynthFinder (public USPTO policy,
with search depth/width/iteration budget aligned to SynOmega) solved
**467/1000 (46.7%)**, vs 85.1% (simplify) / 81.8% (original) for SynOmega -- roughly
1.8x the solve rate. With budget matched, the remaining differences are the
single-step model and the search algorithm (retro* vs MCTS). Data & analysis:
`../aizynth_comparison/`.
