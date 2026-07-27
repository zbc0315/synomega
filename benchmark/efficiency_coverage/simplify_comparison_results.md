# Simplification-constrained vs unconstrained model: efficiency-coverage comparison

Matched-budget comparison of the constrained ("simplify") single-step model against
the unconstrained ("full") model, on all **1000 ChEMBL drug molecules**
(`../target_set/targets.smi`), same building-block set (ZINC), same retro* budget.

- Budget: expansion width **k=10** (top-10 candidate reactant sets per node
  expansion), time limit 8 s, <=100 node expansions, depth <=5, 60 s hard
  wall-clock cap per molecule.
- Run single-process, sharded across GPUs.
- Script: `compare_simplify_vs_full.py`. Per-molecule results: `full.csv`, `simplify.csv`.
  Stats: `python paired_stats.py`.

## Aggregate (n=1000)

| Metric | full | simplify |
|---|---|---|
| Solved rate | 81.8% (818/1000) | **85.1% (851/1000)** |
| Solved by this model only | 35 | **68** |
| Mean bb-coverage | 0.919 | 0.919 |
| Mean expansions (all) | 38.5 | **32.0** |
| Mean expansions (solved by both, n=783) | 22.0 | **15.7** |
| Median time / target | 0.57 s | **0.32 s** |

## Paired (solved by both, n=783)

- Expansions: full mean 22.0 -> simplify mean **15.7** (**-28%**).
- Per molecule: simplify uses **fewer** expansions on **283**, **more** on **136** (tie 364).
- Wilcoxon signed-rank (one-sided, full > simplify): **p = 4.3e-21**.
- Solve-rate difference (McNemar on discordant 68 vs 35): **p = 0.001**.

## Conclusion

On out-of-distribution ChEMBL drug molecules the simplification constraint cuts node
expansions ~28% on jointly-solved targets and roughly halves the median search time,
**and modestly improves solvability** (85.1% vs 81.8%, +3.3 pp) at identical mean
bb-coverage (0.919). The constrained action space concentrates a limited budget on
productive (fragmenting) disconnections; the broader action space of the full model
disperses the same budget across non-simplifying branches.

## External baseline: AiZynthFinder

On all 1000 targets and the same ZINC stock, AiZynthFinder (public USPTO policy,
with search depth/width/iteration budget aligned to SynOmega) solved
**465/1000 (46.5%)**, vs 85.1% (simplify) / 81.8% (full) for SynOmega -- roughly
1.8x the solve rate. With budget matched, the remaining differences are the
single-step model and the search algorithm (retro* vs MCTS). Data & analysis:
`../aizynth_comparison/`.
