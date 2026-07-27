# SynOmega JCIM paper — supporting data, scripts, results

Reproduction materials for the manuscript *"SynOmega: Constraining Single-Step
Disconnections to Simplifying Reactions for Efficient Synthesizability Scoring"*
(target: J. Chem. Inf. Model.). Manuscript source lives separately in
`retrosyn/paper_chem_syn/` (Overleaf project `6a6607303c122e7ddc95bc58`).

> Status: staged for the paper, **not yet committed / released** — may still change.
> Large artifacts (trained models, featurized shards, the reaction corpus) are **not**
> included here; the corpus is from a commercial database and is not redistributed.

> Target set (2026-07-27): all benchmarks use **1000 drug molecules randomly sampled
> from ChEMBL 35** (seed 20260727; out-of-distribution from the training reactions).
> Sampling + run drivers + analysis in `target_set/`. **All experiments use the
> full 1000** (the 200-subset runs are deprecated).

## Layout & mapping to the paper

### `target_set/` — the ChEMBL benchmark targets (Methods; SI S3)
- `sample_chembl.py` — reproducible seeded sampler (ChEMBL small molecules, largest
  organic fragment, 5-60 heavy atoms, InChIKey dedup) -> `targets.smi` + `sample_meta.json`.
- `run_synomega_chembl.sh` / `run_aizynth_chembl.sh` — server drivers (8s/100exp/k10).
- `analyze_chembl.py` — one-shot efficiency + correlation + AiZynth analysis.

### `simplify_model/` — the simplifying-template filter (Methods; Fig. 2)
- `build_simplify_filter.py` — filters the 64,366 radius-0 templates to the
  simplifying (fragmentation) subset: retro templates whose product is one molecule
  and whose reactant side has two or more molecules. Output in `filter_meta.json`.
- `filter_meta.json` — **64,366 -> 42,028 kept** (65.3%).
- `build_simplify_featurized.py` — filters the featurized shards to the kept
  templates + remaps labels (no re-featurization).
- `r20_center_simplify.yaml` — training config for the constrained "simplify"
  model (warm-started from the full r20_center encoder). Best val top-1 = 0.575.

### `efficiency_coverage/` — simplify vs full, matched budget (Results; Table 1, Fig. 3a,b)
- `compare_simplify_vs_full.py` — per-molecule scorer with a hard wall-clock cap.
- `full.csv`, `simplify.csv` — per-molecule results on all 1000 targets (solved,
  bb_coverage, min_steps, min_route_depth, expansions, sec, status).
- `paired_stats.py` — Wilcoxon signed-rank on paired solved-by-both expansions.
- `simplify_comparison_results.md` — **-28% expansions on 783 solved-by-both
  (22.0->15.7), Wilcoxon p=4e-21; solved 81.8% (full) vs 85.1% (simplify), +3.3pp
  McNemar p=0.001; unique 35 vs 68; mean bb-coverage 0.919 both; median time
  0.57->0.32 s.**

### `aizynth_comparison/` — SynOmega vs AiZynthFinder, all 1000 targets (Results; Fig. 3c,d)
- `aizynth_out.json.gz` — AiZynthFinder (v4.4.1) on all 1000 targets, **on the same
  GPU as SynOmega** and with **depth/width/iteration budget aligned** (cutoff_number=10,
  max_transforms=5, iteration_limit=100; config `../target_set/aizynth_config_aligned.yml`).
- `compare_aizynth.py` — matches targets by canonical SMILES and reproduces every
  AiZynthFinder-comparison number: solved rate, per-target agreement, search-time.
- `aizynth_comparison_results.md` — **solved 46.7% (AiZynth) vs 81.8/85.1%
  (SynOmega) ~1.8x; median search 4.1 s vs 0.32 s (simplify) ~13x / 0.49 s (full)
  ~8x; agreement both 460 / SynOmega-only 391 / AiZynth-only 7 / neither 142.**
  Node/iteration counts are not compared across the two search formulations.
  These feed Figure 3 (c: solved rate; d: agreement).

### `synthesizability_baselines/` — SynOmega vs SAscore/SCScore/RAscore (Results; Table 2, Cost)
- `score_synomega.py` — bb-coverage / solved@N over the 1000 targets (per-mol `sec`).
- `score_baselines.py` — SAscore + SCScore (per-mol `sa_sec`/`sc_sec`).
- `score_rascore.py` — RAscore XGB (per-mol `ra_sec`; env needs numpy<2 +
  xgboost==1.2.1 + scikit-learn==1.0.2 to load the official pickle).
- `bootstrap_ci.py` — three-way Spearman + 95% bootstrap CI (Table 2).
- `analyze_scores.py` — legacy three-way Spearman (the solved@5=0 breakdown it also
  prints is retained for reference but is **no longer in the manuscript**).
- `*.csv` — `syn_targets` (= full.csv, bb_coverage/solved), `base_targets` (SA/SC),
  `rascore_targets` (RA).
- `synth_eval_results.md` — **rho: SA -0.536 [-0.576,-0.492] / RA +0.503
  [+0.459,+0.544] / SC -0.208 [-0.265,-0.150]; cost median 0.57 s vs sub-ms/tens-ms
  scalar predictors.**

### Figures
The four manuscript figures and their generation scripts live with the manuscript
source in `retrosyn/paper_chem_syn/figures/` (e.g. `make_results_fig.py`, which
reads the `efficiency_coverage/` and `aizynth_comparison/` data), not here. Fig. 3
is a single 2x2 figure: (a,b) simplify vs full, (c,d) vs AiZynthFinder.

## Reproduce
1. Sample targets: `target_set/sample_chembl.py` -> `targets.smi` (seed 20260727).
2. Filter templates + train the simplify model: `simplify_model/`.
3. Efficiency-coverage: `efficiency_coverage/compare_simplify_vs_full.py` on
   `targets.smi`, once per model (full=run_r20 / simplify=run_simplify), then
   `paired_stats.py`.
4. Synthesizability baselines: the three `score_*.py` on the 1000 targets, then
   `synthesizability_baselines/bootstrap_ci.py`.
5. AiZynthFinder: `target_set/run_aizynth_chembl.sh` (all 1000, depth/width/budget
   aligned via `aizynth_config_aligned.yml`), then `aizynth_comparison/compare_aizynth.py`.
