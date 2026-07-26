# SynOmega JCIM paper — supporting data, scripts, results

Reproduction materials for the manuscript *"SynOmega: Constraining Single-Step
Disconnections to Simplifying Reactions for Efficient Synthesizability Scoring"*
(target: J. Chem. Inf. Model.). Manuscript source lives separately in
`retrosyn/paper_chem_syn/` (Overleaf project `6a6607303c122e7ddc95bc58`).

> Status: staged for the paper, **not yet committed / released** — may still change.
> Large artifacts (trained models, featurized shards, the reaction corpus) are **not**
> included here; the corpus is from a commercial database and is not redistributed.

## Layout & mapping to the paper

### `simplify_model/` — the simplifying-template filter (Methods; Fig. 2)
- `build_simplify_filter.py` — filters the 64,366 radius-0 templates to the
  simplifying (fragmentation) subset: retro templates whose product is one molecule
  and whose reactant side has two or more molecules. Output in `filter_meta.json`.
- `filter_meta.json` — **64,366 -> 42,028 kept** (65.3%).
- `build_simplify_featurized.py` — filters the featurized shards to the kept
  templates + remaps labels (no re-featurization).
- `r20_center_simplify.yaml` — training config for the constrained "simplify"
  model (warm-started from the full r20_center encoder). Best val top-1 = 0.575.

### `efficiency_coverage/` — simplify vs full, matched budget (Results; Table 1, Fig. 3)
- `compare_simplify_vs_full.py` — per-molecule scorer with a hard wall-clock cap.
- `subset200.smi` — the 200 drug-like targets.
- `full.csv`, `simplify.csv` — per-molecule results (solved, bb_coverage,
  min_steps, min_route_depth, expansions, sec, status).
- `simplify_comparison_results.md` — analysis (top-10 budget): **-30% expansions on
  127 solved-by-both (28.3->19.7), 53 vs 24 per-molecule; solved 72.5% (full) vs
  72.0% (simplify); unique 18 vs 17; mean bb-coverage 0.865 vs 0.822; median time
  1.27->0.59 s.** External baseline AiZynthFinder: 36.5% on the same set.

### `synthesizability_baselines/` — SynOmega vs SAscore/SCScore/RAscore (Results; Table 2, Fig. 4, Cost)
- `score_synomega.py` — bb-coverage / solved@N over the 1000 targets (per-mol `sec`).
- `score_baselines.py` — SAscore + SCScore (per-mol `sa_sec`/`sc_sec`).
- `score_rascore.py` — RAscore XGB (per-mol `ra_sec`; env needs numpy<2 +
  xgboost==1.2.1 + scikit-learn==1.0.2 to load the official pickle).
- `analyze_scores.py` — three-way Spearman + solved@5=0 subset analysis.
- `run_syn_shards.sh` — 8-shard parallel scoring driver.
- `*.csv` — `syn_targets` (bb_coverage/solved), `base_targets` (SA/SC),
  `rascore_targets` (RA), `syn_targets_timing` (clean single-process timing, n=150).
- `synth_eval_results.md` — **full-set rho: SA -0.522 / RA +0.518 / SC -0.209;
  solved@5=0 subset (n=234): bb-coverage s.d. 0.278, SA -0.387, RA +0.363;
  cost: median 1.25 s vs sub-25 ms scalar predictors.**

### `figures/`
- `fig1_pipeline.png`, `fig2_templates.png`, `fig3_efficiency.png`,
  `fig4_bbcoverage.png` — the four manuscript figures.
- `paper_chem_syn_*.py` — the generation scripts (read the CSVs above).

## Reproduce
1. Filter templates + train the simplify model: `simplify_model/`.
2. Efficiency-coverage comparison: `efficiency_coverage/compare_simplify_vs_full.py`
   on `subset200.smi`, once per model (full / simplify), then the analysis in the md.
3. Synthesizability baselines: run the three `score_*.py` on the 1000-target set,
   then `analyze_scores.py`.
