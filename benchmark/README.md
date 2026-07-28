# SynOmega JCIM paper -- supporting data, scripts, results

Reproduction materials for the manuscript *"SynOmega: Simplifying Retrosynthesis for
Efficient Synthesizability Scoring"* (J. Chem. Inf. Model.). The manuscript source
lives separately in `retrosyn/paper_chem_syn/` (Overleaf project
`6a6607303c122e7ddc95bc58`).

> Status: released with the paper. Large artifacts (trained models, featurized
> shards, the commercial reaction corpus) are **not** included here. The corpus is
> drawn from a commercial reaction database and is not redistributed; the trained
> single-step models download automatically through the PyPI package.

## Models

Two single-step template models are compared throughout:

- **original** -- unconstrained model over all **64,366** radius-0 templates.
- **simplifying** -- constrained to the **42,028** simplifying templates: retro
  templates whose product side is a single molecule and whose reactant side has two
  or more molecules (65.3% of the templates are kept).

In the data files these are the `original.*` and `simplify.*` arms. The `simplify`
label matches the released model asset `r20_center_simplify` and the PyPI package's
`simplify=True` API; it is the same word root as the paper's "simplifying".

## Target set

All benchmarks use **1000 drug molecules randomly sampled from ChEMBL 35**
(seed **20260727**), 5-60 heavy atoms, deduplicated by InChIKey; these are
out-of-distribution relative to the training reactions. The sampler
`target_set/sample_chembl.py` writes `targets.smi` + `sample_meta.json`. Unified
search budget for every run: expansion width **k=10**, **8 s** time limit, **<=100**
node expansions, depth **<=5**, **60 s** hard wall-clock cap per molecule.

## Layout and mapping to the paper

### `zinc_score_distribution/` -- Figure 1(a-c) and SI Table S1
SAscore / RAscore / SCScore distributions over **20,000 ZINC building-block
molecules** (`sample_zinc.py`, `zinc_sa_sc.csv`, `zinc_ra.csv`). Every molecule is
already purchasable, motivating a stock-aware, route-based score.

### `simplify_model/` -- the simplifying-template filter (Figure 2; SI S2)
- `build_simplify_filter.py` -- filters the 64,366 radius-0 templates to the
  simplifying set (product is one molecule, reactant side has two or more).
- `filter_meta.json` -- **64,366 -> 42,028 kept (65.3%)**.
- `build_simplify_featurized.py` -- filters the featurized shards to the kept
  templates and remaps labels (no re-featurization).
- `r20_center_simplify.yaml` -- training config for the constrained simplifying
  model (warm-started from the original r20_center encoder). Best val top-1 = 0.575.

### `ksweep/` -- Figure 3(a-c) and SI Table S5
Expansion-width sweep. Both models are run at **k = 3..10**, with the **original
model at k = 50 as the per-molecule gold reference**.
- `original_k03..10.csv`, `simplify_k03..10.csv` -- per-molecule sweep results.
- `gold_original_k50.csv` -- the gold reference.
- `ksweep_analyze.py` -- efficiency and scoring-accuracy tables vs the gold.

### `efficiency_coverage/` -- Figure 3(d) and SI Tables S6/S7
The k=10 head-to-head of the two models on all 1000 targets.
- `original.csv`, `simplify.csv` -- per-molecule solved / bb_coverage / min_steps /
  expansions / sec / status.
- `compare_simplify_vs_original.py` -- the per-molecule scorer (hard wall-clock cap).
- `paired_stats.py` -- Wilcoxon signed-rank on paired solved-by-both expansions.

### `target_set/routes/` + `route_examples.json` + `extract_routes.py` / `extract_candidates.py` -- Figure 3(e,f)
Route examples where the simplifying model reaches purchasable material in fewer
steps than the original model on a representative target. `extract_routes.py` dumps
both models' best routes to `routes/*.jsonl`; `extract_candidates.py` selects the
illustrative short-vs-long contrasts.

### `aizynth_comparison/` -- Figure 4
SynOmega vs **AiZynthFinder v4.4.1** on all 1000 targets, on the **same GPU** and
with **depth / width / iteration budget aligned** to SynOmega (`cutoff_number=10`,
`max_transforms=5`, `iteration_limit=100`; config
`../target_set/aizynth_config_aligned.yml`).
- `aizynth_out.json.gz` -- the AiZynthFinder run.
- `compare_aizynth.py` -- matches targets by canonical SMILES and reproduces the
  solved rate, per-target agreement, and search-time contrast. Node/iteration counts
  are not compared across the two search formulations.

### `synthesizability_baselines/` -- SUPPLEMENTARY only
Baseline-score implementations (SAscore / SCScore / RAscore) and their per-molecule
timing, behind **SI Table S1**'s timing column and the SI baseline section.
- `score_baselines.py` (SAscore + SCScore; point `SCSCORE_MODEL` at your local
  SCScore 1024-bit model), `score_rascore.py` (RAscore XGB), `score_synomega.py`.
- `bootstrap_ci.py`, `analyze_scores.py` -- Spearman correlation + bootstrap CI.
  **Note:** the Spearman-correlation analysis these compute is **no longer in the
  manuscript** (the correlation table and the cost section were removed). The scripts
  and data are kept for completeness and labelled supplementary.

### `target_set/` -- benchmark targets and run drivers (Methods; SI S3)
`sample_chembl.py` -> `targets.smi` + `sample_meta.json`; server drivers
`run_synomega_chembl.sh`, `run_ksweep_chembl.sh`, `run_ksweep_original.sh`,
`run_aizynth_chembl.sh`; `analyze_chembl.py` (one-shot efficiency + AiZynth analysis);
`aizynth_config_aligned.yml`.

### Figures
The manuscript figures are generated by `make_fig1_arch.py`, `make_fig2.py`,
`make_fig3_ksweep.py`, and `make_fig4_aizynth.py`, which live with the manuscript
source in `retrosyn/paper_chem_syn/figures/`, not in this repository.

## Headline results

- **Solved rate** (depth <= 5, all-purchasable route): **81.8%** (original) /
  **85.1%** (simplifying) / **46.7%** (AiZynthFinder). SynOmega is ~1.8x AiZynth.
- **Node expansions**: **38.8 -> 32.0** over all 1000 targets (18% fewer);
  **22.0 -> 15.7** on the 783 jointly-solved targets (~30% fewer). Two-sided
  Wilcoxon signed-rank **W = 20881, p = 8.6e-21**.
- **Solve-rate difference**: McNemar exact **p = 0.0015** (discordant 68
  simplifying-only / 35 original-only).
- **Mean bb-coverage**: **0.919** for both models.
- **Median search time**: **0.49 s** (original) / **0.32 s** (simplifying) /
  **4.1 s** (AiZynthFinder). AiZynth's median is ~13x the simplifying model and ~8x
  the original model.
- **SynOmega vs AiZynthFinder agreement**: both-solved **460** / SynOmega-only
  **391** / AiZynth-only **7** / neither **142**.
- **Templates**: **64,366 -> 42,028** kept (65.3%). ChEMBL sampling seed 20260727.

## Reproduce

1. Sample targets: `target_set/sample_chembl.py` -> `targets.smi` (seed 20260727).
2. Filter templates and train the simplifying model: `simplify_model/`
   (`build_simplify_filter.py`, then `build_simplify_featurized.py`, then train with
   `r20_center_simplify.yaml`).
3. Efficiency-coverage: `efficiency_coverage/compare_simplify_vs_original.py` on
   `targets.smi`, once per model (original = run_r20 / simplify = run_simplify), then
   `paired_stats.py`.
4. Expansion-width sweep: `target_set/run_ksweep_chembl.sh` (gold + simplify) and
   `target_set/run_ksweep_original.sh` (original), then `ksweep/ksweep_analyze.py`.
5. AiZynthFinder: `target_set/run_aizynth_chembl.sh` (all 1000, depth/width/budget
   aligned via `aizynth_config_aligned.yml`), then
   `aizynth_comparison/compare_aizynth.py`.
6. Supplementary baselines: the three `synthesizability_baselines/score_*.py` on the
   1000 targets, then `bootstrap_ci.py`.
