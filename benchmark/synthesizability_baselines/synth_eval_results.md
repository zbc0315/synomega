# SynOmega score vs SAscore/SCScore/RAscore (1000 ChEMBL targets)

> **SUPPLEMENTARY ONLY.** These scripts and CSVs back the SI baseline section and the
> per-molecule timing column of **SI Table S1**. The Spearman-correlation analysis
> reported below is **NO LONGER in the manuscript** (the correlation table and the
> cost section were removed); the material is kept here for completeness.

Data: 1000 ChEMBL drug molecules (`../target_set/targets.smi`, seed 20260727, see
`../target_set/`). Scoring: SynOmega bb_coverage / solved, unified budget
**top-10 / 8 s / 100 expansions / depth 5 / 60 s hard cap** (identical to the
efficiency comparison, using the unconstrained original model; `syn_targets.csv`
= `../efficiency_coverage/original.csv`). Baselines: SAscore (RDKit sascorer),
SCScore (Coley numpy standalone), RAscore (XGB).
Scripts: `score_synomega.py` / `score_baselines.py` / `score_rascore.py`;
correlation + CI: `bootstrap_ci.py`.

## Overview (n=1000)
- SynOmega solved rate = **81.8%**; bb_coverage mean = 0.919 (over all 1000, solved
  molecules recorded as 1.0).

## Spearman correlation (entire set n=1000, 10000x bootstrap 95% CI)
| SynOmega bb_coverage vs | rho | 95% CI | note |
|---|---|---|---|
| **SAscore** | **-0.536** | [-0.576, -0.492] | SA low = easy, bb high = easy -> negative is correct, strong |
| **RAscore** | **+0.503** | [+0.459, +0.544] | RA high = easy -> positive is correct, strong |
| SCScore | -0.208 | [-0.265, -0.150] | sign correct but weak |

All three CIs exclude 0; SA/RA strong, SC weak. Two independent accessibility scores
cross-check the plausibility of the SynOmega score. (This was positioned as a sanity
check rather than a validity proof: bb_coverage saturates to 1 on the ~82% solved
targets, so the correlation is driven mainly by the unsolved tail. Removed from the
current manuscript.)

## Per-molecule timing (cost)
Per-molecule timing in `base_targets.csv` (`sa_sec`/`sc_sec`), `rascore_targets.csv`
(`ra_sec`), `syn_targets.csv` (`sec`).

| Method | Median | Note |
|---|---|---|
| SAscore | 0.22 ms | fingerprint / fragment counts |
| SCScore | 65 ms | 1024-bit FP forward (incl. per-call overhead) |
| RAscore (XGB) | 63 ms | ECFP counts + XGBoost |
| SynOmega bb_coverage | **0.57 s** | real multi-step route search |

SynOmega is ~0.57 s/molecule, about 3 orders of magnitude slower than the fastest
SAscore (and ~9x SC/RA), because it runs an actual route search. A tail remains
(retro* does not strictly honour time_limit on pathological molecules); the hard 60 s
cap bounds it.

## Change log
- 2026-07-27: target set switched from "reaction-corpus products" to **ChEMBL 35
  random drug molecules** (out-of-distribution external test); budget unified to
  8 s / 100 expansions; correlations recomputed (SA/RA/SC close to previous values).
  The old solved==0 subset argument was removed from the paper.
- 2026-07-27: **SynOmega score changed to `score = bb_coverage + (+1 when solved)`**
  (solved -> 2, unsolved -> [0,1)), widening the solved/unsolved gap (in
  `MoleculeReport.score`). This is a per-sequence monotone transform, so the
  **Spearman coefficients are unchanged** (-0.536/+0.503/-0.208, identical to
  bb_coverage; `bootstrap_ci.py` now validates with score); solve rate is unchanged.
