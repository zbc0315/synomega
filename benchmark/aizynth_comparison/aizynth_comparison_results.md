# SynOmega vs AiZynthFinder (all 1000 ChEMBL targets, budget-aligned)

External-planner comparison on all 1000 ChEMBL drug targets, same ZINC building-block
set. AiZynthFinder (v4.4.1): public USPTO expansion policy + USPTO filter + ZINC
stock, with its **search depth, width and iteration budget aligned to SynOmega**:
`cutoff_number=10` (top-10 templates per expansion), `max_transforms=5` (depth),
`iteration_limit=100`. Config: `../target_set/aizynth_config_aligned.yml`. SynOmega
`full` / `simplify` numbers from `../efficiency_coverage/`. Regenerate:
`python compare_aizynth.py`.

With depth/width/budget matched, the remaining differences are the single-step model
(SynOmega's D-MPNN template classifier on a commercial corpus vs AiZynthFinder's
USPTO policy) and the search algorithm (retro* vs MCTS).

## Solved rate (depth <= 5, all-purchasable route)

| Planner | Solved / 1000 | Rate |
|---|---|---|
| AiZynthFinder | 465 | 46.5% |
| SynOmega (full) | 818 | 81.8% |
| SynOmega (simplify) | 851 | 85.1% |

SynOmega reaches roughly **1.8x** AiZynthFinder's solved rate.

## Search time per target

| Planner | Median (s) |
|---|---|
| AiZynthFinder | 4.11 |
| SynOmega (full) | 0.57 |
| SynOmega (simplify) | 0.32 |

AiZynthFinder's median is **~13x** the simplify model and **~7x** the full model.
Node/iteration counts are not compared across the two search formulations.

## Per-target agreement: SynOmega (simplify) vs AiZynthFinder

| Category | Targets |
|---|---|
| Solved by both | 458 |
| SynOmega only | 393 |
| AiZynthFinder only | 7 |
| Neither | 142 |

393 targets solved only by SynOmega vs 7 only by AiZynthFinder. These are panel (d)
of the merged results figure (Fig. 3); the solved-rate bars are panel (c).
