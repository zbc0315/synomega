# SynOmega vs AiZynthFinder (all 1000 ChEMBL targets, budget-aligned, GPU)

External-planner comparison on all 1000 ChEMBL drug targets, same ZINC building-block
set. AiZynthFinder (v4.4.1): public USPTO expansion policy + USPTO filter + ZINC
stock, on the **same GPU as SynOmega** and with its **search depth, width and iteration budget aligned to SynOmega**:
`cutoff_number=10` (top-10 templates per expansion), `max_transforms=5` (depth),
`iteration_limit=100`. Config: `../target_set/aizynth_config_aligned.yml`. SynOmega
`full` / `simplify` numbers from `../efficiency_coverage/`. Regenerate:
`python compare_aizynth.py`.

With depth/width/budget matched, the remaining differences are the single-step model
(commercial corpus vs USPTO policy) and the search algorithm (retro* vs MCTS);
hardware is matched (both on GPU).

## Solved rate (depth <= 5, all-purchasable route)

| Planner | Solved / 1000 | Rate |
|---|---|---|
| AiZynthFinder | 467 | 46.7% |
| SynOmega (full) | 818 | 81.8% |
| SynOmega (simplify) | 851 | 85.1% |

SynOmega reaches roughly **1.8x** AiZynthFinder's solved rate.

## Search time per target

| Planner | Median (s) |
|---|---|
| AiZynthFinder | 4.07 |
| SynOmega (full) | 0.49 |
| SynOmega (simplify) | 0.32 |

AiZynthFinder's median is **~13x** the simplify model and **~8x** the full model.
Node/iteration counts are not compared across the two search formulations.

## Per-target agreement: SynOmega (simplify) vs AiZynthFinder

| Category | Targets |
|---|---|
| Solved by both | 460 |
| SynOmega only | 391 |
| AiZynthFinder only | 7 |
| Neither | 142 |

391 targets solved only by SynOmega vs 7 only by AiZynthFinder. These are panel (d)
of the merged results figure (Fig. 3); the solved-rate bars are panel (c).
