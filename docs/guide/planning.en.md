# Multi-step route planning

**What it does**: given a target molecule, repeatedly call single-step retro over an
AND-OR graph to search for a full route down to **purchasable building blocks**.
Search algorithms and evaluation: [Research · Multi-step Route
Planning](../research/planning.en.md); install: [Install & Overview](index.en.md).

## Command line

```bash
synomega plan --target "CC(=O)Nc1ccccc1O" --max-steps 5 --simplify
```

## Python

```python
import synomega

planner = synomega.load_default_planner()          # default: original model + retrostar; downloads model + stock on first use
result = planner.plan("CC(=O)Nc1ccccc1O", max_depth=5)

print(result.solved)                                # whether an all-purchasable route was found
print(result.best_route.describe())                 # best route, step by step
for r in result.routes[:3]:                          # first few candidate routes
    print(r.num_steps, r.depth, r.bb_coverage)
print(result.stats.expansions, result.stats.terminated_by)   # search cost and stop reason
```

Example `best_route.describe()` output (numbers vary with model / stock):

```
target: CC(=O)Nc1ccccc1O
solved: True  steps: 2  depth: 2  bb_coverage: 1.00
  [1] ...>>CC(=O)Nc1ccccc1O   (score=0.43)
  [2] ...                      (score=0.22)
```

`solved=True` means every leaf is in the building-block set; `bb_coverage` is the
fraction of purchasable leaves (read it on a near-miss; 1.00 = fully solved).

## Parameters

| Parameter (CLI / Python) | Default | Meaning |
|---|---|---|
| `--algorithm` | retrostar | `retrostar` (default) / `mcts` (steadier with a weak single-step model) / `bfs` (baseline) |
| `--max-steps` / `max_depth=` | 5 | route depth cap |
| `--expansion-width` | 50 | single-step top-k candidate reactant sets per molecule node |
| `--time-limit` / `--max-expansions` | 60 s / 500 | search budget (time / node expansions) |
| `--exclude-target` | off | treat the target as not purchasable, avoiding a trivial zero-step solve |
| `--simplify` | off | use the simplification-constrained single-step model (cheaper search) |
| `--stock` / `--stock-is-keys` | download ZINC | custom building-block set (`.keys` or a SMILES catalogue) |

## Notes

- Caching is on by default (each molecule is expanded once); `Planner(cache_path="x.sqlite")`
  persists it to SQLite for reuse across processes.
- All three algorithms share the same AND-OR graph, budget, and route extractor, so
  their results are directly comparable.
- To get the route tree and search stats together from a single search, use
  `SynthesizabilityScorer(planner).score_detailed(smiles)`.
