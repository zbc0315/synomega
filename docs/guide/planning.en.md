# Multi-step route planning

Given a target molecule, search an AND-OR graph for a full route down to
purchasable building blocks. For the search algorithms and evaluation see
[Research · Multi-step Route Planning](../research/planning.en.md); for
installation see [Install & Overview](index.en.md).

```bash
synomega plan --target "CC(=O)Nc1ccccc1O" --max-steps 5 --simplify
```

```python
import synomega

planner = synomega.load_default_planner()            # default: original model + retrostar
result = planner.plan("CC(=O)Nc1ccccc1O")
print(result.solved)
print(result.best_route.describe())
```

Common knobs: `--algorithm {retrostar,mcts,bfs}`, `--expansion-width` (top-k
candidates per node), `--max-steps` (depth cap), `--time-limit`,
`--max-expansions`, `--exclude-target` (treat the target as not purchasable to
avoid a trivial zero-step solve), `--simplify` (use the
simplification-constrained model for cheaper search).
