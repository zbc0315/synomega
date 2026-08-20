# Multi-component evolution

Starting from a set of reactants, repeatedly react two at a time and add the
products back to the pool, growing a forward **synthesis network**. Each molecule
carries a **total score** (`min(parent totals) × step probability`, starting
reactants = 1.0) and a **synthesis-tree depth**. For the algorithm and validation
see [Research · Multi-component Evolution](../research/evolution.en.md); for
installation see [Install & Overview](index.en.md).

```bash
synomega evolve --reactants "CC(=O)c1ccccc1.C=O.CNC" \
                --max-depth 3 --score-threshold 0.01 --out network.json
```

```python
from synomega.forward import ForwardTemplateGNN, MultiComponentEvolution

evo = MultiComponentEvolution(ForwardTemplateGNN.default(),
                              max_depth=3, score_threshold=0.01)
result = evo.evolve(["CC(=O)c1ccccc1", "C=O", "CNC"])   # three-component Mannich reactants
print(result.describe())
for m in result.top(10, min_depth=1):
    print(m.total_score, f"d{m.depth}", m.smiles)
result.close()
```

Common options: `--mode {memory,disk,auto}` (use `disk` with `--work-dir` for
many starting reactants, spilling to SQLite), `--forward-top-k` (products per
pair), `--frontier-width` (cap pairs per round to control fan-out),
`--no-self-pair` (forbid A+A).
