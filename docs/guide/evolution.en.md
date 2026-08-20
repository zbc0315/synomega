# Multi-component evolution

**What it does**: from a set of starting reactants, repeatedly pick two molecules
from a growing "pool", react them, and add the products back — growing a forward
**synthesis network**. It answers which molecules these starting materials can
evolve into over several rounds, along which route, and with what confidence — good
for exploring multi-component / one-pot chemistry. Algorithm and end-to-end
validation: [Research · Multi-component Evolution](../research/evolution.en.md);
install: [Install & Overview](index.en.md).

**Each molecule carries two quantities**: a **total score** (`min(parent totals) ×
step probability`, starting reactants = 1.0 — a weakest-link product) and a
**synthesis-tree depth** (`max(parent depths) + 1` — tree height, not step count).

## Command line

```bash
synomega evolve --reactants "CC(=O)c1ccccc1.C=O.CNC" \
                --max-depth 3 --score-threshold 0.01 --out network.json
```

Real example output (three-component Mannich: acetophenone + formaldehyde +
dimethylamine):

```
molecules: 95497  pairs-run: 30628  reaction-edges: 145713  rounds: 3  stop: exhausted
top 15 products by total score:
  0.9021  d1  C=CC(=O)c1ccccc1       (step=0.9021)   ← enone intermediate (aldol condensation)
  0.7611  d2  CN(C)CCC(=O)c1ccccc1   (step=0.8438)   ← classic Mannich base (aza-Michael)
  0.4990  d1  CN(C)C ...
```

The network grows the enone intermediate at d1 and reaches the Mannich base at d2,
matching the textbook mechanism. `--out network.json` saves the whole network (all
reaction edges) for later analysis.

## Python

```python
from synomega.forward import ForwardTemplateGNN, MultiComponentEvolution

evo = MultiComponentEvolution(ForwardTemplateGNN.default(),
                              max_depth=3, score_threshold=0.01)
result = evo.evolve(["CC(=O)c1ccccc1", "C=O", "CNC"])   # three-component Mannich reactants
print(result.describe())                    # summary + top products
for m in result.top(10, min_depth=1):       # min_depth=1 keeps real products (excludes sources)
    print(m.total_score, f"d{m.depth}", m.smiles)
for edge in result.reactions():             # iterate every reaction edge
    print(edge.reaction_smiles, edge.step_score)
result.close()                              # in disk mode, always close (release the SQLite handle)
```

Handy `result` methods: `describe()`, `top(n, min_depth=, min_score=)`,
`reactions()`, `best_route(smiles)` (trace one molecule's best route), `to_json()`.
`with evo.evolve(...) as result:` closes automatically.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `--max-depth` | required | synthesis-tree depth cap (gates whether a molecule may keep reacting; not step count) |
| `--score-threshold` | required | a molecule below this total score cannot react further; higher prunes harder and runs faster |
| `--forward-top-k` | 5 | products taken per reaction pair |
| `--mode {memory,disk,auto}` | memory | use `disk` (SQLite, needs `--work-dir`) for many reactants; `auto` switches by source count |
| `--frontier-width` | unlimited | pair only the top-N highest-scoring molecules per round, capping the O(n²) fan-out |
| `--no-self-pair` | A+A allowed | forbid a molecule reacting with itself |

## Notes

- Cost grows as O(n²) in the number of sources; at scale always use
  `--frontier-width` + `--mode disk`.
- The score is a relative ordering of route **confidence**, not yield or
  thermodynamic feasibility, and does not replace judgement about conditions or
  selectivity.
- To keep a higher-scoring route, propagation may push a molecule's recorded depth
  above `max_depth` (score-first, intended).
