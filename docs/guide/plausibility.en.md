# Reaction plausibility

**What it does**: score 0–1 how likely a candidate reaction (a set of reactants → a
product) actually happens — "do these reactants really give this product". Its main
use is to **filter** clearly-implausible single-step disconnections (drop-only, no
re-ranking of survivors). Model (mapping-free dual-tower D-MPNN) and evaluation:
[Research · Reaction Plausibility](../research/plausibility.en.md); install:
[Install & Overview](index.en.md).

**Off by default**: measured to be net-negative on single-step top-k recall and it
adds latency, so it is not enabled unless you ask for it.

## Way 1: attach to the planner to filter every step

```python
import synomega

planner = synomega.load_default_planner(plausibility=True,
                                        plausibility_threshold=0.4)
# every single-step candidate in plan / score is then screened:
# disconnections whose reactants → target plausibility is below 0.4 are dropped
```

## Way 2: score a batch of reactions directly

```python
from synomega.plausibility import PlausibilityScorer

scorer = PlausibilityScorer.default()               # downloads the plausibility model on first use
scores = scorer.score_reactions([
    ("CC(=O)O.NCc1ccccc1", "CC(=O)NCc1ccccc1"),      # each item is a (reactants, product) tuple
    ("CCO.CC(=O)O",        "CC(=O)OCC"),
])
print(scores)   # -> e.g. [0.99, 0.95]; one [0,1] score per reaction; unparseable → 0.0
```

!!! warning "The input is tuples, not reaction SMILES"
    `score_reactions` takes an iterable of `(reactants_smiles, product_smiles)`
    tuples — **not** `"A.B>>C"` reaction-SMILES strings. Reactant/product graphs are
    cached, so scoring many disconnections of the same target is cheap.

## Parameters and notes

- `plausibility_threshold` (default 0.4): higher filters more aggressively; the
  filter only **drops** candidates, never re-ranks survivors.
- To keep a minimum number, over-fetch, or re-rank inside the filter, pass
  `plausibility_kwargs={"min_keep": ..., "overfetch": ..., "rerank": ...}` to
  `load_default_planner` or `Planner`.
- `PlausibilityScorer.default(device="cuda:0")` picks the device;
  `scorer.meta["val_auc"]` is the validation AUC.
