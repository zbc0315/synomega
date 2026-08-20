# Reaction plausibility

Score 0–1 how likely a set of reactants actually gives the product, to **filter**
clearly-wrong single-step disconnections. Measured to be net-negative on
single-step top-k recall, so it is **off by default** — enable it explicitly. For
the model and evaluation see [Research · Reaction
Plausibility](../research/plausibility.en.md); for installation see [Install &
Overview](index.en.md).

```python
import synomega

# attach to the planner; drops implausible candidates at every step (drop-only, no re-ranking)
planner = synomega.load_default_planner(plausibility=True, plausibility_threshold=0.4)

# or score a batch of candidate reactions directly
from synomega.plausibility import PlausibilityScorer
scorer = PlausibilityScorer.default()
scores = scorer.score_reactions(["CC(=O)O.NC>>CC(=O)NC"])   # 0–1 per reaction
```
