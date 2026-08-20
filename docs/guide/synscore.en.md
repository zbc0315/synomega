# Synthesizability score (SynScore)

Give a target a continuous **synthesizability score** SynScore = \(1/(U+1)^U\)
(`U` = number of non-purchasable starting materials in the best route; all
purchasable → 1, lower as more are missing), for ranking a set of molecules. For
the score definition and operating point see [Research · Synthesizability
Score](../research/synscore.en.md); for installation see [Install &
Overview](index.en.md).

```bash
# defaults to the simplification-constrained model @ expansion width k=10 (recommended);
# --original switches to the unconstrained model
synomega score --targets targets.smi --out scores.jsonl
```

```python
import synomega

scorer = synomega.load_default_scorer()              # default simplify=True, k=10
report = scorer.score("CC(=O)Nc1ccccc1O")
print(report.as_dict())                              # score / solved / min_steps ...

batch = scorer.score_batch(open("targets.smi").read().split())
print(batch.solve_rate)
```
