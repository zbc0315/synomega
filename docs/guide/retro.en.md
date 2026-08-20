# Single-step retrosynthesis

Given a product, rank the likely **reactants** (disconnections). There is no
standalone CLI subcommand — it is the engine behind `plan` / `score`; to take
single-step retro candidates on their own, use Python. For the model and
evaluation see [Research · Single-step Retrosynthesis](../research/retro.en.md);
for installation see [Install & Overview](index.en.md).

```python
from synomega.singlestep import TemplateGNN

for p in TemplateGNN.default().predict("CC(=O)Nc1ccccc1O", top_k=5):
    print(p.score, p.reactants)          # p.reactants is a ranked tuple of canonical SMILES
```

`TemplateGNN.simplify()` is the **simplification-constrained variant** (emits only
disconnections that split the target into two or more precursors), the recommended
backend for synthesizability scoring.
