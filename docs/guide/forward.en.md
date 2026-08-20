# Single-step forward prediction

Given reactants, rank the likely **products**. For the model and evaluation see
[Research · Single-step Forward Prediction](../research/forward.en.md); for
installation see [Install & Overview](index.en.md).

```bash
synomega forward "CC(=O)O.NCc1ccccc1" --top-k 5     # dot-separate multiple molecules
```

```python
from synomega.forward import ForwardTemplateGNN

fwd = ForwardTemplateGNN.default()                   # forward model downloads on first use
for pred in fwd.predict("CC(=O)O.NCc1ccccc1", top_k=5):
    print(pred.score, pred.product, pred.template_id)
```

Output is ranked products, each with `product` (SMILES), `score` (forward
probability), and `template_id`. `--topk-templates` (default 10) sets how many
templates are searched.
