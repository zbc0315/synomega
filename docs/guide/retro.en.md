# Single-step retrosynthesis

**What it does**: given a product molecule, rank the likely **reactants** (a
one-step disconnection). It is the engine behind [route planning](planning.md) and
[synthesizability scoring](synscore.md), with no standalone CLI subcommand — to take
single-step retro candidates on their own, use Python. Model and evaluation:
[Research · Single-step Retrosynthesis](../research/retro.en.md); install:
[Install & Overview](index.en.md).

**Input / output**: input is a product SMILES; output is ranked candidates, each a
set of reactants (tuple of canonical SMILES) plus a score (template probability).

## Python

```python
from synomega.singlestep import TemplateGNN

model = TemplateGNN.default()                 # downloads the default model on first use
for p in model.predict("CC(=O)Nc1ccccc1O", top_k=5):
    print(round(p.score, 4), p.reactants)     # p.reactants is a ranked tuple of canonical SMILES
    print(p.smiles)                            # = ".".join(p.reactants), the reactant side as one string
    print(p.template_id, p.meta["center_avg"])
```

`predict` returns a list of `Prediction`: `reactants` (tuple), `score` (0–1 template
probability), `template_id`, `meta["center_avg"]` (mean reaction-center confidence,
used to break ties **between different match sites of the same template**). Default
`top_k=50`; batch with `model.predict_batch([...])`.

## Two backends

| Entry point | Action space | Use for |
|---|---|---|
| `TemplateGNN.default()` | all 64,366 templates | general single-step retro, route planning |
| `TemplateGNN.simplify()` | only "simplifying" disconnections (split into ≥2 precursors) | recommended for synthesizability scoring; cheaper multi-step search |

For your own checkpoint: `TemplateGNN.from_pretrained("run_dir")`, or point the env
vars `SYNOMEGA_MODEL` / `SYNOMEGA_SIMPLIFY_MODEL` at a run directory.

## Notes

- Single-step is one step only; to reach purchasable building blocks use
  [route planning](planning.md).
- Hitting the right template does not always uniquely reproduce the true reactants
  (regio/site ambiguity) — a structural ceiling of the template method; `center_avg`
  is exactly what picks the more plausible one among such same-template candidates.
