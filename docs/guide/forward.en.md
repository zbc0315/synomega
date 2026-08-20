# Single-step forward prediction

**What it does**: given a set of reactants (one or more molecules, dot-separated
into one SMILES), rank the most likely **products**. It is the mirror of the
retro single-step model — it reuses the same 64,366-template library, reverses the
matched retro templates, and applies them forward with RDKit. Model and evaluation:
[Research · Single-step Forward Prediction](../research/forward.en.md); install:
[Install & Overview](index.en.md).

**Input / output**: input is a reactant SMILES (e.g. `CC(=O)O.NCc1ccccc1` = acetic
acid + benzylamine); output is a ranked list, each candidate carrying a product
SMILES, a score (forward probability, higher = more confident), and the template id.

## Command line

```bash
synomega forward "CC(=O)O.NCc1ccccc1" --top-k 5
```

Real example output (acetic acid + benzylamine — amide coupling dominates):

```
 1. CC(=O)NCc1ccccc1	score=0.8434	template=0
 2. CC(=O)OCc1ccccc1	score=0.0176	template=3418
 3. CCNCc1ccccc1	score=0.0158	template=3376
 4. NCC1CCCCC1	score=0.0027	template=918
 5. CCO	score=0.0025	template=26
```

The top-1 `CC(=O)NCc1ccccc1` (N-benzylacetamide) is the amide from the acid + amine,
at 0.84 — far above the rest.

## Python

```python
from synomega.forward import ForwardTemplateGNN

fwd = ForwardTemplateGNN.default()             # downloads the forward model on first use (~149 MiB)
for pred in fwd.predict("CC(=O)O.NCc1ccccc1", top_k=5):
    print(pred.score, pred.product, pred.template_id)
    print(pred.meta["n_templates"])            # how many templates produce this product (support)
```

`predict` returns a list of `ForwardPrediction`: `product` (canonical SMILES),
`score` (0–1 forward probability), `template_id`, `meta["n_templates"]`. Use
`fwd.predict_batch([...])` for batches (faster on GPU).

## Parameters

| Parameter (CLI / Python) | Default | Meaning |
|---|---|---|
| `--top-k` / `top_k` | 10 | how many ranked products to return |
| `--topk-templates` / `topk_templates` | 10 | how many top softmax templates to apply forward; raising it can surface rarer products but is slower |
| `--model` / `from_pretrained(run_dir)` | auto-download | use your own forward-model run directory |
| `--device` / `device` | auto | `cpu` or `cuda:0` |

## Reading the output / notes

- The score is a template softmax probability: a product inherits the max
  probability among the templates that produce it; ties break by how many
  templates support it (`meta["n_templates"]`).
- Products are sanitized, freed of radical/carbene artifacts, reduced to the
  largest organic fragment, and canonicalized.
- Accuracy is capped by the template method (validation product top-1 ≈ 0.64) —
  treat results as **candidates**, not guarantees.
