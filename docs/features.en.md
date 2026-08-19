# Main Features

SynOmega offers five capabilities: **single-step retrosynthesis, single-step forward prediction, multi-step route planning, synthesizability scoring, and reaction plausibility scoring**. This page provides a quick start for installation, the command line, and the Python API; the model/algorithm details of each capability are covered in the [Research Report](research/index.md).

## Installation

```bash
pip install synomega           # core: rdkit + numpy (the template-rule backend works as-is)
pip install "synomega[gnn]"    # add the D-MPNN neural single-step backend (torch)
```

The neural backend is an **optional extra**: even without torch you can run the whole pipeline on the pure template-rule backend; install `[gnn]` only when you need the accuracy of neural template classification. The default model weights and the ZINC purchasable building-block set are **downloaded on demand at first call** into a local cache (`~/.cache/synomega`, overridable via the environment variable `SYNOMEGA_CACHE`), and are not shipped with the wheel.

## Capability list

| Capability | Command line | Python entry point |
|---|---|---|
| Single-step retrosynthesis | — (called via `plan`/`score`) | `TemplateGNN.default().predict(smiles, top_k)` |
| Single-step forward prediction | `synomega forward` | `ForwardTemplateGNN.default().predict(reactants, top_k)` |
| Multi-step route planning | `synomega plan` | `synomega.load_default_planner().plan(target)` |
| Synthesizability scoring | `synomega score` | `synomega.load_default_scorer().score(smiles)` |
| Reaction plausibility | — (can be hooked into the planner) | `PlausibilityScorer.default().score_reactions([...])` |
| Building-block stock | `synomega build-stock` | `InMemoryStock.from_file(...)` |

## Command line

```bash
# single-step forward prediction: reactants (join multiple with .) -> ranked products
synomega forward "CC(=O)O.NC" --top-k 5

# multi-step planning: find routes to a target (default Retro*; --simplify uses the constrained model)
synomega plan --target "O=C(Nc1ccccc1)c1ccccc1" --max-steps 5 --simplify

# synthesizability scoring: batch-score a SMILES list (one per line)
# uses the simplification-constrained model by default (recommended for scoring); --original switches to the unconstrained model
synomega score --targets targets.smi --out scores.jsonl

# optionally pre-download the default weights and building-block stock
synomega download

# turn a catalogue (a SMILES column) into an InChIKey stock file for --stock
synomega build-stock --catalogue emolecules.smi --out stock_keys.txt
```

Common knobs for `plan` / `score`: `--algorithm {retrostar,mcts,bfs}`, `--expansion-width` (take the top-k candidates per node), `--max-steps` (upper bound on route depth), `--time-limit`, `--max-expansions`, `--device`, `--cache` (path to the SQLite expansion cache).

## Python API

Ready out of the box: the two convenience entry points automatically download the default model and building-block set into `~/.cache/synomega` at first call.

```python
import synomega

# multi-step planning: find routes to a target (default: unconstrained model + retrostar)
planner = synomega.load_default_planner()
result = planner.plan("CC(=O)Nc1ccccc1O")
print(result.best_route.describe())

# synthesizability scoring: defaults to the simplification-constrained model + k=10 (recommended for scoring; see the research report)
scorer = synomega.load_default_scorer()
report = scorer.score("CC(=O)Nc1ccccc1O")
print(report.as_dict())        # solved / bb_coverage / min_steps / score ...
```

When you need to use the single-step backend directly (for example, to fetch just the one-step retrosynthetic candidates):

```python
from synomega.singlestep import TemplateGNN

# product -> ranked reactant candidates
for p in TemplateGNN.default().predict("CC(=O)Nc1ccccc1O", top_k=5):
    print(p.score, p.smiles)   # p.reactants is a sorted canonical tuple
```

For lower-level assembly (custom model / stock / algorithm) use `Planner(model, stock, algorithm=...)` +
`SynthesizabilityScorer(planner)`; `load_default_*` are simply their default wrappers.

Forward prediction (independent of the retrosynthesis interface, so the planner will not misuse it):

```python
from synomega.forward import ForwardTemplateGNN

fwd = ForwardTemplateGNN.default()
for pred in fwd.predict("CC(=O)O.NC", top_k=5):
    print(pred.score, pred.product)   # meta["n_templates"] = number of templates that yield this product
```

## Design trade-offs (at a glance)

- **Narrow interface**: the single-step backend only implements `predict(smiles, top_k) -> [Prediction]`; search, scoring, and plausibility filtering are all built on top of this interface, so the backend can be swapped painlessly.
- **InChIKey as molecular identity**: AND-OR graph deduplication, stock lookup, and cache hits all share the same convention, which is why the graph is a DAG rather than a tree.
- **Layered optional dependencies**: the core depends only on rdkit+numpy; the neural backend and the SQLite very-large building-block stock are all enabled on demand.
- **Honest defaults**: single-step reaction plausibility filtering was evaluated to be a **net negative**, so it is **off by default** in the software, while remaining available to enable explicitly (see [Reaction Plausibility Scoring](research/plausibility.md)).
