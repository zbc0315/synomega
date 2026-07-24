# synomega

Retrosynthesis toolkit: **single-step prediction → multi-step route planning → synthesizability scoring**.

```
synthesizability   is this target reachable from purchasable material, in N steps?
     ↑
search             Retro* / MCTS / best-first over an AND-OR graph
     ↑
singlestep         product SMILES -> ranked reactant candidates
```

The layers are decoupled by a deliberately narrow interface: a single-step
backend only implements `predict(smiles, top_k) -> [Prediction]`. Whether it is
a graph neural network, a transformer, or plain template matching is invisible
to the planner.

## Install

```bash
pip install -e .            # core: rdkit + numpy, no torch
pip install -e '.[gnn]'     # adds the D-MPNN neural backend
```

The neural backend is an **optional** extra on purpose — the template-rule
backend runs anywhere, with no GPU and no torch.

## Quick start

```python
from synomega import Planner, SynthesizabilityScorer
from synomega.singlestep import TemplateGNN
from synomega.stock import InMemoryStock

model   = TemplateGNN.from_pretrained("../ml-template-gnn/runs/uspto50k_r0_min10")
stock   = InMemoryStock.from_keys_file("emolecules.keys.gz")
planner = Planner(model, stock, algorithm="retrostar")

result = planner.plan("CC(=O)Nc1ccccc1", max_depth=5, time_limit=60)
print(result.solved)
print(result.best_route.describe())
```

```
target: CC(=O)Nc1ccccc1
solved: True  steps: 2  depth: 2  bb_coverage: 1.00
  [1] CC(=O)O.Nc1ccccc1>>CC(=O)Nc1ccccc1  (score=0.4348)
  [2] O=[N+]([O-])c1ccccc1>>Nc1ccccc1  (score=0.2174)
```

### Synthesizability

```python
scorer = SynthesizabilityScorer(planner)

r = scorer.score("CC(=O)Nc1ccccc1", max_steps=5)
r.solved         # True — a complete route to purchasable material exists
r.bb_coverage    # 1.0 — fraction of leaves that are buyable
r.min_depth      # 2 — steps in the shortest solved route

report = scorer.score_batch(targets, max_steps=5)
report.solve_rate         # headline benchmark number
report.mean_bb_coverage
report.to_dataframe()
```

## The two synthesizability metrics

These get conflated in the literature; synomega keeps them apart because they
answer different questions.

| Metric | Meaning | Use it for |
|---|---|---|
| `solved@N` / `solve_rate` | Binary: does a route of depth ≤ N exist whose leaves are **all** purchasable? | Comparing against published numbers |
| `bb_coverage@N` | Continuous: fraction of the best route's leaves that are purchasable | Ranking molecules by how close they are |

`bb_coverage` matters because most targets are unsolved at realistic step
limits. A 5-step route with 4 of 5 leaves buyable scores 0.8, not 0 — so a
near-miss is distinguishable from a total failure.

## CLI

```bash
# one-time: precompute InChIKeys so later loads take seconds, not minutes
synomega build-stock --catalogue emolecules.smi.gz --out emolecules.keys.gz

synomega plan  --target "CC(=O)Nc1ccccc1" --model runs/uspto50k_r0_min10 \
             --stock emolecules.keys.gz --stock-is-keys --max-steps 5

synomega score --targets targets.smi --model runs/uspto50k_r0_min10 \
             --stock emolecules.keys.gz --stock-is-keys \
             --max-steps 5 --out report.json
```

## Search algorithms

| Algorithm | Character | When |
|---|---|---|
| `retrostar` | Expands the frontier molecule with the lowest estimated **total route cost** (Chen et al. 2020) | Default |
| `mcts` | UCT with greedy rollouts; tolerates an unreliable top-1 | When the single-step model is weak |
| `bfs` | Best-first on `g + h` | Baseline, debugging |

All three share the AND-OR graph, the budget, and the route extractor, so they
are directly comparable.

## Design notes

**AND-OR graph, not a tree.** A molecule node is solved if it is in stock or
*any* of its reactions is solved; a reaction is solved if *all* its reactants
are. Molecules are interned by InChIKey, so an intermediate reached down two
branches is one node expanded once. Cycles are rejected when the edge is
created — a route that makes X from X is not a route.

**Batched expansion.** Search is naturally "expand one node, call the model
once", which leaves a GPU idle. The algorithms pull a batch of frontier nodes
and issue one `predict_batch`.

**Caching.** Search revisits the same molecule constantly. `Planner(cache=True)`
(the default) memoizes expansions; `cache_path=` persists them to SQLite so the
cache survives across runs.

**Stock membership is by InChIKey**, so keys computed here match a vendor
catalogue written by a different toolkit. There is deliberately no Bloom-filter
backend: false positives would inflate solve-rate and quietly break
comparability with published numbers.

## Status

Implemented and tested: chem layer, single-step interface + template-rule and
D-MPNN backends, expansion cache, AND-OR graph, all three search algorithms,
route extraction/scoring/serialization, synthesizability metrics, CLI.

Reserved but not implemented: reaction conditions (`RouteStep.conditions` is
always `None`); a condition model can fill it without touching the search layer.

```bash
pytest          # 43 tests
```
