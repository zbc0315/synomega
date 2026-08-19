# SynOmega

[![PyPI](https://img.shields.io/pypi/v/synomega.svg)](https://pypi.org/project/synomega/)
[![Python](https://img.shields.io/pypi/pyversions/synomega.svg)](https://pypi.org/project/synomega/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A retrosynthesis toolkit that turns a target molecule into synthesis routes and
a **continuous synthesizability score**. Three decoupled layers:

```
synthesizability   is this target reachable from purchasable material, in N steps?
     ↑
search             Retro* / MCTS / best-first over an AND-OR graph
     ↑
single-step        product SMILES -> ranked reactant candidates
```

The layers meet at a deliberately narrow interface — a single-step backend only
implements `predict(smiles, top_k) -> [Prediction]` — so the planner and the
scorer do not care whether predictions come from a graph neural network, a
transformer, or plain template matching.

## Documentation

Full documentation — features, CLI/API, and a per-module research report
(single-step forward & retro prediction, multi-step planning, reaction
plausibility, synthesizability scoring) with architectures, pseudocode, training
sets, and evaluation figures — is published at
**<https://zbc0315.github.io/synomega/>** (built from `docs/` on every push).

## Installation

```bash
pip install synomega           # core: rdkit + numpy
pip install "synomega[gnn]"    # + the D-MPNN neural single-step backend (torch)
```

The neural backend is an optional extra on purpose: the template-rule backend
runs anywhere, with no GPU and no PyTorch. Requires Python ≥ 3.10.

## Zero-config quickstart

`pip install synomega` ships only code. The first time you ask for the default
model or stock, synomega downloads them (a few hundred MB) into
`~/.cache/synomega` — the same way spaCy and HuggingFace fetch models. So this
works out of the box:

```python
import synomega

planner = synomega.load_default_planner()          # downloads model + stock once
print(planner.plan("CC(=O)Nc1ccccc1O").best_route.describe())

# Synthesizability scoring uses the simplification-constrained ("simplifying")
# single-step model by default -- the recommended model for scoring -- at the
# k=10 expansion-width operating point.
scorer = synomega.load_default_scorer()            # simplify=True, k=10 by default
score = scorer.score("CC(=O)Nc1ccccc1O", max_steps=5)
print(score.score, score.solved)                   # SynScore, and whether a purchasable route was found
```

Or pre-fetch from the command line, then use the CLI with no `--model`/`--stock`:

```bash
synomega download                                   # cache the default assets
synomega plan --target "CC(=O)Nc1ccccc1O" --max-steps 5
```

**Download mirrors.** Assets are hosted on both a USTC GitLab registry (fast in
China) and (soon) GitHub. SynOmega auto-selects the faster reachable one by
latency; override with `SYNOMEGA_MIRROR=ustc|github` or point
`SYNOMEGA_ASSETS_BASE=<url>` at your own mirror. Change the cache location with
`SYNOMEGA_CACHE`.

## Quick start

```python
from synomega import Planner, SynthesizabilityScorer
from synomega.singlestep import TemplateGNN
from synomega.stock import InMemoryStock

model   = TemplateGNN.from_pretrained("path/to/model_run")   # a trained checkpoint
stock   = InMemoryStock.from_keys_file("building_blocks.keys.gz")
planner = Planner(model, stock, algorithm="retrostar")

result = planner.plan("CC(=O)Nc1ccccc1", max_depth=5, time_limit=60)
print(result.solved)
print(result.best_route.describe())
```

```
target: CC(=O)Nc1ccccc1
solved: True  steps: 2  depth: 2
  [1] CC(=O)O.Nc1ccccc1>>CC(=O)Nc1ccccc1  (score=0.4348)
  [2] O=[N+]([O-])c1ccccc1>>Nc1ccccc1     (score=0.2174)
```

### Synthesizability scoring

```python
scorer = SynthesizabilityScorer(planner)

r = scorer.score("CC(=O)Nc1ccccc1", max_steps=5)
r.solved            # True — a complete route to purchasable material exists
r.score             # 1.0 — SynScore = 1/(U+1)**U (U = non-purchasable starting materials)
r.min_steps         # 2  — reactions in the shortest solved route
r.min_route_depth   # 2  — longest linear sequence of that route

report = scorer.score_batch(targets, max_steps=5)
report.solve_rate         # fraction of targets solved
report.to_dataframe()
```

**Excluding the target from stock.** A molecule that is itself a catalogue item
is otherwise "solved" in zero steps. Pass `exclude_target=True` to force a real
disconnection — the target is treated as not purchasable, while its intermediates
still are. Available on both planning and scoring (default off):

```python
planner.plan("CC(=O)Nc1ccccc1O", exclude_target=True)
scorer.score("CC(=O)Nc1ccccc1O", max_steps=5, exclude_target=True)
```

## Reaction-plausibility filtering

Every single-step prediction is screened by a **mapping-free dual-tower reaction-
plausibility model**: two shared-encoder D-MPNN towers embed the candidate
reactants and the target product separately (no atom mapping needed), and score
how likely those reactants actually give the product. Candidates below a
threshold are **dropped** — the filter only removes wrong disconnections, it never
re-ranks the survivors. Because search and synthesizability both expand through
the single-step model, this screens *every* single-step prediction in the system.

It is **off by default**: benchmarks (`scripts/BENCHMARKS.md`) show it does not
improve top-k retrieval of the recorded reaction (it is marginally negative,
−0.2…−0.9 pp) and adds latency (×1.7 GPU / ×4.6 CPU). Enable it when you want the
candidate list pruned of implausible disconnections rather than maximal recall.

```python
# off by default:
planner = synomega.load_default_planner()

# enable / tune:
planner = synomega.load_default_planner(plausibility=True)
planner = synomega.load_default_planner(plausibility=True, plausibility_threshold=0.5)

# bring your own single-step model + explicit scorer:
from synomega.plausibility import PlausibilityScorer
scorer = PlausibilityScorer.default(device="cpu")
planner = Planner(model, stock, plausibility=scorer, plausibility_threshold=0.4)
```

When enabled, each surviving prediction carries its raw plausibility in
`prediction.meta["plausibility"]`.

## Simplification-constrained model

An alternative single-step model is restricted, at the reaction-template level, to
**simplifying disconnections** — those that split the target into
two or more precursors. In multi-step search it reaches purchasable material with
fewer node expansions at matched solvability (about 30% fewer expansions and ~2x
faster median search on a drug-like benchmark; see [`benchmark/`](benchmark/) and
the accompanying paper). It downloads on first use like the default model:

```python
planner = synomega.load_default_planner(simplify=True)   # simplifying model

from synomega.singlestep import TemplateGNN
model = TemplateGNN.simplify(device="cpu")               # or load it directly
```

This is the default model for synthesizability scoring: `synomega.load_default_scorer()`
uses it out of the box (pass `simplify=False` to score with the unconstrained model),
and the `synomega score` CLI defaults to it too (`--original` reverts). Override the
download with `SYNOMEGA_SIMPLIFY_MODEL=/path/to/run_dir`.

## Two synthesizability metrics

These are conflated in the literature; SynOmega keeps them apart because they
answer different questions.

| Metric | Meaning | Use it for |
|---|---|---|
| `solved@N` / `solve_rate` | **Binary** — does a route of depth ≤ N exist whose leaves are *all* purchasable? | Comparing against published numbers |
| `score` | **Continuous** — `1/(U+1)**U` where `U` = number of non-purchasable starting materials (U=0 → 1.0, 1 → 0.5, 2 → 0.11; no route → 0) | Ranking with a sharp solved / few-missing / many-missing separation |

The headline **`score`** = `1/(U+1)**U`, where `U` is the number of non-purchasable
starting materials in the best route, falls off sharply with each missing building
block (U=0 → 1.0, 1 → 0.5, 2 → 0.11, 3 → 0.016), so it cleanly separates a solved
target, one missing a few materials, and one missing many. A near-miss is therefore
distinguishable from a total failure, and a set of molecules can be *ranked* rather
than merely split into solved/unsolved.

## Search algorithms

| Algorithm | Character | When to use |
|---|---|---|
| `retrostar` | Expands the frontier molecule with the lowest estimated total route cost (Chen et al. 2020) | Default |
| `mcts` | UCT with greedy rollouts; tolerant of an unreliable top-1 | Weak single-step model |
| `bfs` | Best-first on `g + h` | Baseline / debugging |

All three share the AND-OR graph, the budget, and the route extractor, so their
results are directly comparable.

## Command line

```bash
# one-time: precompute building-block InChIKeys so later loads take seconds
synomega build-stock --catalogue catalogue.smi.gz --out building_blocks.keys.gz

synomega plan  --target "CC(=O)Nc1ccccc1" --model path/to/model_run \
               --stock building_blocks.keys.gz --stock-is-keys --max-steps 5

synomega score --targets targets.smi --model path/to/model_run \
               --stock building_blocks.keys.gz --stock-is-keys \
               --max-steps 5 --out report.json
```

## Bring your own single-step model

Any object implementing the `SingleStepModel` interface plugs into the planner:

```python
from synomega.singlestep import SingleStepModel, Prediction

class MyModel(SingleStepModel):
    name = "my-model"
    def predict(self, smiles: str, top_k: int = 50) -> list[Prediction]:
        # return candidate disconnections, best first
        return [Prediction(reactants=("CCO", "CC(=O)O"), score=0.9)]

planner = Planner(MyModel(), stock, algorithm="retrostar")
```

Built-in backends: `TemplateGNN` (D-MPNN template classifier, needs `[gnn]`) and
`TemplateRuleModel` (pure template matching, no PyTorch).

## Design notes

- **AND-OR graph, not a tree.** A molecule is solved if it is in stock *or* any
  of its reactions is solved; a reaction is solved if *all* its reactants are.
  Molecules are interned by InChIKey, so an intermediate reached down two
  branches is one node, expanded once. Cycles are rejected at edge creation.
- **Batched expansion.** The search pulls a batch of frontier molecules and
  issues one `predict_batch`, so a GPU-backed model is not left idle.
- **Caching.** `Planner(cache=True)` (default) memoizes expansions;
  `cache_path=` persists them to SQLite across runs.
- **Stock membership is by InChIKey**, matching a vendor catalogue written by a
  different toolkit. There is deliberately no Bloom-filter backend — false
  positives would inflate solve-rate and break comparability with published
  numbers.

## Development

```bash
git clone https://github.com/zbc0315/synomega
cd synomega
pip install -e ".[gnn,dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
