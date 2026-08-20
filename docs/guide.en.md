# Feature Guide

SynOmega exposes six capabilities. This page gives the **how-to** for each
(command line and Python API); the model / algorithm behind each is in the
matching [research report](research/index.md) chapter — the two correspond
one-to-one:

| Feature | How to use (this page) | How it works (research) |
|---|---|---|
| Single-step forward prediction | [↓](#forward) | [research/forward](research/forward.en.md) |
| Multi-component evolution | [↓](#evolution) | [research/evolution](research/evolution.en.md) |
| Single-step retrosynthesis | [↓](#retro) | [research/retro](research/retro.en.md) |
| Multi-step route planning | [↓](#planning) | [research/planning](research/planning.en.md) |
| Reaction plausibility | [↓](#plausibility) | [research/plausibility](research/plausibility.en.md) |
| Synthesizability score (SynScore) | [↓](#synscore) | [research/synscore](research/synscore.en.md) |

## Install

```bash
pip install synomega           # core: rdkit + numpy (the template-rule backend works as is)
pip install "synomega[gnn]"    # + the D-MPNN neural single-step backend (torch), recommended
```

The neural backend is an **optional extra**: the template-rule backend runs
without torch; install `[gnn]` when you want the neural template classifier
(forward / retro / evolution / plausibility all build on it). The default model
weights and the ZINC in-stock building-block set are **downloaded on first use**
into `~/.cache/synomega` (override with `SYNOMEGA_CACHE`; pick a mirror with
`SYNOMEGA_MIRROR=ustc|github`), not shipped in the wheel. Pre-fetch with
`synomega download`. Requires Python ≥ 3.10.

## Single-step forward prediction {#forward}

Given reactants, rank the likely **products**.

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
probability), and `template_id`. See [Single-step Forward
Prediction](research/forward.en.md).

## Multi-component evolution {#evolution}

Starting from a set of reactants, repeatedly react two at a time and add the
products back to the pool, growing a forward **synthesis network**. Each molecule
carries a **total score** (`min(parent totals) × step probability`, starting
reactants = 1.0) and a **synthesis-tree depth**.

```bash
synomega evolve --reactants "CC(=O)c1ccccc1.C=O.CNC" \
                --max-depth 3 --score-threshold 0.01 --out network.json
```

```python
from synomega.forward import ForwardTemplateGNN, MultiComponentEvolution

evo = MultiComponentEvolution(ForwardTemplateGNN.default(),
                              max_depth=3, score_threshold=0.01)
result = evo.evolve(["CC(=O)c1ccccc1", "C=O", "CNC"])   # three-component Mannich reactants
print(result.describe())
for m in result.top(10, min_depth=1):
    print(m.total_score, f"d{m.depth}", m.smiles)
result.close()
```

Common options: `--mode {memory,disk,auto}` (use `disk` with `--work-dir` for
many starting reactants, spilling to SQLite), `--forward-top-k` (products per
pair), `--frontier-width` (cap pairs per round to control fan-out),
`--no-self-pair` (forbid A+A). See [Multi-component
Evolution](research/evolution.en.md).

## Single-step retrosynthesis {#retro}

Given a product, rank the likely **reactants** (disconnections). There is no
standalone CLI subcommand — it is the engine behind `plan` / `score`; to take
single-step retro candidates on their own, use Python:

```python
from synomega.singlestep import TemplateGNN

for p in TemplateGNN.default().predict("CC(=O)Nc1ccccc1O", top_k=5):
    print(p.score, p.reactants)          # p.reactants is a ranked tuple of canonical SMILES
```

`TemplateGNN.simplify()` is the **simplification-constrained variant** (emits only
disconnections that split the target into two or more precursors), the recommended
backend for synthesizability scoring. See [Single-step
Retrosynthesis](research/retro.en.md).

## Multi-step route planning {#planning}

Given a target molecule, search an AND-OR graph for a full route down to
purchasable building blocks.

```bash
synomega plan --target "CC(=O)Nc1ccccc1O" --max-steps 5 --simplify
```

```python
import synomega

planner = synomega.load_default_planner()            # default: original model + retrostar
result = planner.plan("CC(=O)Nc1ccccc1O")
print(result.solved)
print(result.best_route.describe())
```

Common knobs: `--algorithm {retrostar,mcts,bfs}`, `--expansion-width` (top-k
candidates per node), `--max-steps` (depth cap), `--time-limit`,
`--max-expansions`, `--exclude-target` (treat the target as not purchasable to
avoid a trivial zero-step solve), `--simplify` (use the
simplification-constrained model for cheaper search). See [Multi-step Route
Planning](research/planning.en.md).

## Reaction plausibility {#plausibility}

Score 0–1 how likely a set of reactants actually gives the product, to **filter**
clearly-wrong single-step disconnections. Measured to be net-negative on
single-step top-k recall, so it is **off by default** — enable it explicitly.

```python
import synomega

# attach to the planner; drops implausible candidates at every step (drop-only, no re-ranking)
planner = synomega.load_default_planner(plausibility=True, plausibility_threshold=0.4)

# or score a batch of candidate reactions directly
from synomega.plausibility import PlausibilityScorer
scorer = PlausibilityScorer.default()
scores = scorer.score_reactions(["CC(=O)O.NC>>CC(=O)NC"])   # 0–1 per reaction
```

For why it is off by default, see [Reaction
Plausibility](research/plausibility.en.md).

## Synthesizability score (SynScore) {#synscore}

Give a target a continuous **synthesizability score** SynScore = \(1/(U+1)^U\)
(`U` = number of non-purchasable starting materials in the best route; all
purchasable → 1, lower as more are missing), for ranking a set of molecules.

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

For the score definition and operating point, see [Synthesizability
Score](research/synscore.en.md).
