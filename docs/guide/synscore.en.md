# Synthesizability score (SynScore)

**What it does**: give a target molecule a continuous **synthesizability score**
(SynScore) for ranking a set of molecules by how easy they are to make. It runs one
[route planning](planning.md) search internally, then folds the outcome into a
score. Definition and operating point: [Research · Synthesizability
Score](../research/synscore.en.md); install: [Install & Overview](index.en.md).

$$\mathrm{SynScore} = \frac{1}{(U+1)^{U}}$$

`U` = the number of **non-purchasable starting materials** in the best route: all
purchasable (U=0) → 1.0, U=1 → 0.5, U=2 → 0.11, U=3 → 0.016, no route at all → 0.
It falls off sharply, so it cleanly separates "fully solved / a few blocks missing /
many missing".

## Command line

```bash
# defaults to the simplification-constrained model @ expansion width k=10 (recommended);
# --original switches to the unconstrained model
synomega score --targets targets.smi --out scores.jsonl
```

## Python

```python
import synomega

scorer = synomega.load_default_scorer()              # default simplify=True, k=10
r = scorer.score("CC(=O)Nc1ccccc1O", max_steps=5)
print(r.as_dict())
# {'smiles': ..., 'solved': True, 'score': 1.0, 'bb_coverage': 1.0,
#  'min_steps': 2, 'min_route_depth': 2, 'num_leaves': 2,
#  'num_purchasable_leaves': 2, 'expansions': ..., 'terminated_by': 'solved', ...}

# batch: one SMILES per line
report = scorer.score_batch(open("targets.smi").read().split())
print(report.solve_rate, report.mean_bb_coverage)
print(report.describe())
df = report.to_dataframe()                            # one row per molecule (needs pandas)
```

A single target returns a `MoleculeReport` (`.score` / `.solved` / `.bb_coverage` /
`.min_steps` / `.num_unpurchasable_leaves` (= U) / `.as_dict()`); a batch returns a
`BatchReport` (`.solve_rate` / `.mean_bb_coverage` / `.describe()` / `.to_dataframe()`
/ `.to_json()`).

## Reading the score

| Case | U | SynScore |
|---|---|---|
| all starting materials purchasable (solved) | 0 | 1.0 |
| one block short | 1 | 0.5 |
| two short | 2 | 0.11 |
| no route at all | — | 0 |

`solved` is the binary "is there an all-purchasable route within the depth", for
comparing to published `solve_rate`; `SynScore` is the continuous, near-miss-aware,
rankable version for ordering a set of molecules.

## Parameters

| Parameter (CLI / Python) | Default | Meaning |
|---|---|---|
| `--targets` | required | one SMILES per line |
| `--original` / `simplify=` | simplify model | score with the unconstrained single-step model instead |
| `--max-steps` / `max_steps=` | 5 | route depth cap |
| `--exclude-target` | off | treat the target as not purchasable (avoid a buyable target scoring 1.0 in zero steps) |
