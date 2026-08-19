# Research Report · Overview and Evaluation Protocol

This chapter is where we define the **training corpus, evaluation sets, evaluation
protocol, metrics, and terminology** in one place. The subsequent functional chapters
([Forward](forward.md) / [Retro](retro.md) / [Multi-step planning](planning.md) /
[Plausibility](plausibility.md) / [SynScore](synscore.md)) build on this foundation and
describe only their own models, pseudocode, and results, without redefining conventions.

## 1. Training corpus and reaction templates

All single-step models are trained on a single **large-scale atom-mapped commercial
reaction corpus**. The raw reactions are not distributed with the software; only the
trained weights are released publicly.

Reaction templates are extracted at **radius 0** (RDChiral), keeping templates that
occur **at least 20 times** in the corpus, yielding **64,366 template classes** that
serve as the classification label space of the single-step models. This template library
is shared across retrosynthesis, forward prediction, and plausibility negative-sample
generation.

The template frequency follows an extreme long tail:

| Statistic | Reactions per template |
|---|---|
| Min / median / mean / max | 20 / 37 / 245.6 / 473,296 |
| Templates needed to cover 50% of reactions | 185 (0.3%) |
| Templates needed to cover 90% of reactions | 17,415 (27.1%) |

Of these, **42,028 (65.3%)** templates are "simplifying" (writing the retro direction,
the product is a single molecule and the reactants are two or more molecules); this
subset is the action space of the simplification-constrained model in the
[Retro chapter](retro.md).

!!! note "Data-split conventions"
    Different models use different train/val/test splits (for example, the forward model
    splits by reaction id `%20`, whereas the original retrosynthesis model uses another
    stratified split based on the minimum per-template sample count). **Each model's split
    figures are stated in its own chapter with the source noted**; this overview
    deliberately avoids committing to a single number so that conventions do not drift.

## 2. Evaluation sets

| Evaluation set | Purpose | Convention |
|---|---|---|
| **ZINC purchasable building-block set** | In-stock determination + sample for structure-based baseline scoring | Purchasability by InChIKey; baseline scoring randomly samples **20,000** molecules |
| **1000 ChEMBL target set** | Unified evaluation targets for multi-step planning, SynScore, and plausibility filtering | ChEMBL 35, small molecule, heavy atoms **5–60**; sampled with a fixed random seed, desalted to the largest organic component, deduplicated by InChIKey; 1,017 sampled, 17 over the limit, **1,000** retained |

The 1000 ChEMBL target set is independent of the training corpus (out-of-distribution
evaluation). Its physicochemical descriptor distribution (median [5th, 95th]):

| Descriptor | Median [5th, 95th] |
|---|---|
| Heavy atom count | 27 [16, 43] |
| Molecular weight (Da) | 388 [240, 616] |
| Ring count / aromatic ring count | 3 [1, 6] / 2 [0, 4] |
| Rotatable bonds | 5 [1, 12] |
| H-bond donors / acceptors | 1 [0, 4] / 5 [2, 9] |
| TPSA (Å²) | 76 [28, 156] |
| Fraction sp³ carbon | 0.31 [0.05, 0.73] |

## 3. Multi-step search budget (unified operating point)

Multi-step planning and SynScore evaluations are conducted under a **budget-aligned**
setting:

- Expansion width **k = 10** (each molecule node takes the single-step model's top-10 candidate reactant sets)
- Depth **≤ 5**, node expansions **≤ 100**, per-molecule search time limit **8 s**, hard cap **60 s**

## 4. Metric definitions

| Metric | Definition |
|---|---|
| **template top-k** | Fraction of cases where the true template is among the single-step model's top-k predicted templates |
| **product top-1** (forward) | Fraction of cases where the forward top-1 product canonical SMILES matches the true product |
| **solved rate** | Fraction of targets for which a route with all leaves purchasable (U=0) is found within depth ≤5 |
| **U** | Number of **non-purchasable starting materials** in the best route (smaller is better) |
| **SynScore** | \( \mathrm{SynScore} = 1/(U+1)^{U} \), see [SynScore chapter](synscore.md) |
| **bb_coverage** | Fraction of purchasable leaves in the best route (continuous coverage, \([0,1]\)) |
| **mean/median expansions** | Mean / median node expansions per target (search cost, smaller is better) |
| **score fidelity r** | Pearson correlation between SynScore under a given setting and the reference setting (original model @ k=50) |
| **val AUC** (plausibility) | ROC AUC of the binary reaction-plausibility classifier on the validation set |

## 5. Terminology and writing conventions

- On first appearance a term is given in Chinese with the English in parentheses; class
  names, CLI subcommands, and hyperparameter names (such as `TemplateGNN`, `predict`,
  `hidden_dim`) are always kept in their original English form.
- **Retrosynthesis templates** are written as `product >> reactants` (RDChiral style);
  when applied in the forward direction they are reversed to `reactants >> product`.
- Molecular identity: **InChIKey** throughout the pipeline.

| Abbreviation | Meaning |
|---|---|
| D-MPNN | Directed message-passing neural network |
| AND-OR graph | Search graph with molecule nodes as OR and reaction nodes as AND |
| Retro\* | The retrieval-style AND-OR search algorithm of Chen et al. (ICML 2020) |
| CGR | Condensed Graph of Reaction |
| bb | building block, a purchasable building block |

## 6. Chapter at a glance

| Chapter | Core | Key results (details in each chapter) |
|---|---|---|
| [Single-step forward prediction](forward.md) | D-MPNN template-classification mirror + RDKit forward application | Validation template top-1 **0.759**, product top-1 **0.636** |
| [Single-step retro prediction](retro.md) | D-MPNN template classifier (+ simplification-constrained variant) | Original model test top-1 **0.403** / top-10 **0.742**; simplification model held-out top-1 **0.575** |
| [Multi-step route planning](planning.md) | Retro\* (default) / MCTS / best-first | 1000 ChEMBL solved **85.1%** (simplification) / **81.8%** (original), about **1.8×** that of AiZynthFinder |
| [Reaction plausibility scoring](plausibility.md) | Two-tower D-MPNN (no mapping) | val AUC **0.9946**; single-step filtering has a net negative benefit → off by default |
| [Synthesizability scoring](synscore.md) | Based on the number of non-purchasable starting materials in the best route | SynScore \(=1/(U+1)^U\), continuous and rankable |
