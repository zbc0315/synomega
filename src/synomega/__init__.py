"""synomega — retrosynthesis toolkit.

Three layers, deliberately decoupled:

    synthesizability   is this target reachable from purchasable material?
        ↑
    search             multi-step route planning over an AND-OR graph
        ↑
    singlestep         product SMILES -> ranked reactant candidates

Quick start::

    from synomega import Planner, SynthesizabilityScorer
    from synomega.singlestep import TemplateGNN
    from synomega.stock import InMemoryStock

    model   = TemplateGNN.from_pretrained("runs/uspto50k_r0_min10")
    stock   = InMemoryStock.from_file("emolecules.smi")
    planner = Planner(model, stock, algorithm="retrostar")

    result = planner.plan("CC(=O)Nc1ccccc1", max_depth=5)
    print(result.best_route.describe())

    scorer = SynthesizabilityScorer(planner)
    print(scorer.score("CC(=O)Nc1ccccc1", max_steps=5))
"""

from __future__ import annotations

__version__ = "0.7.0"

from .chem import Molecule, Reaction
from .planner import Planner
from .route import Route
from .search import Budget, SearchResult
from .singlestep import Prediction, SingleStepModel
from .stock import BuildingBlockSet, InMemoryStock
from .synthesizability import BatchReport, MoleculeReport, SynthesizabilityScorer

def load_default_planner(
    *, algorithm: str = "retrostar", device: str = "cpu",
    simplify: bool = False,
    plausibility: bool = False, plausibility_threshold: float = 0.4,
    **planner_kwargs,
) -> "Planner":
    """A ready-to-use planner backed by the default model + stock.

    Downloads the default pretrained model and building-block stock on first use
    (see :mod:`synomega.data`), so ``pip install synomega`` works out of the box::

        import synomega
        planner = synomega.load_default_planner()
        print(planner.plan("CC(=O)Nc1ccccc1O").best_route.describe())

    The first call fetches a few hundred MB into ``~/.cache/synomega``.

    Pass ``simplify=True`` to back the planner with the simplification-constrained
    single-step model (fragmentation-only disconnections), which reaches purchasable
    material with fewer node expansions at matched solvability; it is downloaded on
    first use like the default model.

    Reaction-plausibility screening of single-step predictions is **off by
    default** (benchmarks show it does not improve top-k retrieval of the recorded
    reaction and adds latency). Pass ``plausibility=True`` to enable it, optionally
    with a ``plausibility_threshold``.
    """
    from .singlestep import TemplateGNN
    from .stock import InMemoryStock

    model = (TemplateGNN.simplify(device=device) if simplify
             else TemplateGNN.default(device=device))
    stock = InMemoryStock.default()
    scorer = None
    if plausibility:
        from .plausibility import PlausibilityScorer

        scorer = PlausibilityScorer.default(device=device)
    return Planner(model, stock, algorithm=algorithm, plausibility=scorer,
                   plausibility_threshold=plausibility_threshold, **planner_kwargs)


def load_default_scorer(
    *, algorithm: str = "retrostar", device: str = "cpu",
    simplify: bool = True, expansion_width: int = 10,
    plausibility: bool = False, plausibility_threshold: float = 0.4,
    **planner_kwargs,
) -> "SynthesizabilityScorer":
    """A ready-to-use synthesizability scorer backed by the default model + stock.

    Downloads the model and building-block stock on first use, so scoring works
    out of the box::

        import synomega
        scorer = synomega.load_default_scorer()
        print(scorer.score("CC(=O)Nc1ccccc1O").as_dict())

    Unlike :func:`load_default_planner`, this defaults to the
    simplification-constrained (``breaking``) single-step model
    (``simplify=True``): restricting single-step predictions to simplifying
    (fragmentation) disconnections reaches purchasable building blocks with fewer
    node expansions at matched solvability, which makes it the recommended model
    for route-based synthesizability scoring. Pass ``simplify=False`` to score with
    the unconstrained (``original``) model instead.

    Scoring also defaults to an expansion width of ``k=10`` -- the operating point
    at which the breaking model is near-converged yet inexpensive (see the
    accompanying paper), rather than the planner's wider default of 50.
    """
    planner = load_default_planner(
        algorithm=algorithm, device=device, simplify=simplify,
        expansion_width=expansion_width,
        plausibility=plausibility, plausibility_threshold=plausibility_threshold,
        **planner_kwargs,
    )
    return SynthesizabilityScorer(planner)


__all__ = [
    "__version__",
    "Planner",
    "load_default_planner",
    "load_default_scorer",
    "SynthesizabilityScorer",
    "MoleculeReport",
    "BatchReport",
    "Route",
    "SearchResult",
    "Budget",
    "Molecule",
    "Reaction",
    "Prediction",
    "SingleStepModel",
    "BuildingBlockSet",
    "InMemoryStock",
]
