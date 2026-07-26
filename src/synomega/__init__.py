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

__version__ = "0.4.1"

from .chem import Molecule, Reaction
from .planner import Planner
from .route import Route
from .search import Budget, SearchResult
from .singlestep import Prediction, SingleStepModel
from .stock import BuildingBlockSet, InMemoryStock
from .synthesizability import BatchReport, MoleculeReport, SynthesizabilityScorer

def load_default_planner(
    *, algorithm: str = "retrostar", device: str = "cpu",
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

    Reaction-plausibility screening of single-step predictions is **off by
    default** (benchmarks show it does not improve top-k retrieval of the recorded
    reaction and adds latency). Pass ``plausibility=True`` to enable it, optionally
    with a ``plausibility_threshold``.
    """
    from .singlestep import TemplateGNN
    from .stock import InMemoryStock

    model = TemplateGNN.default(device=device)
    stock = InMemoryStock.default()
    scorer = None
    if plausibility:
        from .plausibility import PlausibilityScorer

        scorer = PlausibilityScorer.default(device=device)
    return Planner(model, stock, algorithm=algorithm, plausibility=scorer,
                   plausibility_threshold=plausibility_threshold, **planner_kwargs)


__all__ = [
    "__version__",
    "Planner",
    "load_default_planner",
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
