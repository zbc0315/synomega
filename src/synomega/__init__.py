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

__version__ = "0.1.0"

from .chem import Molecule, Reaction
from .planner import Planner
from .route import Route
from .search import Budget, SearchResult
from .singlestep import Prediction, SingleStepModel
from .stock import BuildingBlockSet, InMemoryStock
from .synthesizability import BatchReport, MoleculeReport, SynthesizabilityScorer

__all__ = [
    "__version__",
    "Planner",
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
