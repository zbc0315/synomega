"""Forward reaction prediction: reactants -> ranked product candidates.

Quick start::

    from synomega.forward import ForwardTemplateGNN

    model = ForwardTemplateGNN.default()            # downloads on first use
    for pred in model.predict("CC(=O)O.CN", top_k=5):
        print(pred.product, pred.score)
"""

from __future__ import annotations

from .base import ForwardModel, ForwardPrediction

__all__ = [
    "ForwardModel",
    "ForwardPrediction",
    "ForwardTemplateGNN",
    "MultiComponentEvolution",
    "EvolutionResult",
    "PoolMolecule",
    "build_evolver",
]

# Names served lazily from a submodule, so `import synomega.forward` stays cheap
# (the GNN backend pulls in torch; evolution only needs it via the model passed in).
_LAZY = {
    "ForwardTemplateGNN": "template_gnn",
    "MultiComponentEvolution": "evolution",
    "EvolutionResult": "evolution",
    "PoolMolecule": "evolution",
    "build_evolver": "evolution",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is not None:
        import importlib

        mod = importlib.import_module(f".{module}", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
