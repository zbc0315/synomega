"""Forward reaction prediction: reactants -> ranked product candidates.

Quick start::

    from synomega.forward import ForwardTemplateGNN

    model = ForwardTemplateGNN.default()            # downloads on first use
    for pred in model.predict("CC(=O)O.CN", top_k=5):
        print(pred.product, pred.score)
"""

from __future__ import annotations

from .base import ForwardModel, ForwardPrediction

__all__ = ["ForwardModel", "ForwardPrediction", "ForwardTemplateGNN"]


def __getattr__(name: str):
    # Deferred so `import synomega` does not pull in torch.
    if name == "ForwardTemplateGNN":
        from .template_gnn import ForwardTemplateGNN

        return ForwardTemplateGNN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
