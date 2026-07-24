"""Single-step retrosynthesis backends.

`TemplateGNN` is imported lazily so the package stays usable without torch.
"""

from __future__ import annotations

from .base import Prediction, SingleStepModel
from .cache import CachedModel
from .template_rule import TemplateRuleModel

__all__ = [
    "Prediction",
    "SingleStepModel",
    "CachedModel",
    "TemplateRuleModel",
    "TemplateGNN",
]


def __getattr__(name: str):
    # Deferred so `import synomega` does not pull in torch.
    if name == "TemplateGNN":
        from .template_gnn import TemplateGNN

        return TemplateGNN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
