"""Pure template-matching backend — no neural network, no torch.

Applies every template in a library to the product and ranks the resulting
reactant sets by template popularity (the `reaction_count` each template was
extracted with). Useful as:

  * a dependency-free baseline (`pip install synomega` with no extras),
  * a correctness reference for the neural backends,
  * a fallback when no checkpoint is available.

It is O(len(library)) substructure matches per call, so keep libraries small
(hundreds, not tens of thousands) or pre-filter them.
"""

from __future__ import annotations

from ..chem.template import TemplateLibrary, apply_template
from .base import Prediction, SingleStepModel


class TemplateRuleModel(SingleStepModel):
    """Rank candidates by template prior, not by a learned model."""

    name = "template-rule"

    def __init__(
        self,
        library: TemplateLibrary,
        priors: dict[int, float] | None = None,
        *,
        max_templates: int | None = None,
    ):
        """
        Args:
            library: label -> retro SMARTS.
            priors: label -> prior weight (e.g. training-set reaction_count).
                Missing labels get weight 1.0. Weights are normalized to sum 1
                so scores are comparable with probability-like backends.
            max_templates: cap on templates tried per call, highest prior first.
        """
        self.library = library
        raw = {label: float(priors.get(label, 1.0)) if priors else 1.0
               for label in library.templates}
        total = sum(raw.values()) or 1.0
        self.priors = {k: v / total for k, v in raw.items()}
        # Try high-prior templates first so max_templates keeps the useful ones.
        self._order = sorted(self.priors, key=lambda k: -self.priors[k])
        if max_templates is not None:
            self._order = self._order[:max_templates]

    def predict(self, smiles: str, top_k: int = 50) -> list[Prediction]:
        best: dict[tuple[str, ...], Prediction] = {}
        for label in self._order:
            smarts = self.library.get(label)
            if smarts is None:
                continue
            score = self.priors[label]
            for outcome in apply_template(smarts, smiles):
                prev = best.get(outcome.reactants)
                if prev is None or score > prev.score:
                    best[outcome.reactants] = Prediction(
                        reactants=outcome.reactants,
                        score=score,
                        template_id=label,
                    )
        ranked = sorted(best.values(), key=lambda p: -p.score)
        return ranked[:top_k]


__all__ = ["TemplateRuleModel"]
