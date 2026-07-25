"""Wrap a single-step model so every prediction is screened by the dual-tower
plausibility model.

For each candidate disconnection ``reactants -> target``, the plausibility model
scores how likely those reactants really give the target. Candidates below
``threshold`` are **dropped**; everything else is left exactly as the single-step
model ranked it — the filter only removes wrong reactions, it never re-orders the
survivors. To avoid dead-ending search when nothing clears the bar, the single-
step model's own top ``min_keep`` candidates are retained. The raw plausibility is
recorded in ``prediction.meta["plausibility"]`` and each prediction's ``score`` is
left untouched.

Because search and synthesizability both expand nodes through the single-step
model, wrapping it here screens *every* single-step prediction in the system.
"""

from __future__ import annotations

from dataclasses import replace

from ..singlestep.base import Prediction, SingleStepModel


class PlausibilityFilteredModel(SingleStepModel):
    def __init__(self, base: SingleStepModel, scorer, *, threshold: float = 0.4,
                 min_keep: int = 1, overfetch: int = 2):
        self.base = base
        self.scorer = scorer
        self.threshold = threshold
        self.min_keep = min_keep
        self.overfetch = max(1, overfetch)
        self.name = f"{base.name}+plausibility"

    def predict(self, smiles: str, top_k: int = 50) -> list[Prediction]:
        return self.predict_batch([smiles], top_k)[0]

    def predict_batch(self, smiles, top_k: int = 50):
        # Over-fetch so filtering still leaves ~top_k plausible candidates.
        base = self.base.predict_batch(smiles, top_k * self.overfetch)

        # Flatten every (candidate, target) into ONE plausibility batch.
        reactions, spans = [], []
        for prod, preds in zip(smiles, base):
            start = len(reactions)
            for p in preds:
                reactions.append((p.smiles, prod))
            spans.append((start, len(reactions)))
        scores = self.scorer.score_reactions(reactions)

        out = []
        for preds, (a, b) in zip(base, spans):
            # Annotate each candidate with its plausibility, keeping the single-
            # step model's original order (best-first) untouched.
            scored = [
                (replace(p, meta={**p.meta, "plausibility": s}), s)
                for p, s in zip(preds, scores[a:b])
            ]
            kept = [p for p, s in scored if s >= self.threshold]
            if not kept and scored:
                # Nothing cleared the bar — keep the single-step model's own best
                # so search does not dead-end (still no re-ordering).
                kept = [p for p, _ in scored[:self.min_keep]]
            out.append(kept[:top_k])
        return out

    def __getattr__(self, item):
        # Transparently expose base-model attributes (e.g. hit_rate, device).
        if item in ("base", "scorer"):
            raise AttributeError(item)
        return getattr(self.__dict__["base"], item)

    def __repr__(self) -> str:
        return (f"<PlausibilityFilteredModel base={self.base.name} "
                f"threshold={self.threshold}>")


__all__ = ["PlausibilityFilteredModel"]
