"""Load the dual-tower plausibility model and score reactions.

``PlausibilityScorer.score_reactions([(reactants, product), ...])`` returns a
plausibility probability in ``[0, 1]`` for each reaction — how likely the given
reactants actually produce that product. Reactant/product graphs are cached, so
scoring many candidate disconnections of the same target is cheap.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .features import mol_to_graph
from .model import build_dual_model


class PlausibilityScorer:
    def __init__(self, checkpoint: str | Path, device: str = "cpu",
                 batch_size: int = 256):
        import torch

        self._torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        ck = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model = build_dual_model(ck["config"]).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.meta = {"epoch": ck.get("epoch"),
                     "val_auc": ck.get("metrics", {}).get("auc")}

    @classmethod
    def default(cls, *, device: str = "cpu", **kwargs) -> "PlausibilityScorer":
        """Download (first use) and load the default dual-tower plausibility model."""
        from ..data import ensure_default_plausibility_model

        return cls(ensure_default_plausibility_model(), device=device, **kwargs)

    @lru_cache(maxsize=200_000)
    def _graph(self, smiles: str):
        return mol_to_graph(smiles)

    def score_reactions(self, reactions):
        """reactions: iterable of (reactants_smiles, product_smiles) -> list[float].

        A reaction whose reactants or product cannot be parsed scores 0.0.
        """
        import torch
        from torch_geometric.data import Batch

        reactions = list(reactions)
        scores = [0.0] * len(reactions)
        rgs, pgs, idx = [], [], []
        for i, (r, p) in enumerate(reactions):
            rg = self._graph(r)
            pg = self._graph(p)
            if rg is not None and pg is not None:
                rgs.append(rg); pgs.append(pg); idx.append(i)
        if not idx:
            return scores
        with torch.no_grad():
            for s in range(0, len(idx), self.batch_size):
                sl = slice(s, s + self.batch_size)
                rb = Batch.from_data_list(rgs[sl]).to(self.device)
                pb = Batch.from_data_list(pgs[sl]).to(self.device)
                probs = torch.sigmoid(self.model(rb, pb)).cpu().tolist()
                if isinstance(probs, float):   # single-item batch
                    probs = [probs]
                for k, prob in enumerate(probs):
                    scores[idx[s + k]] = float(prob)
        return scores

    def __repr__(self) -> str:
        return (f"<PlausibilityScorer dual-tower device={self.device} "
                f"val_auc={self.meta.get('val_auc')}>")


__all__ = ["PlausibilityScorer"]
