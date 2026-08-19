"""D-MPNN forward template-classifier backend.

Reuses the same checkpoint, D-MPNN and featurizer as the retro TemplateGNN:

  1. reactant SMILES (all fragments kept) -> graph -> D-MPNN -> softmax over templates
  2. take the top-K templates
  3. reverse each retro template and apply it forward with RDKit, collecting every
     distinct, sanitized, radical-free product
  4. rank products by (template probability desc, number of producing templates desc)

A product inherits the max probability of the templates that produced it. The
reactant-center head is loaded but not used for ranking — aligning matched atoms
across reactant fragments is fiddly.

Requires the `gnn` extra: `pip install synomega[gnn]`.
"""

from __future__ import annotations

from pathlib import Path

from ..chem.features import (
    BOND_FDIM,
    atom_features,
    bond_features,
    compute_gasteiger,
)
from ..chem.template import (
    TemplateLibrary,
    apply_template_forward,
    load_template_library,
)
from .base import ForwardModel, ForwardPrediction

_TORCH_HINT = (
    "ForwardTemplateGNN needs torch + torch_geometric. "
    "Install with: pip install 'synomega[gnn]'"
)


class ForwardTemplateGNN(ForwardModel):
    """Neural forward-template classifier + RDKit forward template application."""

    name = "forward-template-gnn"

    def __init__(
        self,
        checkpoint: str | Path,
        templates: TemplateLibrary,
        *,
        device: str | None = None,
        topk_templates: int = 10,
        batch_size: int = 128,
    ):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(_TORCH_HINT) from exc
        from ..singlestep._dmpnn import load_dmpnn

        self.torch = torch
        self.templates = templates
        self.topk_templates = topk_templates
        self.batch_size = batch_size
        self.model, self.config, self.num_classes, self.device = load_dmpnn(
            checkpoint, device
        )

    # ------------------------------------------------------------- loading

    @classmethod
    def default(cls, **kwargs) -> "ForwardTemplateGNN":
        """Load the forward model, downloading it on first use.

        Convenience for ``from_pretrained(ensure_forward_model())``. Override the
        download with ``SYNOMEGA_FORWARD_MODEL=/path/to/run_dir``.
        """
        from ..data import ensure_forward_model

        return cls.from_pretrained(ensure_forward_model(), **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        run_dir: str | Path,
        *,
        checkpoint_name: str = "best.pt",
        templates_path: str | Path | None = None,
        **kwargs,
    ) -> "ForwardTemplateGNN":
        """Load from a run directory (``best.pt`` + label->retro-SMARTS map).

        The forward model shares the retro template inventory, so the same
        ``label_to_template_smarts.json`` is used (reversed at apply time).
        """
        ckpt, lib = load_template_library(
            run_dir, checkpoint_name=checkpoint_name, templates_path=templates_path
        )
        return cls(ckpt, lib, **kwargs)

    # ------------------------------------------------------------ featurize

    def _graph(self, smiles: str):
        """Reactant SMILES -> PyG Data, keeping ALL fragments (multi-reactant).

        Same featurization as the retro backend except it does NOT reduce to the
        largest fragment — a forward input is a set of reactant molecules and all
        of them must be encoded.
        """
        from rdkit import Chem
        from torch_geometric.data import Data

        torch = self.torch
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        compute_gasteiger(mol)

        x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

        ei, ea = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feat = bond_features(bond)
            ei.append([i, j]); ea.append(feat)
            ei.append([j, i]); ea.append(feat)

        if ei:
            edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(ea, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, BOND_FDIM), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # -------------------------------------------------------------- predict

    def predict(self, reactants: str, top_k: int = 10) -> list[ForwardPrediction]:
        return self.predict_batch([reactants], top_k)[0]

    def predict_batch(
        self, reactants: list[str], top_k: int = 10
    ) -> list[list[ForwardPrediction]]:
        torch = self.torch
        from torch_geometric.data import Batch

        results: list[list[ForwardPrediction]] = [[] for _ in reactants]

        graphs: list[tuple[int, object]] = []
        for i, s in enumerate(reactants):
            g = self._graph(s)
            if g is not None:
                graphs.append((i, g))
        if not graphs:
            return results

        k_tpl = min(self.topk_templates, self.num_classes)

        for start in range(0, len(graphs), self.batch_size):
            chunk = graphs[start:start + self.batch_size]
            batch = Batch.from_data_list([g for _, g in chunk]).to(self.device)

            with torch.no_grad():
                out = self.model(batch)
                tpl_logits = out[0] if isinstance(out, tuple) else out
                probs = torch.softmax(tpl_logits, dim=-1)
                top_probs, top_idx = probs.topk(k_tpl, dim=-1)
                top_probs = top_probs.cpu()
                top_idx = top_idx.cpu()

            for local, (orig_i, _) in enumerate(chunk):
                results[orig_i] = self._decode(
                    reactants[orig_i],
                    top_idx[local].tolist(),
                    top_probs[local].tolist(),
                    top_k,
                )

        return results

    def _decode(
        self,
        reactants_smiles: str,
        labels: list[int],
        probs: list[float],
        top_k: int,
    ) -> list[ForwardPrediction]:
        """Apply top templates forward and rank the resulting products."""
        # product -> [best_prob, producing_template_count, best_template_id]
        best: dict[str, list] = {}

        for label, prob in zip(labels, probs):
            smarts = self.templates.get(label)
            if smarts is None:
                continue
            for outcome in apply_template_forward(smarts, reactants_smiles):
                p = float(prob)
                entry = best.get(outcome.product)
                if entry is None:
                    best[outcome.product] = [p, 1, label]
                else:
                    if p > entry[0]:
                        entry[0] = p
                        entry[2] = label
                    entry[1] += 1

        ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
        return [
            ForwardPrediction(
                product=product,
                score=prob,
                template_id=tid,
                meta={"n_templates": count},
            )
            for product, (prob, count, tid) in ranked[:top_k]
        ]


__all__ = ["ForwardTemplateGNN"]
