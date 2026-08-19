"""D-MPNN template-classifier backend.

Pipeline (this is the `evaluate_reactants_v3.py` ranking, packaged):

  1. product SMILES -> graph -> D-MPNN -> softmax over templates (+ per-atom
     reaction-center logits when the checkpoint has a center head)
  2. take the top-K templates
  3. apply each with RDKit, keeping EVERY distinct reactant set (so top-K
     coverage is not capped by picking one match per template)
  4. rank by (template probability desc, center score desc)

The center score only reorders candidates that share a template probability —
i.e. different substructure matches of the same template — so it lifts top-1
without costing top-K coverage.

Requires the `gnn` extra: `pip install synomega[gnn]`.
"""

from __future__ import annotations

from pathlib import Path

from ..chem.features import (
    BOND_FDIM,
    atom_features,
    bond_features,
    compute_gasteiger,
    largest_fragment,
)
from ..chem.template import TemplateLibrary, apply_template, load_template_library
from .base import Prediction, SingleStepModel

_TORCH_HINT = (
    "TemplateGNN needs torch + torch_geometric. Install with: pip install 'synomega[gnn]'"
)


class TemplateGNN(SingleStepModel):
    """Neural template classifier + RDKit template application."""

    name = "template-gnn"

    def __init__(
        self,
        checkpoint: str | Path,
        templates: TemplateLibrary,
        *,
        device: str | None = None,
        topk_templates: int = 50,
        use_center: bool = True,
        batch_size: int = 128,
    ):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(_TORCH_HINT) from exc
        from ._dmpnn import load_dmpnn

        self.torch = torch
        self.templates = templates
        self.topk_templates = topk_templates
        self.batch_size = batch_size

        # Shared checkpoint loader (also used by ForwardTemplateGNN); includes
        # the atom+bond feature-dimension drift guard.
        self.model, self.config, self.num_classes, self.device = load_dmpnn(
            checkpoint, device
        )
        self.use_center = bool(
            use_center and getattr(self.model, "predict_center", False)
        )

    # ------------------------------------------------------------- loading

    @classmethod
    def default(cls, **kwargs) -> "TemplateGNN":
        """Load the default pretrained model, downloading it on first use.

        Convenience for ``from_pretrained(ensure_default_model())`` — see
        :mod:`synomega.data`. Extra keyword args (e.g. ``device``,
        ``topk_templates``) are forwarded.
        """
        from ..data import ensure_default_model

        return cls.from_pretrained(ensure_default_model(), **kwargs)

    @classmethod
    def simplify(cls, **kwargs) -> "TemplateGNN":
        """Load the simplification-constrained model, downloading it on first use.

        This model proposes only *simplifying* (fragmentation) disconnections --
        those that split the target into two or more precursors. In multi-step
        search it reaches purchasable material with fewer node expansions at
        matched solvability (see the accompanying paper). Convenience for
        ``from_pretrained(ensure_simplify_model())``; extra keyword args (e.g.
        ``device``) are forwarded. Override the download with
        ``SYNOMEGA_SIMPLIFY_MODEL=/path/to/run_dir``.
        """
        from ..data import ensure_simplify_model

        return cls.from_pretrained(ensure_simplify_model(), **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        run_dir: str | Path,
        *,
        checkpoint_name: str = "best.pt",
        templates_path: str | Path | None = None,
        **kwargs,
    ) -> "TemplateGNN":
        """Load from an ml-template-gnn run directory.

        Looks for `<run_dir>/best.pt` and a template map. The template map is
        searched for at `templates_path`, then `<run_dir>/label_to_template_smarts.json`,
        then the processed-data dir recorded in the checkpoint's config.
        """
        ckpt, lib = load_template_library(
            run_dir, checkpoint_name=checkpoint_name, templates_path=templates_path
        )
        return cls(ckpt, lib, **kwargs)

    # ------------------------------------------------------------ featurize

    def _graph(self, smiles: str):
        """SMILES -> PyG Data, or None if unusable."""
        from rdkit import Chem
        from torch_geometric.data import Data

        torch = self.torch
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        mol = largest_fragment(mol)
        if mol.GetNumAtoms() == 0:
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

    def predict(self, smiles: str, top_k: int = 50) -> list[Prediction]:
        return self.predict_batch([smiles], top_k)[0]

    def predict_batch(
        self, smiles: list[str], top_k: int = 50
    ) -> list[list[Prediction]]:
        torch = self.torch
        from torch_geometric.data import Batch

        results: list[list[Prediction]] = [[] for _ in smiles]

        # Featurize once; unparseable inputs simply get no predictions.
        graphs: list[tuple[int, object]] = []
        for i, s in enumerate(smiles):
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
                if isinstance(out, tuple):
                    tpl_logits, center_logits = out
                else:
                    tpl_logits, center_logits = out, None
                probs = torch.softmax(tpl_logits, dim=-1)
                top_probs, top_idx = probs.topk(k_tpl, dim=-1)
                top_probs = top_probs.cpu()
                top_idx = top_idx.cpu()
                center = (
                    torch.sigmoid(center_logits).cpu()
                    if (center_logits is not None and self.use_center)
                    else None
                )
                node_batch = batch.batch.cpu()

            for local, (orig_i, _) in enumerate(chunk):
                center_probs = (
                    center[node_batch == local].tolist() if center is not None else None
                )
                results[orig_i] = self._rank(
                    smiles[orig_i],
                    top_idx[local].tolist(),
                    top_probs[local].tolist(),
                    center_probs,
                    top_k,
                )

        return results

    def _rank(
        self,
        product_smiles: str,
        labels: list[int],
        probs: list[float],
        center_probs: list[float] | None,
        top_k: int,
    ) -> list[Prediction]:
        """Apply top templates and rank by (template prob, center score)."""
        # reactants -> (template_prob, center_avg, template_id)
        best: dict[tuple[str, ...], tuple[float, float, int]] = {}

        for label, prob in zip(labels, probs):
            smarts = self.templates.get(label)
            if smarts is None:
                continue
            for outcome in apply_template(smarts, product_smiles):
                center_avg = 0.0
                if center_probs and outcome.match_atoms:
                    hits = [
                        center_probs[i]
                        for i in outcome.match_atoms
                        if i < len(center_probs)
                    ]
                    if hits:
                        center_avg = sum(hits) / len(hits)
                cand = (float(prob), center_avg, label)
                prev = best.get(outcome.reactants)
                if prev is None or cand > prev:
                    best[outcome.reactants] = cand

        ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
        return [
            Prediction(
                reactants=reactants,
                score=prob,
                template_id=tid,
                meta={"center_avg": c_avg},
            )
            for reactants, (prob, c_avg, tid) in ranked[:top_k]
        ]


__all__ = ["TemplateGNN"]
