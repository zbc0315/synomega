"""D-MPNN inference model (vendored from ml-template-gnn).

Mirrors `ml-template-gnn/src/template_gnn/model.py` so a checkpoint can be
loaded without the training repo on the path. Inference only — no DDP, no
training-time plumbing.

Imports torch lazily via the module-level import guard in `template_gnn.py`;
this module is only imported once torch is known to be present.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from ..chem.features import ATOM_FDIM, BOND_FDIM


class DMPNN(nn.Module):
    """Directed message-passing network over bonds (Yang et al. 2019)."""

    def __init__(
        self,
        num_classes: int,
        atom_fdim: int = ATOM_FDIM,
        bond_fdim: int = BOND_FDIM,
        hidden_dim: int = 300,
        depth: int = 3,
        dropout: float = 0.1,
        head_hidden: int = 600,
        head_dropout: float = 0.2,
        readout: str = "sum",
        predict_center: bool = False,
        center_head_hidden: int = 128,
    ):
        super().__init__()
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.readout = readout
        self.predict_center = predict_center

        self.W_input = nn.Linear(atom_fdim + bond_fdim, hidden_dim, bias=False)
        self.W_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_output = nn.Linear(atom_fdim + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_classes),
        )

        if predict_center:
            self.center_head = nn.Sequential(
                nn.Linear(hidden_dim, center_head_hidden),
                nn.ReLU(),
                nn.Linear(center_head_hidden, 1),
            )
        else:
            self.center_head = None

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch,
        )
        src, dst = edge_index[0], edge_index[1]
        E = edge_index.size(1)

        h0 = F.relu(self.W_input(torch.cat([x[src], edge_attr], dim=-1)))
        h_edge = h0

        # Bonds were emitted as (i,j) then (j,i), so edge 2k's reverse is 2k+1.
        rev_index = torch.arange(E, device=edge_index.device).view(-1, 2).flip(-1).reshape(-1)

        for _ in range(self.depth - 1):
            agg_in = scatter(h_edge, dst, dim=0, dim_size=x.size(0), reduce="sum")
            m = agg_in[src] - h_edge[rev_index]
            h_edge = F.relu(h0 + self.W_hidden(m))
            h_edge = self.dropout(h_edge)

        m_atom = scatter(h_edge, dst, dim=0, dim_size=x.size(0), reduce="sum")
        h_atom = F.relu(self.W_output(torch.cat([x, m_atom], dim=-1)))
        h_atom = self.dropout(h_atom)

        h_graph = scatter(h_atom, batch, dim=0, reduce=self.readout)
        template_logits = self.head(h_graph)

        if self.center_head is None:
            return template_logits
        center_logits = self.center_head(h_atom).squeeze(-1)
        return template_logits, center_logits


def build_from_config(config: dict, num_classes: int) -> DMPNN:
    m = config["model"]
    return DMPNN(
        num_classes=num_classes,
        atom_fdim=ATOM_FDIM,
        bond_fdim=BOND_FDIM,
        hidden_dim=m["hidden_dim"],
        depth=m["depth"],
        dropout=m["dropout"],
        head_hidden=m["head_hidden"],
        head_dropout=m["head_dropout"],
        readout=m["readout"],
        predict_center=bool(m.get("predict_center", False)),
        center_head_hidden=int(m.get("center_head_hidden", 128)),
    )


def load_dmpnn(checkpoint, device=None):
    """Load a D-MPNN checkpoint into an eval-ready model.

    Shared by the retro :class:`~synomega.singlestep.TemplateGNN` and the forward
    :class:`~synomega.forward.ForwardTemplateGNN`: both consume the same
    ml-template-gnn checkpoint format (``state["config" / "num_classes" / "model"]``).

    Returns ``(model, config, num_classes, device)`` with the model already moved
    to ``device`` and put in eval mode.

    Raises ``RuntimeError`` when the checkpoint's input width disagrees with the
    vendored featurizer — this checks the atom+bond width only, not fragment
    handling.
    """
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    config = state["config"]
    num_classes = int(state["num_classes"])

    expected = state["model"]["W_input.weight"].shape[1]
    if expected != ATOM_FDIM + BOND_FDIM:
        raise RuntimeError(
            f"feature dimension mismatch: checkpoint expects {expected} "
            f"(atom+bond) but synomega.chem.features gives {ATOM_FDIM + BOND_FDIM}. "
            f"The vendored featurizer is out of sync with this checkpoint."
        )

    dev = torch.device(
        device if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = build_from_config(config, num_classes)
    model.load_state_dict(state["model"])
    model = model.to(dev).eval()
    return model, config, num_classes, dev


__all__ = ["DMPNN", "build_from_config", "load_dmpnn"]
