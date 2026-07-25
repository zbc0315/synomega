"""Mapping-free dual-tower D-MPNN reaction-plausibility classifier.

A shared directed-MPNN encoder embeds the reactant graph and the product graph
independently (no atom mapping, no condensed graph). The two graph vectors are
combined as ``[h_r, h_p, h_p - h_r, h_p * h_r]`` and an MLP head outputs one
plausibility logit. Architecture matches the checkpoint trained in the
``reaction-plausibility`` project so its ``state_dict`` loads directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from .features import ATOM_FDIM, BOND_FDIM


class DMPNNEncoder(nn.Module):
    def __init__(self, atom_fdim, bond_fdim, hidden_dim, depth, dropout, readout="sum"):
        super().__init__()
        self.depth = depth
        self.readout = readout
        self.W_input = nn.Linear(atom_fdim + bond_fdim, hidden_dim, bias=False)
        self.W_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_output = nn.Linear(atom_fdim + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch, num_graphs) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        E = edge_index.size(1)
        if E == 0:
            h_atom = F.relu(self.W_output(torch.cat(
                [x, torch.zeros(x.size(0), self.W_hidden.out_features, device=x.device)],
                dim=-1)))
            return scatter(h_atom, batch, dim=0, dim_size=num_graphs, reduce=self.readout)
        h0 = F.relu(self.W_input(torch.cat([x[src], edge_attr], dim=-1)))
        h_edge = h0
        rev = torch.arange(E, device=edge_index.device).view(-1, 2).flip(-1).reshape(-1)
        for _ in range(self.depth - 1):
            agg = scatter(h_edge, dst, dim=0, dim_size=x.size(0), reduce="sum")
            m = agg[src] - h_edge[rev]
            h_edge = F.relu(h0 + self.W_hidden(m))
            h_edge = self.dropout(h_edge)
        m_atom = scatter(h_edge, dst, dim=0, dim_size=x.size(0), reduce="sum")
        h_atom = F.relu(self.W_output(torch.cat([x, m_atom], dim=-1)))
        h_atom = self.dropout(h_atom)
        return scatter(h_atom, batch, dim=0, dim_size=num_graphs, reduce=self.readout)


class DualPlausibilityNet(nn.Module):
    def __init__(self, hidden_dim=300, depth=4, dropout=0.1,
                 head_hidden=300, head_dropout=0.2, shared=True):
        super().__init__()
        self.enc_r = DMPNNEncoder(ATOM_FDIM, BOND_FDIM, hidden_dim, depth, dropout)
        self.enc_p = self.enc_r if shared else DMPNNEncoder(
            ATOM_FDIM, BOND_FDIM, hidden_dim, depth, dropout)
        self.head = nn.Sequential(
            nn.Linear(4 * hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, rb, pb) -> torch.Tensor:
        hr = self.enc_r(rb.x, rb.edge_index, rb.edge_attr, rb.batch, rb.num_graphs)
        hp = self.enc_p(pb.x, pb.edge_index, pb.edge_attr, pb.batch, pb.num_graphs)
        feat = torch.cat([hr, hp, hp - hr, hp * hr], dim=-1)
        return self.head(feat).squeeze(-1)


def build_dual_model(config: dict) -> DualPlausibilityNet:
    m = config["model"]
    return DualPlausibilityNet(
        hidden_dim=m["hidden_dim"], depth=m["depth"], dropout=m["dropout"],
        head_hidden=m["head_hidden"], head_dropout=m["head_dropout"],
        shared=m.get("shared", True),
    )


__all__ = ["DualPlausibilityNet", "DMPNNEncoder", "build_dual_model"]
