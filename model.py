"""Inference-compatible BanglaBERT + GAT + GCN model.

The module/parameter names match the training architecture from the notebook:
  bb, proj, gat1, gcn1, n1, gat2, gcn2, n2, pool, fc.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from torch_geometric.nn import GATConv, GCNConv


class HybridNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        name = self.cfg.get("model_name", "csebuetnlp/banglabert")
        g_dim = int(self.cfg.get("g_dim", 256))
        heads = int(self.cfg.get("heads", 8))
        drop = float(self.cfg.get("drop", 0.1))
        if g_dim % heads:
            raise ValueError("g_dim must be divisible by heads")

        cfg_dir = Path(__file__).resolve().parent / "backbone_config"
        bb_cfg = AutoConfig.from_pretrained(str(cfg_dir) if cfg_dir.is_dir() else name)
        self.bb = AutoModel.from_config(bb_cfg)
        h = self.bb.config.hidden_size
        self.proj = nn.Linear(h, g_dim)

        self.gat1 = GATConv(g_dim, g_dim // heads, heads=heads,
                            dropout=drop, add_self_loops=False)
        self.gcn1 = GCNConv(g_dim, g_dim, add_self_loops=False)
        self.n1 = nn.LayerNorm(g_dim)

        self.gat2 = GATConv(g_dim, g_dim // heads, heads=heads,
                            dropout=drop, add_self_loops=False)
        self.gcn2 = GCNConv(g_dim, g_dim, add_self_loops=False)
        self.n2 = nn.LayerNorm(g_dim)

        self.drop = nn.Dropout(drop)
        self.pool = nn.Linear(g_dim, 1)
        self.fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(h + 3 * g_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(h, 2),
        )

    def _run(self, input_ids=None, attention_mask=None, token_type_ids=None,
             w=None, nm=None, adj=None, labels=None, details=False):
        h = self.bb(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        ).last_hidden_state

        batch_size, _, hidden = h.shape
        n = nm.shape[1]
        ids = w.unsqueeze(-1).expand(-1, -1, hidden)
        mask = (w > 0).unsqueeze(-1).to(h.dtype)
        x = h.new_zeros(batch_size, n, hidden).scatter_add(1, ids, h * mask)
        den = h.new_zeros(batch_size, n, 1).scatter_add(1, w.unsqueeze(-1), mask)
        x = x / den.clamp_min(1)
        x = torch.cat([h[:, :1], x[:, 1:]], dim=1)

        graph_x = self.proj(x).reshape(batch_size * n, -1)
        batch_id, src, dst = adj.nonzero(as_tuple=True)
        edge = torch.stack([batch_id * n + src, batch_id * n + dst])

        if details:
            out1, (edge1, alpha1) = self.gat1(
                graph_x, edge, return_attention_weights=True
            )
        else:
            out1 = self.gat1(graph_x, edge)
        gat_out1 = self.drop(F.elu(out1))
        gcn_out1 = self.drop(F.elu(self.gcn1(graph_x, edge)))
        graph_x = self.n1(graph_x + gat_out1 + gcn_out1)

        if details:
            out2, (edge2, alpha2) = self.gat2(
                graph_x, edge, return_attention_weights=True
            )
        else:
            out2 = self.gat2(graph_x, edge)
        gat_out2 = self.drop(F.elu(out2))
        gcn_out2 = self.drop(F.elu(self.gcn2(graph_x, edge)))
        graph_x = self.n2(graph_x + gat_out2 + gcn_out2).reshape(batch_size, n, -1)

        word_mask = nm.clone()
        word_mask[:, 0] = False
        pool_score = self.pool(graph_x).squeeze(-1)
        pool_score = pool_score.masked_fill(~word_mask, -1e4)
        pool_prob = pool_score.softmax(dim=1) * word_mask
        pool_prob = pool_prob / pool_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
        v_attn = (pool_prob.unsqueeze(-1) * graph_x).sum(dim=1)
        v_max = graph_x.masked_fill(~word_mask.unsqueeze(-1), -1e4).max(dim=1).values
        features = torch.cat([h[:, 0], graph_x[:, 0], v_attn, v_max], dim=1)
        logits = self.fc(features)

        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(logits, labels, label_smoothing=0.05)
        if not details:
            return result

        extra = {
            "alpha1": alpha1,
            "edge1": edge1,
            "alpha2": alpha2,
            "edge2": edge2,
            "branches": (gat_out1, gcn_out1, gat_out2, gcn_out2),
            "num_nodes": n,
            "batch_size": batch_size,
        }
        return result, extra

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                w=None, nm=None, adj=None, labels=None):
        return self._run(input_ids, attention_mask, token_type_ids,
                         w, nm, adj, labels, details=False)

    def forward_with_details(self, input_ids=None, attention_mask=None,
                             token_type_ids=None, w=None, nm=None, adj=None):
        return self._run(input_ids, attention_mask, token_type_ids,
                         w, nm, adj, labels=None, details=True)
