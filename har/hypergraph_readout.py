"""
hypergraph_readout.py - hypergraph readout (fixed version)
修复了 HierarchicalHypergraphReadout 的索引映射问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


class HypergraphReadout(nn.Module):
    """
    Hypergraph readout module
    c = Σ β_v · z_v   (paper eq.)
    支持 attention, max, mean, gated 几种池化
    """

    def __init__(self, hidden_dim: int = 512, readout_type: str = 'attention'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.readout_type = readout_type

        if readout_type == 'attention':
            self.W_read = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.xavier_uniform_(self.W_read.weight)
        elif readout_type == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            self.W_read = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.xavier_uniform_(self.W_read.weight)

    def forward(
            self,
            z_sub: torch.Tensor,
            q_proj: torch.Tensor,
            node_indices: Optional[List[int]] = None,
            return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if q_proj.dim() == 1:
            q_proj = q_proj.unsqueeze(0)

        N = z_sub.shape[0]
        if N == 0:
            if return_weights:
                return torch.zeros(self.hidden_dim, device=z_sub.device), None
            return torch.zeros(self.hidden_dim, device=z_sub.device)

        if self.readout_type == 'attention':
            proj = self.W_read(z_sub)                     # [N, d]
            scores = torch.mm(q_proj, proj.T) / math.sqrt(self.hidden_dim)
            beta = F.softmax(scores, dim=1)               # [1, N]
            ctx = torch.mm(beta, z_sub).squeeze(0)
        elif self.readout_type == 'max':
            beta = None
            ctx = z_sub.max(dim=0)[0]
        elif self.readout_type == 'mean':
            beta = None
            ctx = z_sub.mean(dim=0)
        elif self.readout_type == 'gated':
            proj = self.W_read(z_sub)
            q_exp = q_proj.expand(N, -1)
            gi = torch.cat([proj, q_exp], dim=-1)
            gw = self.gate(gi)                            # gate weights
            scores = torch.mm(q_proj, proj.T) / math.sqrt(self.hidden_dim)
            beta = F.softmax(scores, dim=1) * gw.mean(dim=-1, keepdim=True)
            beta = F.softmax(beta, dim=1)                 # renormalize
            ctx = torch.mm(beta, z_sub).squeeze(0)
        else:
            raise ValueError(f"unknown readout: {self.readout_type}")

        if return_weights:
            return ctx, beta.squeeze(0) if beta is not None else None
        return ctx, None


class HierarchicalHypergraphReadout(nn.Module):
    """
    hierarchical readout (fixed)
    修正：用 global_to_sub_idx 替换了原来的错误映射
    """

    def __init__(self, hidden_dim=512, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.edge_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )
        self.node_readout = HypergraphReadout(hidden_dim, 'attention')

    def forward(
            self,
            z_sub,                # [num_sub_nodes, d]
            q_proj,               # [1, d]
            hyperedge_nodes,      # list of list of global indices per hyperedge
            global_to_sub_idx     # dict: global id -> position in z_sub
    ):
        if q_proj.dim() == 1:
            q_proj = q_proj.unsqueeze(0)

        # Step 1: aggregate each hyperedge
        edge_embs = []
        valid_idx = []
        for ei, nids in enumerate(hyperedge_nodes):
            local_ids = [global_to_sub_idx[g] for g in nids if g in global_to_sub_idx]
            if len(local_ids) < 2:
                continue
            emb = z_sub[local_ids].mean(dim=0)
            edge_embs.append(emb)
            valid_idx.append(ei)

        if len(edge_embs) == 0:
            # fallback: readout all sub-nodes
            ctx, nw = self.node_readout(z_sub, q_proj, return_weights=True)
            return ctx, torch.tensor([1.0], device=z_sub.device), nw

        edge_stack = torch.stack(edge_embs)                    # [E, d]

        # Step 2: hyperedge-level attention
        attn_out, edge_w = self.edge_attn(
            query=q_proj.unsqueeze(1),
            key=edge_stack.unsqueeze(0),
            value=edge_stack.unsqueeze(0),
            need_weights=True
        )
        edge_ctx = attn_out.squeeze(1)                         # [1, d]
        edge_w = edge_w.squeeze(0)                             # [E]

        # Step 3: node-level attention on top hyperedge
        top_i = edge_w.argmax().item()
        top_global = hyperedge_nodes[valid_idx[top_i]]
        top_local = [global_to_sub_idx[g] for g in top_global if g in global_to_sub_idx]
        if len(top_local) >= 2:
            z_top = z_sub[top_local]
            ctx, nw = self.node_readout(z_top, q_proj, return_weights=True)
        else:
            ctx, nw = self.node_readout(z_sub, q_proj, return_weights=True)

        final_ctx = edge_ctx.squeeze(0) + ctx
        final_ctx = F.normalize(final_ctx, dim=0)
        return final_ctx, edge_w, nw


class HyperedgeAwareReasoner(nn.Module):
    """
    Hyperedge-aware reasoner (fixed)
    """

    def __init__(self, hidden_dim=512, top_k_seed=10, M=5, readout_type='attention'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k_seed = top_k_seed
        self.M = M

        self.readout = HypergraphReadout(hidden_dim, readout_type)
        self.hier_readout = HierarchicalHypergraphReadout(hidden_dim)

        self.edge_matcher = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def _edge_rel(self, edge_nodes, z, q_proj, H_dict):
        # compute relevance of one hyperedge
        if len(edge_nodes) < 2:
            return torch.tensor(0.0, device=z.device)
        e_emb = z[edge_nodes].mean(dim=0, keepdim=True)
        c = torch.cat([e_emb, q_proj], dim=-1)
        return torch.sigmoid(self.edge_matcher(c).squeeze())

    def retrieve_subgraph(self, z, q_proj, hypergraph, H_dict, node_to_idx):
        """
        return z_sub, sub_node_indices, sub_edges, edge_relevance_dict, global_to_sub_idx
        """
        N = z.shape[0]
        sim = F.cosine_similarity(q_proj, z, dim=1)
        k = min(self.top_k_seed, N)
        seeds = torch.topk(sim, k).indices.tolist()
        seed_set = set(seeds)

        # flatten all hyperedges
        all_edges = hypergraph.get('hyperedges', [])
        if isinstance(all_edges, dict):
            flat = []
            for t, elist in all_edges.items():
                for e in elist:
                    ec = e.copy()
                    ec['type'] = t
                    flat.append(ec)
            all_edges = flat

        cand_edges = []
        scores = []
        for e in all_edges:
            nids = e.get('nodes', [])
            idxs = [node_to_idx[nid] for nid in nids if nid in node_to_idx]
            if len(idxs) < 2:
                continue
            if not (set(idxs) & seed_set):
                continue
            rel = self._edge_rel(idxs, z, q_proj, H_dict)
            cand_edges.append(idxs)
            scores.append(rel.item())

        if len(cand_edges) == 0:
            z_sub = z[seeds]
            g2l = {idx: pos for pos, idx in enumerate(seeds)}
            return z_sub, seeds, [], {}, g2l

        # pick top M edges
        sorted_ = sorted(zip(cand_edges, scores), key=lambda x: x[1], reverse=True)
        top_edges = sorted_[:self.M]

        # 种子节点必须保留
        sub_nodes = set(seeds)
        sub_edges = []
        edge_rel = {}
        for eidxs, scr in top_edges:
            sub_nodes.update(eidxs)
            sub_edges.append(eidxs)
            edge_rel[str(eidxs)] = scr

        sub_list = list(sub_nodes)
        g2l = {g: pos for pos, g in enumerate(sub_list)}
        z_sub = z[sub_list]
        return z_sub, sub_list, sub_edges, edge_rel, g2l

    def forward(self, z, q_proj, hypergraph, H_dict, node_to_idx, use_hier=True):
        z_sub, sub_ids, sub_edges, edge_rel, g2l = self.retrieve_subgraph(
            z, q_proj, hypergraph, H_dict, node_to_idx)

        info = {
            'sub_indices': sub_ids,
            'sub_edges': sub_edges,
            'edge_relevance': edge_rel,
            'num_sub_nodes': len(sub_ids),
            'num_sub_edges': len(sub_edges)
        }

        if use_hier and len(sub_edges) > 0:
            ctx, ew, nw = self.hier_readout(z_sub, q_proj, sub_edges, g2l)
            info['edge_weights'] = ew.tolist() if ew is not None else []
            info['node_weights'] = nw.tolist() if nw is not None else []
        else:
            ctx, nw = self.readout(z_sub, q_proj, return_weights=True)
            info['node_weights'] = nw.tolist() if nw is not None else []
            info['edge_weights'] = []

        return ctx, info


def create_reasoner(hidden_dim=512, top_k_seed=10, M=5, readout_type='attention'):
    return HyperedgeAwareReasoner(hidden_dim, top_k_seed, M, readout_type)