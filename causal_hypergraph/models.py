# causal_hypergraph/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from har.hypergraph_readout import HyperedgeAwareReasoner, create_reasoner


class TypeAwareHypergraphConv(nn.Module):
    """Type-aware hypergraph convolution (Eq.4) - unified version"""
    def __init__(self, in_channels, out_channels, num_types=3, type_names=None):
        super().__init__()
        self.num_types = num_types
        self.type_names = type_names or ['MO', 'OM', 'CO']
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.Theta = nn.ModuleList([
            nn.Linear(in_channels, out_channels) for _ in range(num_types)
        ])
        # attention over types, simple MLP
        self.attention_mlp = nn.Sequential(
            nn.Linear(in_channels + 64, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        self.type_embeddings = nn.Parameter(torch.randn(num_types, 64))
        self.reset_parameters()

    def reset_parameters(self):
        for theta in self.Theta:
            nn.init.xavier_uniform_(theta.weight)
            nn.init.zeros_(theta.bias)
        nn.init.normal_(self.type_embeddings, std=0.1)   # 随便初始化一下

    def _single_type_conv(self, x, H_tau, W_e_tau, D_v_tau, D_e_tau, theta):
        # standard hypergraph convolution for one type
        D_v_inv_sqrt = torch.pow(D_v_tau + 1e-8, -0.5).clamp(min=0)
        D_e_inv = torch.pow(D_e_tau + 1e-8, -1.0).clamp(min=0)
        out = theta(x)
        out = out * D_v_inv_sqrt.unsqueeze(1)
        out = torch.mm(H_tau.T, out)
        out = out * W_e_tau.unsqueeze(1) * D_e_inv.unsqueeze(1)
        out = torch.mm(H_tau, out)
        out = out * D_v_inv_sqrt.unsqueeze(1)
        return out

    def forward(self, x, H_dict, W_e_dict, D_v_dict, D_e_dict):
        type_outputs = []
        gap = x.mean(dim=0, keepdim=True).detach()   # global context for attention

        for tau_idx, tau_name in enumerate(self.type_names):
            if tau_name not in H_dict or H_dict[tau_name].size(1) == 0:
                continue
            tau_out = self._single_type_conv(
                x, H_dict[tau_name], W_e_dict[tau_name],
                D_v_dict[tau_name], D_e_dict[tau_name],
                self.Theta[tau_idx]
            )
            type_outputs.append((tau_idx, tau_out))

        if len(type_outputs) == 0:
            return torch.zeros(x.shape[0], self.out_channels, device=x.device), torch.ones(1, device=x.device)

        # compute attention weights using type embeddings
        attn_inputs = torch.cat([
            torch.cat([gap, self.type_embeddings[tau_idx].unsqueeze(0)], dim=-1)
            for tau_idx, _ in type_outputs
        ], dim=0)
        attn_weights = F.softmax(self.attention_mlp(attn_inputs).squeeze(-1), dim=0)

        out = torch.zeros_like(type_outputs[0][1])
        for idx, (_, tau_out) in enumerate(type_outputs):
            out += attn_weights[idx] * tau_out
        return F.relu(out), attn_weights


class HGNE(nn.Module):
    """Hypergraph Neural Network Encoder - returns (z, attention_log)"""
    def __init__(self, in_channels=1024, hidden_channels=512, num_layers=3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(TypeAwareHypergraphConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(TypeAwareHypergraphConv(hidden_channels, hidden_channels))

        self.dropout = nn.Dropout(0.2)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])

        # 特征拼接后投影
        total_dim = in_channels + num_layers * hidden_channels
        self.proj = nn.Linear(total_dim, hidden_channels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        self.q_proj = nn.Linear(1024, hidden_channels)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)

    def forward(self, x, H_dict, W_e_dict, D_v_dict, D_e_dict):
        x_list = [x]
        attention_log = []

        for i, conv in enumerate(self.convs):
            x_new, attn_weights = conv(x, H_dict, W_e_dict, D_v_dict, D_e_dict)
            x_new = self.layer_norms[i](x_new)
            x_new = self.dropout(x_new)
            x = x_new
            x_list.append(x_new)
            attention_log.append(attn_weights.detach())

        z = torch.cat(x_list, dim=-1)
        z = self.proj(z)
        return z, attention_log

    def project_question(self, q_emb):
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)
        return self.q_proj(q_emb)


class EndToEndModel(nn.Module):
    """Full end-to-end model with L_fine + λ1*L_pre + λ2*L_aux"""
    def __init__(self, hgne: HGNE, hidden_dim=512, num_answers=5,
                 use_har=True, top_k_seed=10, M=5):
        super().__init__()
        self.hgne = hgne
        self.hidden_dim = hidden_dim
        self.use_har = use_har

        if use_har:
            self.reasoner = create_reasoner(hidden_dim, top_k_seed, M)
        else:
            self.reasoner = None

        # classifier for answer prediction
        self.answer_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_answers)
        )

        # auxiliary type classifier
        self.type_aux_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)
        )

        self.freeze_hgne = False

    def forward(self, x, H_dict, W_e_dict, D_v_dict, D_e_dict,
                q_emb, hypergraph=None, node_to_idx=None,
                return_all=False, type_label=None):
        with torch.set_grad_enabled(not self.freeze_hgne):
            z, attn_log = self.hgne(x, H_dict, W_e_dict, D_v_dict, D_e_dict)

        q_proj = self.hgne.project_question(q_emb)

        # decide how to get context: HAR or simple attention
        if self.use_har and hypergraph is not None and node_to_idx is not None:
            context, reasoner_info = self.reasoner(z, q_proj, hypergraph, H_dict, node_to_idx)
        else:
            attn_scores = torch.mm(q_proj, z.T) / (self.hidden_dim ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=1)
            context = torch.mm(attn_weights, z).squeeze(0)
            reasoner_info = {}

        combined = torch.cat([context.unsqueeze(0), q_proj], dim=-1)
        logits = self.answer_classifier(combined).squeeze(0)

        aux_logits = self.type_aux_classifier(combined)
        if type_label is not None and type_label < 3:
            aux_loss = F.cross_entropy(aux_logits, torch.tensor([type_label], device=logits.device))
        else:
            aux_loss = torch.tensor(0.0, device=logits.device)

        if return_all:
            return {
                'logits': logits,
                'context': context,
                'q_proj': q_proj,
                'z': z,
                'attn_log': attn_log,
                'reasoner_info': reasoner_info,
                'aux_logits': aux_logits,
                'aux_loss': aux_loss
            }
        return logits, aux_loss

    def set_freeze_hgne(self, freeze=True):
        self.freeze_hgne = freeze
        for param in self.hgne.parameters():
            param.requires_grad = not freeze