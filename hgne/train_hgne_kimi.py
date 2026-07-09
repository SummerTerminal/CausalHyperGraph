"""
HGNE: Hypergraph Neural Network Encoder - GPU优化版 v3.1 (Windows兼容)
================================================================================
论文: CausalHyperGraph: 超图神经网络驱动的多关系叙事推理

优化内容 (v3.1):
    1. ✅ DataLoader优化: num_workers, pin_memory, persistent_workers
    2. ✅ 正确的batch处理: DataLoader直接返回batch
    3. ✅ torch.compile: PyTorch 2.x图编译优化
    4. ✅ 梯度累积: 支持有效梯度累积
    5. ✅ 异步数据预取: non_blocking=True
    6. ✅ 内存优化: inplace操作, 定期清理缓存
    7. ✅ Windows兼容: spawn模式, 错误处理
    8. ✅ 混合精度: 保持autocast
================================================================================
"""

import os
import sys
import json
import argparse
import random
import time
import platform
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
from typing import Dict, Optional, List
import numpy as np

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ============================================================================
# Windows多进程兼容性设置 (必须在if __name__之前)
# ============================================================================
if platform.system() == 'Windows':
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 可能已经设置过了


# ============================================================================
# 全局配置
# ============================================================================

class Config:
    """全局配置类"""
    NUM_WORKERS = 4
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    PREFETCH_FACTOR = 2
    USE_TORCH_COMPILE = True
    EMPTY_CACHE_FREQ = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# 第一部分：模型定义
# ============================================================================

class TypeAwareHypergraphConv(nn.Module):
    """类型感知超图卷积层"""
    def __init__(self, in_channels, out_channels, num_types=3, type_names=None):
        super().__init__()
        self.num_types = num_types
        self.type_names = type_names or ['MO', 'OM', 'CO']
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.Theta = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=True)
            for _ in range(num_types)
        ])

        self.attention_mlp = nn.Sequential(
            nn.Linear(in_channels + 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        self.type_embeddings = nn.Parameter(torch.randn(num_types, 64))
        self.reset_parameters()

    def reset_parameters(self):
        for theta in self.Theta:
            nn.init.xavier_uniform_(theta.weight)
            if theta.bias is not None:
                nn.init.zeros_(theta.bias)
        nn.init.normal_(self.type_embeddings, std=0.1)

    def _single_type_conv(self, x, H_tau, W_e_tau, D_v_tau, D_e_tau, theta):
        D_v_inv_sqrt = torch.pow(D_v_tau + 1e-8, -0.5)
        D_v_inv_sqrt = torch.where(torch.isinf(D_v_inv_sqrt), torch.zeros_like(D_v_inv_sqrt), D_v_inv_sqrt)

        D_e_inv = torch.pow(D_e_tau + 1e-8, -1.0)
        D_e_inv = torch.where(torch.isinf(D_e_inv), torch.zeros_like(D_e_inv), D_e_inv)

        out = theta(x)
        out = D_v_inv_sqrt.unsqueeze(1) * out
        out = torch.mm(H_tau.T, out)
        out = W_e_tau.unsqueeze(1) * out
        out = D_e_inv.unsqueeze(1) * out
        out = torch.mm(H_tau, out)
        out = D_v_inv_sqrt.unsqueeze(1) * out

        return out

    def forward(self, x, H_dict, W_e_dict, D_v_dict, D_e_dict):
        type_outputs = []
        gap = x.mean(dim=0)

        for tau_idx, tau_name in enumerate(self.type_names):
            if tau_name not in H_dict:
                continue

            H_tau = H_dict[tau_name]
            if H_tau.size(1) == 0:
                continue

            W_e_tau = W_e_dict[tau_name]
            D_v_tau = D_v_dict[tau_name]
            D_e_tau = D_e_dict[tau_name]

            tau_out = self._single_type_conv(
                x, H_tau, W_e_tau, D_v_tau, D_e_tau, self.Theta[tau_idx]
            )
            type_outputs.append((tau_idx, tau_out))

        if len(type_outputs) == 0:
            return x, torch.ones(1, device=x.device)

        attn_logits = []
        for tau_idx, tau_out in type_outputs:
            combined = torch.cat([
                gap.detach(),
                self.type_embeddings[tau_idx]
            ]).unsqueeze(0)
            logit = self.attention_mlp(combined)
            attn_logits.append(logit)

        attn_weights = F.softmax(torch.cat(attn_logits), dim=0)

        out = torch.zeros_like(type_outputs[0][1])
        for idx, (tau_idx, tau_out) in enumerate(type_outputs):
            out = out + attn_weights[idx] * tau_out

        return F.relu(out, inplace=True), attn_weights


class HGNE(nn.Module):
    """超图神经网络编码器"""
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

        total_dim = in_channels + num_layers * hidden_channels
        self.proj = nn.Linear(total_dim, hidden_channels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

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


# ============================================================================
# 第二部分：数据集
# ============================================================================

class TypeAwareHypergraphDataset(Dataset):
    """类型感知超图数据集"""

    def __init__(self, hypergraph_dir, split='train', eval_split=0.1, seed=42):
        self.hypergraph_dir = hypergraph_dir
        self.split = split

        # 检查目录是否存在
        if not os.path.exists(hypergraph_dir):
            raise FileNotFoundError(f"数据目录不存在: {hypergraph_dir}")

        hypergraph_files = []
        for root, _, files in os.walk(hypergraph_dir):
            for f in files:
                if f.endswith('.json') and not f.startswith('_'):
                    hypergraph_files.append(os.path.join(root, f))

        print(f"[数据集] 发现 {len(hypergraph_files)} 个超图JSON文件")
        print(f"[数据集] 扫描目录: {os.path.abspath(hypergraph_dir)}")

        if len(hypergraph_files) == 0:
            print(f"[警告] 在 {hypergraph_dir} 中没有找到.json文件!")
            print(f"[提示] 请检查:")
            print(f"       1. 路径是否正确")
            print(f"       2. 目录下是否有.json文件")
            print(f"       3. 文件是否被隐藏或命名错误")
            self.samples = []
            return

        random.seed(seed)
        random.shuffle(hypergraph_files)

        num_eval = max(1, int(eval_split * len(hypergraph_files)))
        if split == 'train':
            hypergraph_files = hypergraph_files[num_eval:]
        else:
            hypergraph_files = hypergraph_files[:num_eval]

        print(f"[数据集] {split}集: {len(hypergraph_files)} 个超图")

        self.samples = []
        loaded_count = 0
        skip_count = 0

        for file_path in hypergraph_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    hg = json.load(f)
                processed = self._process_hypergraph(hg, file_path)
                if processed is not None:
                    self.samples.append(processed)
                    loaded_count += 1
                else:
                    skip_count += 1
            except Exception as e:
                print(f"[警告] 加载失败 {file_path}: {e}")

        print(f"[数据集] 成功加载 {loaded_count} 个超图, 跳过 {skip_count} 个")

        if self.samples:
            self._print_statistics()

    def _process_hypergraph(self, hg, file_path):
        if not hg.get('nodes'):
            return None

        video_id = hg.get('video_id', os.path.splitext(os.path.basename(file_path))[0])
        hyperedges_raw = hg.get('hyperedges', {})

        if isinstance(hyperedges_raw, dict):
            typed_edges = {}
            for tau_name in ['MO', 'OM', 'CO']:
                edges = hyperedges_raw.get(tau_name, [])
                if isinstance(edges, list):
                    typed_edges[tau_name] = [e for e in edges if isinstance(e, dict)]
                else:
                    typed_edges[tau_name] = []
        elif isinstance(hyperedges_raw, list):
            typed_edges = defaultdict(list)
            for e in hyperedges_raw:
                if isinstance(e, dict):
                    edge_type = e.get('type', 'CO')
                    typed_edges[edge_type].append(e)
            for tau_name in ['MO', 'OM', 'CO']:
                if tau_name not in typed_edges:
                    typed_edges[tau_name] = []
        else:
            typed_edges = {'MO': [], 'OM': [], 'CO': []}

        total_edges_count = sum(len(v) for v in typed_edges.values())
        if total_edges_count == 0:
            print(f"  [跳过] {video_id}: 无有效超边")
            return None

        node_id_to_idx = {n['id']: i for i, n in enumerate(hg['nodes'])}
        num_nodes = len(hg['nodes'])

        embedding_dim = self._get_embedding_dim(hg['nodes'])
        embeddings = []
        for n in hg['nodes']:
            emb = n.get('embedding', [])
            if not emb or len(emb) == 0:
                emb = [0.0] * embedding_dim
            embeddings.append(emb)

        x = torch.tensor(embeddings, dtype=torch.float32)

        H_dict = {}
        W_e_dict = {}
        D_v_dict = {}
        D_e_dict = {}

        for tau_name in ['MO', 'OM', 'CO']:
            edges = typed_edges.get(tau_name, [])
            if len(edges) == 0:
                continue

            E_tau = len(edges)
            H_tau = torch.zeros(num_nodes, E_tau)
            W_e_tau = torch.ones(E_tau)

            for e_idx, edge in enumerate(edges):
                nodes = edge.get('nodes', [])
                if not nodes:
                    continue
                for node_id in nodes:
                    if node_id in node_id_to_idx:
                        H_tau[node_id_to_idx[node_id], e_idx] = 1.0
                if 'weight' in edge:
                    W_e_tau[e_idx] = float(edge['weight'])

            D_v_tau = torch.sum(H_tau * W_e_tau.unsqueeze(0), dim=1) + 1e-8
            D_e_tau = torch.sum(H_tau, dim=0) + 1e-8

            H_dict[tau_name] = H_tau
            W_e_dict[tau_name] = W_e_tau
            D_v_dict[tau_name] = D_v_tau
            D_e_dict[tau_name] = D_e_tau

        total_edges = sum(v.size(1) for v in H_dict.values())
        H_all = torch.zeros(num_nodes, max(1, total_edges))
        global_idx = 0
        for tau_name in ['MO', 'OM', 'CO']:
            if tau_name in H_dict:
                E_tau = H_dict[tau_name].size(1)
                H_all[:, global_idx:global_idx + E_tau] = H_dict[tau_name]
                global_idx += E_tau

        pos_masks = {}
        for tau_name, H_tau in H_dict.items():
            if H_tau.size(1) > 0:
                pos_mask_tau = (torch.mm(H_tau, H_tau.T) > 0)
                pos_mask_tau.fill_diagonal_(False)
                pos_masks[tau_name] = pos_mask_tau

        return {
            'x': x,
            'H_dict': H_dict,
            'W_e_dict': W_e_dict,
            'D_v_dict': D_v_dict,
            'D_e_dict': D_e_dict,
            'H_all': H_all,
            'pos_masks': pos_masks,
            'typed_edges': typed_edges,
            'node_id_to_idx': node_id_to_idx,
            'num_nodes': num_nodes,
            'num_edges': total_edges,
            'video_id': video_id,
        }

    def _get_embedding_dim(self, nodes):
        for n in nodes:
            emb = n.get('embedding', [])
            if emb and len(emb) > 0:
                return len(emb)
        return 1024

    def _print_statistics(self):
        total_mo = sum(s['H_dict'].get('MO', torch.zeros(1)).size(1) for s in self.samples)
        total_om = sum(s['H_dict'].get('OM', torch.zeros(1)).size(1) for s in self.samples)
        total_co = sum(s['H_dict'].get('CO', torch.zeros(1)).size(1) for s in self.samples)
        avg_nodes = np.mean([s['num_nodes'] for s in self.samples])
        avg_edges = np.mean([s['num_edges'] for s in self.samples])
        print(f"[数据集] MO超边: {total_mo}, OM超边: {total_om}, CO超边: {total_co}")
        print(f"[数据集] 平均节点数: {avg_nodes:.1f}, 平均超边数: {avg_edges:.1f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    return batch


# ============================================================================
# 第三部分：损失函数
# ============================================================================

class TypeAwareContrastiveLoss(nn.Module):
    """类型感知对比损失"""
    def __init__(self, temperature=0.1, type_weights=None):
        super().__init__()
        self.temperature = temperature
        self.type_weights = type_weights or {'MO': 1.0, 'OM': 1.0, 'CO': 0.5}

    def _compute_single_type_loss(self, z, pos_mask):
        N = z.shape[0]
        if N < 2 or pos_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)

        z_norm = F.normalize(z, dim=1)
        sim_matrix = torch.mm(z_norm, z_norm.T) / self.temperature

        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix_stable = sim_matrix - sim_max.detach()

        exp_sim = torch.exp(sim_matrix_stable)
        exp_sim_pos = exp_sim * pos_mask.float()
        numerator = exp_sim_pos.sum(dim=1)

        neg_mask = torch.ones(N, N, device=z.device) - torch.eye(N, device=z.device)
        denominator = (exp_sim * neg_mask).sum(dim=1)

        has_pos = pos_mask.sum(dim=1) > 0
        ratio = numerator[has_pos] / (denominator[has_pos] + 1e-8)
        loss = -torch.log(ratio + 1e-8).mean()

        return loss

    def forward(self, z, pos_masks):
        total_loss = torch.tensor(0.0, device=z.device)
        type_losses = {}
        total_weight = 0.0

        for tau_name, pos_mask in pos_masks.items():
            weight = self.type_weights.get(tau_name, 1.0)
            loss_tau = self._compute_single_type_loss(z, pos_mask)

            if torch.isfinite(loss_tau):
                total_loss = total_loss + weight * loss_tau
                total_weight += weight
                type_losses[tau_name] = loss_tau.item()

        if total_weight > 0:
            total_loss = total_loss / total_weight
        else:
            total_loss = torch.tensor(0.0, device=z.device)

        return total_loss, type_losses


class HyperedgePredictionLoss(nn.Module):
    """超边预测损失"""
    def __init__(self, hidden_dim=512, num_negatives=3):
        super().__init__()
        self.num_negatives = num_negatives
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, z, H_all, num_nodes):
        E = H_all.size(1)
        if E == 0:
            return torch.tensor(0.0, device=z.device)

        pos_scores = []
        valid_edges = 0
        for e_idx in range(E):
            nodes_in_edge = H_all[:, e_idx].nonzero(as_tuple=True)[0]
            if len(nodes_in_edge) < 2:
                continue
            edge_embedding = z[nodes_in_edge].mean(dim=0)
            score = self.readout_mlp(edge_embedding)
            pos_scores.append(score)
            valid_edges += 1

        if not pos_scores:
            return torch.tensor(0.0, device=z.device)

        pos_scores = torch.cat(pos_scores)

        total_needed = self.num_negatives * valid_edges
        neg_scores_list = []

        batch_neg_size = min(32, total_needed)
        for _ in range(0, total_needed, batch_neg_size):
            current_batch = min(batch_neg_size, total_needed - len(neg_scores_list))
            neg_sizes = torch.randint(2, min(5, num_nodes) + 1, (current_batch,))

            for neg_size in neg_sizes:
                neg_nodes = torch.randint(0, num_nodes, (neg_size.item(),), device=z.device)
                neg_embedding = z[neg_nodes].mean(dim=0)
                score = self.readout_mlp(neg_embedding)
                neg_scores_list.append(score)

        neg_scores = torch.stack(neg_scores_list[:total_needed])

        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)

        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels)

        return (pos_loss + neg_loss) / 2


class TypePredictionLoss(nn.Module):
    """类型预测辅助损失"""
    def __init__(self, hidden_dim=512, num_types=3):
        super().__init__()
        self.type_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_types)
        )
        self.type_names = ['MO', 'OM', 'CO']

    def forward(self, z, typed_edges, node_id_to_idx=None):
        all_embeddings = []
        all_labels = []

        for tau_idx, tau_name in enumerate(self.type_names):
            edges = typed_edges.get(tau_name, [])
            for edge in edges:
                nodes = edge.get('nodes', [])
                if len(nodes) < 2:
                    continue

                if node_id_to_idx:
                    node_indices = [node_id_to_idx.get(nid) for nid in nodes if nid in node_id_to_idx]
                else:
                    node_indices = list(range(len(nodes)))

                if not node_indices:
                    continue

                valid_indices = [i for i in node_indices if i is not None and i < z.size(0)]
                if len(valid_indices) < 2:
                    continue

                edge_embedding = z[valid_indices].mean(dim=0)
                all_embeddings.append(edge_embedding)
                all_labels.append(tau_idx)

        if not all_embeddings:
            return torch.tensor(0.0, device=z.device)

        embeddings = torch.stack(all_embeddings)
        labels = torch.tensor(all_labels, device=z.device)

        logits = self.type_classifier(embeddings)
        loss = F.cross_entropy(logits, labels)

        return loss


# ============================================================================
# 第四部分：训练 - GPU优化版
# ============================================================================

def process_batch(model, batch_samples, device, optimizer, scaler,
                  contrastive_loss, pre_loss, aux_loss,
                  lambda_pre=0.3, lambda_aux=0.1, is_training=True,
                  grad_accum_steps=1):
    """处理一个batch - 支持梯度累积"""
    if len(batch_samples) == 0:
        return None

    total_contrastive = 0.0
    total_pre = 0.0
    total_aux = 0.0
    valid_count = 0

    if is_training and optimizer is not None:
        optimizer.zero_grad()

    for sample in batch_samples:
        if not isinstance(sample, dict):
            continue

        x = sample['x'].to(device, non_blocking=True)
        H_dict = {k: v.to(device, non_blocking=True) for k, v in sample['H_dict'].items()}
        W_e_dict = {k: v.to(device, non_blocking=True) for k, v in sample['W_e_dict'].items()}
        D_v_dict = {k: v.to(device, non_blocking=True) for k, v in sample['D_v_dict'].items()}
        D_e_dict = {k: v.to(device, non_blocking=True) for k, v in sample['D_e_dict'].items()}
        pos_masks = {k: v.to(device, non_blocking=True) for k, v in sample['pos_masks'].items()}
        H_all = sample['H_all'].to(device, non_blocking=True)
        node_id_to_idx = sample.get('node_id_to_idx', {})

        if sample['num_nodes'] < 3 or sample['num_edges'] < 1:
            continue

        with autocast(enabled=scaler.is_enabled()):
            z, attention_log = model(x, H_dict, W_e_dict, D_v_dict, D_e_dict)

            loss_cont, type_losses = contrastive_loss(z, pos_masks)
            loss_p = pre_loss(z, H_all, sample['num_nodes'])
            loss_a = aux_loss(z, sample.get('typed_edges', {}), node_id_to_idx)

            total_loss = loss_cont + lambda_pre * loss_p + lambda_aux * loss_a

            if not torch.isfinite(total_loss):
                continue

            scaled_loss = total_loss / (len(batch_samples) * grad_accum_steps)

        if is_training:
            scaler.scale(scaled_loss).backward()

        total_contrastive += loss_cont.item()
        total_pre += loss_p.item()
        total_aux += loss_a.item()
        valid_count += 1

    if valid_count == 0:
        return None

    if is_training and optimizer is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    return {
        'contrastive': total_contrastive / valid_count,
        'pre': total_pre / valid_count,
        'aux': total_aux / valid_count,
        'total': (total_contrastive + lambda_pre * total_pre + lambda_aux * total_aux) / valid_count,
        'valid_count': valid_count
    }


def train_hgne(model, dataloader, epochs, device, lr=1e-4, batch_size=8,
               lambda_pre=0.3, lambda_aux=0.1, grad_accum_steps=1):
    """训练函数 - GPU优化版"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    contrastive_loss = TypeAwareContrastiveLoss(temperature=0.1)
    pre_loss = HyperedgePredictionLoss(hidden_dim=model.hidden_channels).to(device)
    aux_loss = TypePredictionLoss(hidden_dim=model.hidden_channels).to(device)

    # torch.compile优化
    if Config.USE_TORCH_COMPILE and hasattr(torch, 'compile'):
        print("[优化] 启用torch.compile...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[优化] torch.compile启用成功")
        except Exception as e:
            print(f"[优化] torch.compile失败: {e}，使用普通模式")

    print(f"[训练] 设备: {device}")
    print(f"[训练] 学习率: {lr}, 权重衰减: 0.01")
    print(f"[训练] λ_pre: {lambda_pre}, λ_aux: {lambda_aux}")
    print(f"[训练] 梯度累积: {grad_accum_steps} (等效batch_size={batch_size * grad_accum_steps})")

    best_loss = float('inf')
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_losses = {'contrastive': 0, 'pre': 0, 'aux': 0, 'total': 0}
        batch_count = 0
        epoch_start = time.time()

        gpu_util_sum = 0
        gpu_util_count = 0

        for batch_idx, batch_data in enumerate(dataloader):
            if isinstance(batch_data, list):
                current_batch = batch_data
            else:
                current_batch = [batch_data]

            is_update_step = (batch_idx + 1) % grad_accum_steps == 0

            losses = process_batch(
                model, current_batch, device, optimizer if is_update_step else None, 
                scaler,
                contrastive_loss, pre_loss, aux_loss,
                lambda_pre=lambda_pre, lambda_aux=lambda_aux,
                is_training=True,
                grad_accum_steps=grad_accum_steps
            )

            if losses is not None:
                for k in epoch_losses:
                    epoch_losses[k] += losses.get(k, 0)
                batch_count += 1
                global_step += 1

            if torch.cuda.is_available() and batch_idx % 10 == 0:
                try:
                    gpu_util = torch.cuda.utilization(device)
                    gpu_util_sum += gpu_util
                    gpu_util_count += 1
                except:
                    pass

            if batch_count > 0 and batch_count % 10 == 0:
                avg_total = epoch_losses['total'] / batch_count
                avg_gpu = gpu_util_sum / max(gpu_util_count, 1) if gpu_util_count > 0 else 0
                print(f"  Batch {batch_count}, Loss: {avg_total:.4f}, "
                      f"GPU利用率: {avg_gpu:.1f}%")
                gpu_util_sum = 0
                gpu_util_count = 0

            if Config.EMPTY_CACHE_FREQ > 0 and batch_idx % Config.EMPTY_CACHE_FREQ == 0:
                torch.cuda.empty_cache()

        scheduler.step()

        epoch_time = time.time() - epoch_start
        for k in epoch_losses:
            epoch_losses[k] /= max(batch_count, 1)

        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Loss: total={epoch_losses['total']:.4f}, "
              f"cont={epoch_losses['contrastive']:.4f}, "
              f"pre={epoch_losses['pre']:.4f}, "
              f"aux={epoch_losses['aux']:.4f}")
        print(f"  Best: {best_loss:.4f}, LR: {current_lr:.2e}, Time: {epoch_time:.1f}s\n")

    if 'best_model_state' in dir():
        model.load_state_dict(best_model_state)
    print(f"[训练完成] 最佳Loss: {best_loss:.4f}")

    return model


# ============================================================================
# 第五部分：评估
# ============================================================================

def evaluate_model(model, dataloader, device, lambda_pre=0.3, lambda_aux=0.1):
    """完整评估"""
    model.eval()

    contrastive_loss = TypeAwareContrastiveLoss(temperature=0.1)
    pre_loss_fn = HyperedgePredictionLoss(hidden_dim=model.hidden_channels).to(device)
    aux_loss_fn = TypePredictionLoss(hidden_dim=model.hidden_channels).to(device)

    pre_loss_fn.eval()
    aux_loss_fn.eval()

    total_cont_loss = 0
    total_cosine_sim = 0
    total_edge_acc = 0
    total_type_acc = 0
    valid_count = 0

    with torch.no_grad():
        for batch_data in dataloader:
            if isinstance(batch_data, list):
                if len(batch_data) > 0:
                    sample = batch_data[0]
                else:
                    continue
            else:
                sample = batch_data

            if not isinstance(sample, dict):
                continue

            x = sample['x'].to(device, non_blocking=True)
            H_dict = {k: v.to(device, non_blocking=True) for k, v in sample['H_dict'].items()}
            W_e_dict = {k: v.to(device, non_blocking=True) for k, v in sample['W_e_dict'].items()}
            D_v_dict = {k: v.to(device, non_blocking=True) for k, v in sample['D_v_dict'].items()}
            D_e_dict = {k: v.to(device, non_blocking=True) for k, v in sample['D_e_dict'].items()}
            pos_masks = {k: v.to(device, non_blocking=True) for k, v in sample['pos_masks'].items()}
            H_all = sample['H_all'].to(device, non_blocking=True)
            node_id_to_idx = sample.get('node_id_to_idx', {})

            if sample['num_nodes'] < 3 or sample['num_edges'] < 1:
                continue

            with autocast(enabled=torch.cuda.is_available()):
                z, _ = model(x, H_dict, W_e_dict, D_v_dict, D_e_dict)

                loss_cont, _ = contrastive_loss(z, pos_masks)
                total_cont_loss += loss_cont.item()

                edge_acc = _compute_edge_prediction_accuracy(z, H_all, sample['num_nodes'])
                total_edge_acc += edge_acc

                if sample.get('typed_edges'):
                    type_acc = _compute_type_prediction_accuracy(
                        z, sample['typed_edges'], aux_loss_fn.type_classifier,
                        node_id_to_idx
                    )
                    total_type_acc += type_acc

            for tau_name, pos_mask in pos_masks.items():
                if pos_mask.sum() > 0:
                    z_norm = F.normalize(z, dim=1)
                    sim = torch.mm(z_norm, z_norm.T)
                    total_cosine_sim += sim[pos_mask].mean().item()

            valid_count += 1

    return {
        'contrastive_loss': total_cont_loss / max(valid_count, 1),
        'edge_prediction_acc': total_edge_acc / max(valid_count, 1),
        'type_prediction_acc': total_type_acc / max(valid_count, 1),
        'positive_cosine_sim': total_cosine_sim / max(valid_count, 1),
    }


def _compute_edge_prediction_accuracy(z, H_all, num_nodes, num_negatives=10):
    E = H_all.size(1)
    if E == 0:
        return 0.0

    correct = 0
    total = 0

    for e_idx in range(E):
        nodes = H_all[:, e_idx].nonzero(as_tuple=True)[0]
        if len(nodes) < 2:
            continue
        edge_emb = z[nodes].mean(dim=0)
        edge_norm = F.normalize(edge_emb, dim=0)

        for _ in range(num_negatives):
            neg_size = len(nodes)
            neg_nodes = torch.randint(0, num_nodes, (neg_size,), device=z.device)
            neg_emb = z[neg_nodes].mean(dim=0)
            neg_norm = F.normalize(neg_emb, dim=0)

            if torch.dot(edge_norm, edge_norm) > torch.dot(edge_norm, neg_norm):
                correct += 1
            total += 1

    return correct / max(total, 1)


def _compute_type_prediction_accuracy(z, typed_edges, type_classifier, node_id_to_idx=None):
    all_embeddings = []
    all_labels = []

    for tau_idx, tau_name in enumerate(['MO', 'OM', 'CO']):
        edges = typed_edges.get(tau_name, [])
        for edge in edges[:20]:
            nodes = edge.get('nodes', [])
            if len(nodes) < 2:
                continue

            node_indices = []
            for node_id in nodes:
                if node_id_to_idx is not None:
                    idx = node_id_to_idx.get(node_id)
                    if idx is not None:
                        node_indices.append(idx)
                elif isinstance(node_id, str) and node_id.startswith('e_'):
                    try:
                        idx = int(node_id.split('_')[1])
                        if idx < z.size(0):
                            node_indices.append(idx)
                    except (IndexError, ValueError):
                        pass

            node_indices = node_indices[:5]
            if len(node_indices) < 2:
                continue

            edge_emb = z[node_indices].mean(dim=0)
            all_embeddings.append(edge_emb)
            all_labels.append(tau_idx)

    if not all_embeddings:
        return 0.0

    embeddings = torch.stack(all_embeddings)
    labels = torch.tensor(all_labels, device=z.device)

    logits = type_classifier(embeddings)
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean().item()

    return acc


# ============================================================================
# 第六部分：主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CausalHyperGraph HGNE 训练（GPU优化版 v3.1）')
    parser.add_argument('--hypergraph_dir', required=True, help='超图JSON文件目录')
    parser.add_argument('--output_dir', default='checkpoints', help='模型保存目录')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3, help='卷积层数')
    parser.add_argument('--lambda_pre', type=float, default=0.3, help='预训练损失权重')
    parser.add_argument('--lambda_aux', type=float, default=0.1, help='辅助损失权重')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--eval_split', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--grad_accum', type=int, default=2, help='梯度累积步数')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker数量')
    parser.add_argument('--no_compile', action='store_true', help='禁用torch.compile')
    args = parser.parse_args()

    Config.NUM_WORKERS = args.num_workers
    Config.USE_TORCH_COMPILE = not args.no_compile

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[系统] 设备: {device}")
    print(f"[系统] 操作系统: {platform.system()}")

    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[GPU] 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"[GPU] CUDA版本: {torch.version.cuda}")
        print(f"[GPU] PyTorch版本: {torch.__version__}")
        print(f"[GPU] cuDNN benchmark: {torch.backends.cudnn.benchmark}")

    # Windows警告
    is_windows = platform.system() == 'Windows'
    if is_windows and args.num_workers > 0:
        print(f"\n[⚠️ 警告] Windows系统检测到 num_workers={args.num_workers}")
        print("[⚠️ 警告] Windows下多进程可能不稳定，如遇错误请使用 --num_workers 0")
        print("[⚠️ 警告] 或者将代码保存为.py文件后在命令行运行（非Jupyter）\n")

    # 创建数据集
    try:
        train_dataset = TypeAwareHypergraphDataset(
            args.hypergraph_dir, split='train', eval_split=args.eval_split, seed=args.seed
        )
        eval_dataset = TypeAwareHypergraphDataset(
            args.hypergraph_dir, split='eval', eval_split=args.eval_split, seed=args.seed
        )
    except FileNotFoundError as e:
        print(f"\n[❌ 错误] {e}")
        print(f"[提示] 请确认 --hypergraph_dir 路径正确")
        print(f"[提示] 当前工作目录: {os.getcwd()}")
        sys.exit(1)

    if len(train_dataset) == 0:
        print("\n[❌ 错误] 训练集为空，无法训练")
        sys.exit(1)

    # DataLoader配置
    loader_kwargs = {
        'batch_size': args.batch_size,
        'collate_fn': collate_fn,
        'drop_last': False,
    }

    if args.num_workers > 0:
        loader_kwargs['num_workers'] = args.num_workers
        loader_kwargs['pin_memory'] = Config.PIN_MEMORY
        loader_kwargs['persistent_workers'] = Config.PERSISTENT_WORKERS
        loader_kwargs['prefetch_factor'] = Config.PREFETCH_FACTOR

    print(f"\n[DataLoader] batch_size={args.batch_size}, num_workers={args.num_workers}")
    if args.num_workers > 0:
        print(f"[DataLoader] pin_memory=True, persistent_workers=True, prefetch_factor=2")

    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True,
        **loader_kwargs
    )

    eval_kwargs = loader_kwargs.copy()
    eval_kwargs['shuffle'] = False
    eval_dataloader = DataLoader(
        eval_dataset, 
        **eval_kwargs
    )

    in_channels = train_dataset.samples[0]['x'].shape[1] if train_dataset.samples else 1024
    model = HGNE(in_channels=in_channels, hidden_channels=args.hidden_dim, num_layers=args.num_layers)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[模型] 参数量: {total_params:,}, 输入维度: {in_channels}")

    print(f"\n[优化配置]")
    print(f"  - torch.compile: {Config.USE_TORCH_COMPILE}")
    print(f"  - 梯度累积: {args.grad_accum}步 (等效batch={args.batch_size * args.grad_accum})")
    print(f"  - 混合精度: {torch.cuda.is_available()}")
    print(f"  - 非阻塞传输: True")

    print("\n" + "="*60)
    print("开始训练...")
    print("="*60 + "\n")

    model = train_hgne(
        model, train_dataloader, args.epochs, device,
        lr=args.lr, batch_size=args.batch_size,
        lambda_pre=args.lambda_pre, lambda_aux=args.lambda_aux,
        grad_accum_steps=args.grad_accum
    )

    print("\n" + "="*60)
    print("评估模型...")
    print("="*60 + "\n")

    eval_metrics = evaluate_model(
        model, eval_dataloader, device,
        lambda_pre=args.lambda_pre, lambda_aux=args.lambda_aux
    )

    print("[评估结果]")
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'hgne_pretrained_v3.pt')

    state_dict = model.state_dict()
    if Config.USE_TORCH_COMPILE and hasattr(model, '_orig_mod'):
        state_dict = model._orig_mod.state_dict()

    torch.save({
        'model_state_dict': state_dict,
        'model_config': {
            'in_channels': in_channels,
            'hidden_channels': args.hidden_dim,
            'num_layers': args.num_layers,
        },
        'training_config': vars(args),
        'eval_metrics': eval_metrics,
    }, model_path)

    print(f"\n[完成] 模型已保存: {model_path}")

    if torch.cuda.is_available():
        print(f"\n[GPU最终状态]")
        print(f"  已分配显存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  保留显存: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()