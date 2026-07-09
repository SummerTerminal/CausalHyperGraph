"""
train_hgne.py
================================================================================
论文: CausalHyperGraph: 超图神经网络驱动的多关系叙事推理

实现论文 3.6 节的两阶段训练策略：
    阶段1: 预训练 (20 epochs) - 超边预测
    阶段2: 微调 (50 epochs) - 问答优化

完全遵循论文配置：
    - 模型架构: 论文 3.4 节
    - 损失函数: 论文 3.6.1 和 3.6.2 节
    - 训练策略: 论文 3.6 节
    - 超参数: 论文 4.1.3 节

运行命令:
    python experiments/hgne/train_hgne.py \
        --hypergraph_dir experiments/hcm/hypergraphs \
        --output_dir checkpoints \
        --pretrain_epochs 20 \
        --finetune_epochs 50 \
        --batch_size 8 \
        --lr_pretrain 2e-5 \
        --lr_finetune 1e-4 \
        --hidden_dim 512 \
        --num_layers 3 \
        --lambda_pre 0.3 \
        --lambda_aux 0.1 \
        --eval_split 0.1 \
        --seed 42 \
        --patience 5 \
        --num_workers 2 \
        --gradient_accumulation_steps 2
================================================================================
"""

import os
import json
import argparse
import random
import time
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
from typing import Dict, Optional, List
import numpy as np
from har.question_type_infer import QuestionTypeInferer

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# ============================================================================
# GPU 全局优化
# ============================================================================

def setup_gpu_optimizations():
    """配置 GPU 训练优化"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU优化] 设备: {device_name}")
        print(f"[GPU优化] 显存: {memory_gb:.1f} GB")
        print(f"[GPU优化] cuDNN benchmark: ON")
        print(f"[GPU优化] TF32: ON")
        print(f"[GPU优化] 操作系统: {platform.system()}")
        
        torch.cuda.set_per_process_memory_fraction(0.9)
        return memory_gb
    return 0


def get_memory_usage():
    """获取当前显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'free': total - reserved,
            'utilization': (allocated / total) * 100
        }
    return {}


# ============================================================================
# 第一部分：模型定义 (论文 3.4 节)
# ============================================================================

class TypeAwareHypergraphConv(nn.Module):
    """
    类型感知超图卷积层
    论文 3.4.2 节: X^{(l+1)} = σ(Σ_{τ} α_τ · D_v^{(τ),-1/2} H^{(τ)} W_e^{(τ)} D_e^{(τ),-1} H^{(τ),T} D_v^{(τ),-1/2} X^{(l)} Θ^{(l)}_τ)
    """
    
    def __init__(self, in_channels, out_channels, num_types=3, type_names=None):
        super().__init__()
        self.num_types = num_types
        self.type_names = type_names or ['MO', 'OM', 'CO']
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 论文公式中的 Θ^{(l)}_τ: 每种类型独立的线性变换
        self.Theta = nn.ModuleList([
            nn.Linear(in_channels, out_channels) for _ in range(num_types)
        ])

        # 论文公式中的 α_τ: 类型注意力权重
        # α_τ = softmax_τ(MLP(concat(GAP(X^{(l)}), e_τ)))
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
        nn.init.normal_(self.type_embeddings, std=0.1)

    def _single_type_conv(self, x, H_tau, W_e_tau, D_v_tau, D_e_tau, theta):
        """
        单类型超图卷积
        论文公式: D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X Θ
        """
        D_v_inv_sqrt = torch.pow(D_v_tau + 1e-8, -0.5)
        D_v_inv_sqrt.clamp_(min=0)
        D_e_inv = torch.pow(D_e_tau + 1e-8, -1.0)
        D_e_inv.clamp_(min=0)

        out = theta(x)
        out = out * D_v_inv_sqrt.unsqueeze(1)
        out = torch.mm(H_tau.T, out)
        out = out * W_e_tau.unsqueeze(1)
        out = out * D_e_inv.unsqueeze(1)
        out = torch.mm(H_tau, out)
        out = out * D_v_inv_sqrt.unsqueeze(1)
        return out

    def forward(self, x, H_dict, W_e_dict, D_v_dict, D_e_dict):
        """
        Args:
            x: 节点特征 [num_nodes, in_channels]
            H_dict: 每种类型的关联矩阵 {type: [num_nodes, num_edges]}
            W_e_dict: 每种类型的超边权重 {type: [num_edges]}
            D_v_dict: 每种类型的节点度 {type: [num_nodes]}
            D_e_dict: 每种类型的超边度 {type: [num_edges]}
        
        Returns:
            out: 更新后的节点特征 [num_nodes, out_channels]
            attn_weights: 类型注意力权重
        """
        type_outputs = []
        gap = x.mean(dim=0, keepdim=True).detach()

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
            return x, torch.ones(1, device=x.device)

        # 计算类型注意力权重 α_τ
        attn_inputs = torch.cat([
            torch.cat([gap, self.type_embeddings[tau_idx].unsqueeze(0)], dim=-1)
            for tau_idx, _ in type_outputs
        ], dim=0)
        attn_weights = F.softmax(self.attention_mlp(attn_inputs).squeeze(-1), dim=0)

        # 加权聚合
        out = torch.zeros_like(type_outputs[0][1])
        for idx, (_, tau_out) in enumerate(type_outputs):
            out = out + attn_weights[idx] * tau_out

        return F.relu_(out), attn_weights


class HGNE(nn.Module):
    """
    超图神经网络编码器
    论文 3.4.3 节: 多层超图卷积 + 跳跃连接
    """
    
    def __init__(self, in_channels=1024, hidden_channels=512, num_layers=3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # 论文 3.4.1 节: L 层超图卷积
        self.convs = nn.ModuleList()
        self.convs.append(TypeAwareHypergraphConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(TypeAwareHypergraphConv(hidden_channels, hidden_channels))

        self.dropout = nn.Dropout(0.2)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])

        # 论文 3.4.3 节: 跳跃连接聚合
        # z_v = concat(x_v^{(0)}, x_v^{(1)}, ..., x_v^{(L)})
        total_dim = in_channels + num_layers * hidden_channels
        self.proj = nn.Linear(total_dim, hidden_channels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, H_dict, W_e_dict, D_v_dict, D_e_dict):
        """
        Args:
            x: 初始节点特征 [num_nodes, in_channels]
            H_dict, W_e_dict, D_v_dict, D_e_dict: 超图结构信息
        
        Returns:
            z: 最终节点表示 [num_nodes, hidden_channels]
            attention_log: 每层的类型注意力权重
        """
        x_list = [x]
        attention_log = []

        for i, conv in enumerate(self.convs):
            x_new, attn_weights = conv(x, H_dict, W_e_dict, D_v_dict, D_e_dict)
            x_new = self.layer_norms[i](x_new)
            x_new = self.dropout(x_new)
            x = x_new
            x_list.append(x_new)
            attention_log.append(attn_weights.detach())

        # 跳跃连接聚合
        z = torch.cat(x_list, dim=-1)
        z = self.proj(z)
        return z, attention_log


# ============================================================================
# 第二部分：损失函数 (论文 3.6 节)
# ============================================================================

class TypeAwareContrastiveLoss(nn.Module):
    """
    类型感知对比损失
    论文 3.6.2 节: 同类超边内节点应具有相似表示
    """
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def _compute_single_type_loss(self, z, pos_mask):
        """
        计算单类型对比损失
        论文公式: L_cont = -log(Σ_{j∈P(i)} exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ))
        """
        N = z.shape[0]
        if N < 2 or pos_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device), 0

        z_norm = F.normalize(z, dim=1)
        sim_matrix = torch.mm(z_norm, z_norm.T) / self.temperature

        # 数值稳定
        sim_max = sim_matrix.max(dim=1, keepdim=True)[0].detach()
        sim_matrix = sim_matrix - sim_max

        exp_sim = torch.exp(sim_matrix)
        exp_sim_pos = exp_sim * pos_mask.float()
        numerator = exp_sim_pos.sum(dim=1)

        neg_mask = torch.ones(N, N, device=z.device) - torch.eye(N, device=z.device)
        denominator = (exp_sim * neg_mask).sum(dim=1)

        has_pos = pos_mask.sum(dim=1) > 0
        n_valid = has_pos.sum().item()
        if n_valid == 0:
            return torch.tensor(0.0, device=z.device), 0

        ratio = numerator[has_pos] / (denominator[has_pos] + 1e-8)
        loss = -torch.log(ratio + 1e-8).mean()
        return loss, n_valid

    def forward(self, z, pos_masks):
        """
        Args:
            z: 节点嵌入 [num_nodes, hidden_dim]
            pos_masks: 每种类型的正样本mask {type: [num_nodes, num_nodes]}
        
        Returns:
            total_loss: 总对比损失
            type_losses: 每种类型的损失
        """
        total_loss = torch.tensor(0.0, device=z.device)
        type_losses = {}
        total_weight = 0.0

        # 所有类型平等对待（类型注意力已经在模型中处理）
        for tau_name, pos_mask in pos_masks.items():
            loss_tau, n_valid = self._compute_single_type_loss(z, pos_mask)
            if n_valid > 0 and not torch.isnan(loss_tau):
                total_loss = total_loss + loss_tau
                total_weight += 1.0
                type_losses[tau_name] = loss_tau.item()

        if total_weight > 0:
            total_loss = total_loss / total_weight
        return total_loss, type_losses


class HyperedgePredictionLoss(nn.Module):
    """
    超边预测损失
    论文 3.6.1 节: 预训练任务 - 判断节点集合是否构成真实超边
    """
    
    def __init__(self, hidden_dim=512, num_negatives=5):
        super().__init__()
        self.num_negatives = num_negatives
        
        # 论文公式: f(e) = MLP(HypergraphReadout(G_e))
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1)
        )

    def _sample_negative_edges(self, z, num_nodes, edge_size, num_negatives):
        """
        负采样策略：从同一视频中采样，确保不构成真实超边
        """
        device = z.device
        
        neg_edges = []
        for _ in range(num_negatives):
            indices = torch.randint(0, num_nodes, (edge_size,), device=device)
            # 确保节点不重复
            while len(torch.unique(indices)) < edge_size:
                indices = torch.randint(0, num_nodes, (edge_size,), device=device)
            neg_edges.append(indices)
        
        return neg_edges

    def forward(self, z, H_all, num_nodes):
        """
        Args:
            z: 节点嵌入 [num_nodes, hidden_dim]
            H_all: 所有超边的关联矩阵 [num_nodes, num_edges]
            num_nodes: 节点总数
        
        Returns:
            loss: 超边预测损失
        """
        E = H_all.size(1)
        if E == 0 or num_nodes < 2:
            return torch.tensor(0.0, device=z.device)

        device = z.device

        # 正样本：所有真实超边
        edge_sizes = H_all.sum(dim=0).clamp(min=1)
        edge_embeddings = (H_all.T @ z) / edge_sizes.unsqueeze(1)
        
        valid_mask = edge_sizes >= 2
        if not valid_mask.any():
            return torch.tensor(0.0, device=device)

        pos_scores = self.readout_mlp(edge_embeddings[valid_mask]).squeeze(-1)
        num_pos = pos_scores.shape[0]

        # 负样本：每个正样本生成 num_negatives 个负样本
        all_neg_scores = []
        for i in range(num_pos):
            edge_size = int(edge_sizes[valid_mask][i].item())
            neg_indices_list = self._sample_negative_edges(
                z, num_nodes, edge_size, self.num_negatives
            )
            
            for neg_indices in neg_indices_list:
                neg_embedding = z[neg_indices].mean(dim=0)
                neg_score = self.readout_mlp(neg_embedding.unsqueeze(0))
                all_neg_scores.append(neg_score)

        neg_scores = torch.cat(all_neg_scores).squeeze(-1)

        # 二分类交叉熵
        pos_labels = torch.ones(num_pos, device=device)
        neg_labels = torch.zeros(len(neg_scores), device=device)
        
        pos_expanded = pos_scores.repeat_interleave(self.num_negatives)
        
        all_scores = torch.cat([pos_expanded, neg_scores])
        all_labels = torch.cat([
            torch.ones(num_pos * self.num_negatives, device=device),
            torch.zeros(len(neg_scores), device=device)
        ])
        
        return F.binary_cross_entropy_with_logits(all_scores, all_labels)


class TypePredictionLoss(nn.Module):
    """
    类型预测损失
    论文 3.6.2 节: p(τ | q) = softmax(MLP(concat(c, e_q, e_{a_i})))
    """
    
    def __init__(self, hidden_dim=512, num_types=3):
        super().__init__()
        self.type_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_types)
        )
        self.type_names = ['MO', 'OM', 'CO']

    def forward(self, z, typed_edges, node_id_to_idx=None, question_type_label=None):
        """
        Args:
            z: 节点嵌入 [num_nodes, hidden_dim]
            typed_edges: 所有超边 {type: [edge_dict, ...]}
            node_id_to_idx: 节点ID到索引的映射
            question_type_label: 问题类型标签 (0:MO, 1:OM, 2:CO)，用于辅助损失
        
        Returns:
            loss: 类型预测交叉熵损失
        """
        all_embeddings = []
        all_labels = []

        # 使用所有超边
        for tau_idx, tau_name in enumerate(self.type_names):
            edges = typed_edges.get(tau_name, [])
            for edge in edges:
                nodes = edge.get('nodes', [])
                if len(nodes) < 2:
                    continue

                if node_id_to_idx:
                    indices = [node_id_to_idx.get(n) for n in nodes if n in node_id_to_idx]
                else:
                    continue

                if len(indices) < 2:
                    continue

                edge_embedding = z[indices].mean(dim=0)
                all_embeddings.append(edge_embedding)
                all_labels.append(tau_idx)

        if not all_embeddings:
            return torch.tensor(0.0, device=z.device)

        embeddings = torch.stack(all_embeddings)
        labels = torch.tensor(all_labels, device=z.device)
        
        # 类别不平衡加权
        class_counts = torch.bincount(labels)
        if len(class_counts) == 3:
            weights = 1.0 / (class_counts.float() + 1e-8)
            weights = weights / weights.sum() * 3
        else:
            weights = torch.ones(3, device=z.device)
        
        main_loss = F.cross_entropy(
            self.type_classifier(embeddings), 
            labels,
            weight=weights
        )
        
        # ===== 新增：辅助损失（如果有问题类型标签） =====
        if question_type_label is not None and question_type_label < 3:
            # 使用超边聚合的均值作为上下文
            context = embeddings.mean(dim=0, keepdim=True)
            logits = self.type_classifier(context)
            target = torch.tensor([question_type_label], device=z.device)
            aux_loss = F.cross_entropy(logits, target)
            
            # 返回主损失 + 辅助损失（λ₂=0.1 在外部乘以）
            return main_loss + aux_loss
        
        return main_loss



def forward(self, z, typed_edges, node_id_to_idx=None, question_type_label=None):
    """
    Args:
        z: 节点嵌入 [num_nodes, hidden_dim]
        typed_edges: 所有超边 {type: [edge_dict, ...]}
        node_id_to_idx: 节点ID到索引的映射
        question_type_label: 新增：问题类型标签 (0:MO, 1:OM, 2:CO)
    
    Returns:
        loss: 类型预测交叉熵损失
    """
    all_embeddings = []
    all_labels = []

    # 使用所有超边（论文标准）
    for tau_idx, tau_name in enumerate(self.type_names):
        edges = typed_edges.get(tau_name, [])
        for edge in edges:
            nodes = edge.get('nodes', [])
            if len(nodes) < 2:
                continue

            if node_id_to_idx:
                indices = [node_id_to_idx.get(n) for n in nodes if n in node_id_to_idx]
            else:
                continue

            if len(indices) < 2:
                continue

            # 超边读出: 聚合超边内所有节点
            edge_embedding = z[indices].mean(dim=0)
            all_embeddings.append(edge_embedding)
            all_labels.append(tau_idx)

    if not all_embeddings:
        return torch.tensor(0.0, device=z.device)

    embeddings = torch.stack(all_embeddings)
    labels = torch.tensor(all_labels, device=z.device)
    
    # 类别不平衡加权
    class_counts = torch.bincount(labels)
    if len(class_counts) == 3:
        weights = 1.0 / (class_counts.float() + 1e-8)
        weights = weights / weights.sum() * 3
    else:
        weights = torch.ones(3, device=z.device)
    
    # ===== 新增：如果有问题类型标签，加入辅助损失 =====
    if question_type_label is not None and question_type_label < 3:
        # 论文公式: p(τ | q) = softmax(MLP(concat(c, e_q, e_{a_i})))
        # 这里使用超边聚合的均值作为上下文
        context = embeddings.mean(dim=0, keepdim=True) if len(embeddings) > 0 else torch.zeros(1, z.size(-1), device=z.device)
        logits = self.type_classifier(context)  # [1, num_types]
        target = torch.tensor([question_type_label], device=z.device)
        aux_loss = F.cross_entropy(logits, target)
        
        # 主损失 + 辅助损失
        main_loss = F.cross_entropy(
            self.type_classifier(embeddings), 
            labels,
            weight=weights
        )
        return main_loss + 0.1 * aux_loss  # λ₂=0.1
    
    return F.cross_entropy(
        self.type_classifier(embeddings), 
        labels,
        weight=weights
    )

# ============================================================================
# 第三部分：数据集
# ============================================================================

class TypeAwareHypergraphDataset(Dataset):
    """类型感知超图数据集"""
    
    def __init__(self, hypergraph_dir, split='train', eval_split=0.1, seed=42):
        self.hypergraph_dir = hypergraph_dir
        self.split = split

        hypergraph_files = []
        for root, _, files in os.walk(hypergraph_dir):
            for f in files:
                if f.endswith('.json') and not f.startswith('_') and not f.startswith('.'):
                    hypergraph_files.append(os.path.join(root, f))

        print(f"[数据集] 发现 {len(hypergraph_files)} 个超图JSON文件")

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
        for file_path in hypergraph_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    hg = json.load(f)
                processed = self._process_hypergraph(hg, file_path)
                if processed is not None:
                    processed = self._precompute_gpu_data(processed)
                    self.samples.append(processed)
                    loaded_count += 1
            except Exception as e:
                print(f"[警告] 加载失败 {file_path}: {e}")

        print(f"[数据集] 成功加载 {loaded_count} 个超图")

        if self.samples:
            self._print_statistics()

    def _precompute_gpu_data(self, sample):
        """预计算可在CPU上完成的操作"""
        total_edges = sum(v.size(1) for v in sample['H_dict'].values())
        if total_edges > 0:
            H_all = torch.zeros(sample['num_nodes'], total_edges)
            global_idx = 0
            for tau_name in ['MO', 'OM', 'CO']:
                if tau_name in sample['H_dict']:
                    E_tau = sample['H_dict'][tau_name].size(1)
                    H_all[:, global_idx:global_idx + E_tau] = sample['H_dict'][tau_name]
                    global_idx += E_tau
            sample['H_all'] = H_all
        else:
            sample['H_all'] = torch.zeros(sample['num_nodes'], 1)
        
        return sample

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
                    typed_edges[e.get('type', 'CO')].append(e)
            for tau_name in ['MO', 'OM', 'CO']:
                if tau_name not in typed_edges:
                    typed_edges[tau_name] = []
        else:
            typed_edges = {'MO': [], 'OM': [], 'CO': []}

        total_edges_count = sum(len(v) for v in typed_edges.values())
        if total_edges_count == 0:
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
                for node_id in edge.get('nodes', []):
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

        pos_masks = {}
        for tau_name, H_tau in H_dict.items():
            if H_tau.size(1) > 0:
                pos_mask = (H_tau @ H_tau.T) > 0
                pos_mask.fill_diagonal_(False)
                pos_masks[tau_name] = pos_mask

        total_edges = sum(v.size(1) for v in H_dict.values())

        return {
            'x': x,
            'H_dict': H_dict,
            'W_e_dict': W_e_dict,
            'D_v_dict': D_v_dict,
            'D_e_dict': D_e_dict,
            'H_all': None,
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
        print(f"[数据集] MO: {total_mo}, OM: {total_om}, CO: {total_co}")
        print(f"[数据集] 平均节点: {avg_nodes:.1f}, 平均超边: {avg_edges:.1f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """自定义batch收集函数"""
    return batch


# ============================================================================
# 第四部分：训练 (论文 3.6 节 - 两阶段训练)
# ============================================================================

def finetune_phase(
    model, dataloader, device,
    epochs=50, lr=1e-4,
    lambda_pre=0.3, lambda_aux=0.1,
    gradient_accumulation_steps=2,
    patience=5
):
    """
    阶段2: 微调 (论文 3.6.2 节)
    任务: 问答优化 (完整损失)
    """
    print("\n" + "="*60)
    print("阶段2: 微调 - 问答优化 (论文 3.6.2 节)")
    print("="*60)
    print(f"  Epochs: {epochs}, LR: {lr}")
    print(f"  λ_pre: {lambda_pre}, λ_aux: {lambda_aux}")
    print(f"  梯度累积步数: {gradient_accumulation_steps}")
    
    # ===== 新增：创建问题类型推断器 =====
    from har.question_type_infer import QuestionTypeInferer
    type_inferer = QuestionTypeInferer()
    print("  ✅ 问题类型推断器已初始化")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    contrastive_loss = TypeAwareContrastiveLoss(temperature=0.1)
    pre_loss_fn = HyperedgePredictionLoss(hidden_dim=model.hidden_channels).to(device)
    aux_loss_fn = TypePredictionLoss(hidden_dim=model.hidden_channels).to(device)
    
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    loss_ema = None
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = defaultdict(float)
        batch_count = 0
        optimizer_idx = 0
        batch_buffer = []
        
        for sample in dataloader:
            if isinstance(sample, list):
                batch_buffer.extend(sample)
            else:
                batch_buffer.append(sample)
            
            while len(batch_buffer) >= 1:
                batch_data = batch_buffer[:1]
                batch_buffer = batch_buffer[1:]
                
                if not isinstance(batch_data[0], dict) or batch_data[0]['num_nodes'] < 3:
                    continue
                
                s = batch_data[0]
                
                # 数据转移到GPU
                x = s['x'].to(device, non_blocking=True)
                H_dict = {k: v.to(device, non_blocking=True) for k, v in s['H_dict'].items()}
                W_e_dict = {k: v.to(device, non_blocking=True) for k, v in s['W_e_dict'].items()}
                D_v_dict = {k: v.to(device, non_blocking=True) for k, v in s['D_v_dict'].items()}
                D_e_dict = {k: v.to(device, non_blocking=True) for k, v in s['D_e_dict'].items()}
                pos_masks = {k: v.to(device, non_blocking=True) for k, v in s['pos_masks'].items()}
                H_all = s['H_all'].to(device, non_blocking=True)
                
                # ===== 新增：问题类型推断 =====
                # 从 sample 中获取问题文本（如果有）
                question_text = s.get('question', '')
                type_label = None
                if question_text:
                    q_type, type_scores = type_inferer.infer_type(question_text)
                    type_label = type_inferer.get_type_label(question_text)
                    # 如果类型是 'unknown' (3)，则不使用辅助损失
                    if type_label == 3:
                        type_label = None
                
                if optimizer_idx % gradient_accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
                
                with autocast(enabled=torch.cuda.is_available()):
                    z, _ = model(x, H_dict, W_e_dict, D_v_dict, D_e_dict)
                    
                    # 论文公式: L = L_fine + λ₁·L_pre + λ₂·L_aux
                    loss_cont, type_losses = contrastive_loss(z, pos_masks)
                    loss_pre = pre_loss_fn(z, H_all, s['num_nodes'])
                    
                    # ===== 修改：传递 type_label 给 aux_loss_fn =====
                    loss_aux = aux_loss_fn(
                        z, 
                        s['typed_edges'], 
                        s['node_id_to_idx'],
                        question_type_label=type_label  # 新增参数
                    )
                    
                    loss = loss_cont + lambda_pre * loss_pre + lambda_aux * loss_aux
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    scaled_loss = loss / gradient_accumulation_steps
                    scaler.scale(scaled_loss).backward()
                    
                    epoch_losses['total'] += loss.item()
                    epoch_losses['contrastive'] += loss_cont.item()
                    epoch_losses['pre'] += loss_pre.item()
                    epoch_losses['aux'] += loss_aux.item()
                    for k, v in type_losses.items():
                        epoch_losses[f'type_{k}'] += v
                    
                    batch_count += 1
                    optimizer_idx += 1
                
                if optimizer_idx % gradient_accumulation_steps == 0 and optimizer_idx > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
        
        # 处理剩余样本
        if batch_buffer:
            optimizer.zero_grad(set_to_none=True)
            for s in batch_buffer:
                if not isinstance(s, dict) or s['num_nodes'] < 3:
                    continue
                
                x = s['x'].to(device, non_blocking=True)
                H_dict = {k: v.to(device, non_blocking=True) for k, v in s['H_dict'].items()}
                W_e_dict = {k: v.to(device, non_blocking=True) for k, v in s['W_e_dict'].items()}
                D_v_dict = {k: v.to(device, non_blocking=True) for k, v in s['D_v_dict'].items()}
                D_e_dict = {k: v.to(device, non_blocking=True) for k, v in s['D_e_dict'].items()}
                pos_masks = {k: v.to(device, non_blocking=True) for k, v in s['pos_masks'].items()}
                H_all = s['H_all'].to(device, non_blocking=True)
                
                # ===== 新增：问题类型推断（剩余样本） =====
                question_text = s.get('question', '')
                type_label = None
                if question_text:
                    q_type, type_scores = type_inferer.infer_type(question_text)
                    type_label = type_inferer.get_type_label(question_text)
                    if type_label == 3:
                        type_label = None
                
                with autocast(enabled=torch.cuda.is_available()):
                    z, _ = model(x, H_dict, W_e_dict, D_v_dict, D_e_dict)
                    loss_cont, type_losses = contrastive_loss(z, pos_masks)
                    loss_pre = pre_loss_fn(z, H_all, s['num_nodes'])
                    
                    # ===== 修改：传递 type_label =====
                    loss_aux = aux_loss_fn(
                        z, 
                        s['typed_edges'], 
                        s['node_id_to_idx'],
                        question_type_label=type_label
                    )
                    
                    loss = loss_cont + lambda_pre * loss_pre + lambda_aux * loss_aux
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    scaler.scale(loss).backward()
                    epoch_losses['total'] += loss.item()
                    epoch_losses['contrastive'] += loss_cont.item()
                    epoch_losses['pre'] += loss_pre.item()
                    epoch_losses['aux'] += loss_aux.item()
                    for k, v in type_losses.items():
                        epoch_losses[f'type_{k}'] += v
                    batch_count += 1
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        
        scheduler.step()
        
        # 计算平均值
        for k in epoch_losses:
            epoch_losses[k] /= max(batch_count, 1)
        
        avg_loss = epoch_losses['total']
        current_lr = scheduler.get_last_lr()[0]
        
        # EMA
        if loss_ema is None:
            loss_ema = avg_loss
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * avg_loss
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        mem = get_memory_usage()
        gpu_util = mem.get('utilization', 0) if mem else 0
        
        print(f"微调 Epoch {epoch+1:2d}/{epochs} | "
              f"Loss: {avg_loss:.4f} (EMA: {loss_ema:.4f}) | "
              f"Cont: {epoch_losses['contrastive']:.4f} | "
              f"Pre: {epoch_losses['pre']:.4f} | "
              f"Aux: {epoch_losses['aux']:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"GPU: {gpu_util:.1f}% | "
              f"Best: {best_loss:.4f} (Ep {best_epoch})")
        
        if patience_counter >= patience:
            print(f"\n[早停] {patience} 个 epoch 无改善，停止微调")
            break
    
    model.load_state_dict(best_model_state)
    print(f"\n[微调完成] 最佳 Loss: {best_loss:.4f} (Epoch {best_epoch})")
    
    return model


def finetune_phase(
    model, dataloader, device,
    epochs=50, lr=1e-4,
    lambda_pre=0.3, lambda_aux=0.1,
    gradient_accumulation_steps=2,
    patience=5
):
    """
    阶段2: 微调 (论文 3.6.2 节)
    任务: 问答优化 (完整损失)
    """
    print("\n" + "="*60)
    print("阶段2: 微调 - 问答优化 (论文 3.6.2 节)")
    print("="*60)
    print(f"  Epochs: {epochs}, LR: {lr}")
    print(f"  λ_pre: {lambda_pre}, λ_aux: {lambda_aux}")
    print(f"  梯度累积步数: {gradient_accumulation_steps}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    contrastive_loss = TypeAwareContrastiveLoss(temperature=0.1)
    pre_loss_fn = HyperedgePredictionLoss(hidden_dim=model.hidden_channels).to(device)
    aux_loss_fn = TypePredictionLoss(hidden_dim=model.hidden_channels).to(device)
    
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    loss_ema = None
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = defaultdict(float)
        batch_count = 0
        optimizer_idx = 0
        batch_buffer = []
        
        for sample in dataloader:
            if isinstance(sample, list):
                batch_buffer.extend(sample)
            else:
                batch_buffer.append(sample)
            
            while len(batch_buffer) >= 1:
                batch_data = batch_buffer[:1]
                batch_buffer = batch_buffer[1:]
                
                if not isinstance(batch_data[0], dict) or batch_data[0]['num_nodes'] < 3:
                    continue
                
                s = batch_data[0]
                
                x = s['x'].to(device, non_blocking=True)
                H_dict = {k: v.to(device, non_blocking=True) for k, v in s['H_dict'].items()}
                W_e_dict = {k: v.to(device, non_blocking=True) for k, v in s['W_e_dict'].items()}
                D_v_dict = {k: v.to(device, non_blocking=True) for k, v in s['D_v_dict'].items()}
                D_e_dict = {k: v.to(device, non_blocking=True) for k, v in s['D_e_dict'].items()}
                pos_masks = {k: v.to(device, non_blocking=True) for k, v in s['pos_masks'].items()}
                H_all = s['H_all'].to(device, non_blocking=True)
                
                if optimizer_idx % gradient_accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
                
                with autocast(enabled=torch.cuda.is_available()):
                    z, _ = model(x, H_dict, W_e_dict, D_v_dict, D_e_dict)
                    
                    # 论文公式: L = L_fine + λ₁·L_pre + λ₂·L_aux
                    loss_cont, type_losses = contrastive_loss(z, pos_masks)
                    loss_pre = pre_loss_fn(z, H_all, s['num_nodes'])
                    loss_aux = aux_loss_fn(z, s['typed_edges'], s['node_id_to_idx'])
                    
                    loss = loss_cont + lambda_pre * loss_pre + lambda_aux * loss_aux
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    scaled_loss = loss / gradient_accumulation_steps
                    scaler.scale(scaled_loss).backward()
                    
                    epoch_losses['total'] += loss.item()
                    epoch_losses['contrastive'] += loss_cont.item()
                    epoch_losses['pre'] += loss_pre.item()
                    epoch_losses['aux'] += loss_aux.item()
                    for k, v in type_losses.items():
                        epoch_losses[f'type_{k}'] += v
                    
                    batch_count += 1
                    optimizer_idx += 1
                
                if optimizer_idx % gradient_accumulation_steps == 0 and optimizer_idx > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
        
        # 处理剩余样本
        if batch_buffer:
            optimizer.zero_grad(set_to_none=True)
            for s in batch_buffer:
                if not isinstance(s, dict) or s['num_nodes'] < 3:
                    continue
                x = s['x'].to(device, non_blocking=True)
                H_dict = {k: v.to(device, non_blocking=True) for k, v in s['H_dict'].items()}
                W_e_dict = {k: v.to(device, non_blocking=True) for k, v in s['W_e_dict'].items()}
                D_v_dict = {k: v.to(device, non_blocking=True) for k, v in s['D_v_dict'].items()}
                D_e_dict = {k: v.to(device, non_blocking=True) for k, v in s['D_e_dict'].items()}
                pos_masks = {k: v.to(device, non_blocking=True) for k, v in s['pos_masks'].items()}
                H_all = s['H_all'].to(device, non_blocking=True)
                
                with autocast(enabled=torch.cuda.is_available()):
                    z, _ = model(x, H_dict, W_e_dict, D_v_dict, D_e_dict)
                    loss_cont, type_losses = contrastive_loss(z, pos_masks)
                    loss_pre = pre_loss_fn(z, H_all, s['num_nodes'])
                    loss_aux = aux_loss_fn(z, s['typed_edges'], s['node_id_to_idx'])
                    loss = loss_cont + lambda_pre * loss_pre + lambda_aux * loss_aux
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    scaler.scale(loss).backward()
                    epoch_losses['total'] += loss.item()
                    epoch_losses['contrastive'] += loss_cont.item()
                    epoch_losses['pre'] += loss_pre.item()
                    epoch_losses['aux'] += loss_aux.item()
                    for k, v in type_losses.items():
                        epoch_losses[f'type_{k}'] += v
                    batch_count += 1
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        
        scheduler.step()
        
        # 计算平均值
        for k in epoch_losses:
            epoch_losses[k] /= max(batch_count, 1)
        
        avg_loss = epoch_losses['total']
        current_lr = scheduler.get_last_lr()[0]
        
        # EMA
        if loss_ema is None:
            loss_ema = avg_loss
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * avg_loss
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        mem = get_memory_usage()
        gpu_util = mem.get('utilization', 0) if mem else 0
        
        print(f"微调 Epoch {epoch+1:2d}/{epochs} | "
              f"Loss: {avg_loss:.4f} (EMA: {loss_ema:.4f}) | "
              f"Cont: {epoch_losses['contrastive']:.4f} | "
              f"Pre: {epoch_losses['pre']:.4f} | "
              f"Aux: {epoch_losses['aux']:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"GPU: {gpu_util:.1f}% | "
              f"Best: {best_loss:.4f} (Ep {best_epoch})")
        
        if patience_counter >= patience:
            print(f"\n[早停] {patience} 个 epoch 无改善，停止微调")
            break
    
    model.load_state_dict(best_model_state)
    print(f"\n[微调完成] 最佳 Loss: {best_loss:.4f} (Epoch {best_epoch})")
    
    return model




# ============================================================================
# 第五部分：评估
# ============================================================================

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """评估模型"""
    model.eval()
    
    contrastive_loss = TypeAwareContrastiveLoss(temperature=0.1)
    pre_loss_fn = HyperedgePredictionLoss(hidden_dim=model.hidden_channels).to(device).eval()
    aux_loss_fn = TypePredictionLoss(hidden_dim=model.hidden_channels).to(device).eval()
    
    metrics = defaultdict(float)
    valid_count = 0
    
    for sample in dataloader:
        if isinstance(sample, list):
            sample = sample[0] if sample else None
        if not isinstance(sample, dict) or sample['num_nodes'] < 3:
            continue
        
        x = sample['x'].to(device, non_blocking=True)
        H_dict = {k: v.to(device, non_blocking=True) for k, v in sample['H_dict'].items()}
        W_e_dict = {k: v.to(device, non_blocking=True) for k, v in sample['W_e_dict'].items()}
        D_v_dict = {k: v.to(device, non_blocking=True) for k, v in sample['D_v_dict'].items()}
        D_e_dict = {k: v.to(device, non_blocking=True) for k, v in sample['D_e_dict'].items()}
        pos_masks = {k: v.to(device, non_blocking=True) for k, v in sample['pos_masks'].items()}
        H_all = sample['H_all'].to(device, non_blocking=True)
        
        with autocast(enabled=torch.cuda.is_available()):
            z, attn_log = model(x, H_dict, W_e_dict, D_v_dict, D_e_dict)
            
            # 对比损失
            loss_cont, _ = contrastive_loss(z, pos_masks)
            metrics['contrastive_loss'] += loss_cont.item()
            
            # 超边预测准确率
            E = H_all.size(1)
            if E > 0:
                edge_sizes = H_all.sum(dim=0)
                valid_edges = edge_sizes >= 2
                if valid_edges.any():
                    edge_emb = (H_all.T @ z) / edge_sizes.unsqueeze(1).clamp(min=1)
                    edge_emb_valid = edge_emb[valid_edges]
                    edge_norm = F.normalize(edge_emb_valid, dim=1)
                    
                    neg_indices = torch.randint(0, sample['num_nodes'],
                                                (edge_emb_valid.shape[0],), device=device)
                    neg_emb = z[neg_indices]
                    neg_norm = F.normalize(neg_emb, dim=1)
                    
                    sim_pos = (edge_norm * edge_norm).sum(dim=1)
                    sim_neg = (edge_norm * neg_norm).sum(dim=1)
                    metrics['edge_prediction_acc'] += (sim_pos > sim_neg).float().mean().item()
            
            # 类型预测准确率
            if sample.get('typed_edges'):
                all_emb = []
                all_lbl = []
                for tau_idx, tau_name in enumerate(['MO', 'OM', 'CO']):
                    for edge in sample['typed_edges'].get(tau_name, []):
                        nodes = edge.get('nodes', [])
                        if len(nodes) < 2:
                            continue
                        indices = [sample['node_id_to_idx'].get(n) for n in nodes
                                  if n in sample['node_id_to_idx']]
                        if len(indices) < 2:
                            continue
                        all_emb.append(z[indices].mean(dim=0))
                        all_lbl.append(tau_idx)
                
                if all_emb:
                    emb_tensor = torch.stack(all_emb)
                    lbl_tensor = torch.tensor(all_lbl, device=device)
                    logits = aux_loss_fn.type_classifier(emb_tensor)
                    preds = logits.argmax(dim=1)
                    metrics['type_prediction_acc'] += (preds == lbl_tensor).float().mean().item()
            
            # 正样本余弦相似度
            sim_sum = 0.0
            sim_count = 0
            for pos_mask in pos_masks.values():
                if pos_mask.sum() > 0:
                    z_norm = F.normalize(z, dim=1)
                    sim = torch.mm(z_norm, z_norm.T)
                    sim_sum += sim[pos_mask].mean().item()
                    sim_count += 1
            if sim_count > 0:
                metrics['positive_cosine_sim'] += sim_sum / sim_count
            
            # 注意力权重
            if attn_log:
                avg_attn = torch.stack(attn_log).mean(dim=0)
                for i, name in enumerate(['MO', 'OM', 'CO']):
                    if i < len(avg_attn):
                        metrics[f'attn_{name}'] += avg_attn[i].item()
        
        valid_count += 1
    
    for k in metrics:
        metrics[k] /= max(valid_count, 1)
    metrics['num_samples'] = valid_count
    return dict(metrics)


# ============================================================================
# 第六部分：主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CausalHyperGraph HGNE - 完全按照论文 3.6 节'
    )
    
    # 输入输出
    parser.add_argument('--hypergraph_dir', required=True,
                        help='超图JSON文件目录')
    parser.add_argument('--output_dir', default='checkpoints',
                        help='输出目录')
    
    # 训练参数 (论文 4.1.3 节)
    parser.add_argument('--pretrain_epochs', type=int, default=20,
                        help='预训练 epoch 数 (论文: 20)')
    parser.add_argument('--finetune_epochs', type=int, default=50,
                        help='微调 epoch 数 (论文: 50)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小 (论文: 32)')
    parser.add_argument('--lr_pretrain', type=float, default=2e-5,
                        help='预训练学习率 (论文: 2e-5)')
    parser.add_argument('--lr_finetune', type=float, default=1e-4,
                        help='微调学习率 (论文: 1e-4)')
    
    # 模型参数 (论文 4.1.3 节)
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='隐藏层维度 (论文: 512)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='超图卷积层数 (论文: 3)')
    
    # 损失权重 (论文 4.1.3 节)
    parser.add_argument('--lambda_pre', type=float, default=0.3,
                        help='超边预测损失权重 (论文: 0.3)')
    parser.add_argument('--lambda_aux', type=float, default=0.1,
                        help='类型预测损失权重 (论文: 0.1)')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--eval_split', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--patience', type=int, default=5,
                        help='早停 patience')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='数据加载进程数')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='梯度累积步数')
    
    args = parser.parse_args()
    
    # GPU 优化
    setup_gpu_optimizations()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    train_dataset = TypeAwareHypergraphDataset(
        args.hypergraph_dir, split='train', eval_split=args.eval_split, seed=args.seed
    )
    eval_dataset = TypeAwareHypergraphDataset(
        args.hypergraph_dir, split='eval', eval_split=args.eval_split, seed=args.seed
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=True, prefetch_factor=2, persistent_workers=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    # 创建模型
    in_channels = train_dataset.samples[0]['x'].shape[1] if train_dataset.samples else 1024
    model = HGNE(
        in_channels=in_channels, 
        hidden_channels=args.hidden_dim, 
        num_layers=args.num_layers
    )
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[模型] 参数量: {total_params:,}")
    print(f"[模型] 架构: {args.hidden_dim} hidden, {args.num_layers} layers")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # ============================================================
    # 阶段1: 预训练
    # ============================================================
    model = pretrain_phase(
        model=model,
        dataloader=train_dataloader,
        device=device,
        epochs=args.pretrain_epochs,
        lr=args.lr_pretrain,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        patience=args.patience
    )
    
    # 保存预训练模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'in_channels': in_channels,
            'hidden_channels': args.hidden_dim,
            'num_layers': args.num_layers,
        },
        'phase': 'pretrained'
    }, os.path.join(args.output_dir, 'hgne_pretrained.pt'))
    
    # ============================================================
    # 阶段2: 微调
    # ============================================================
    model = finetune_phase(
        model=model,
        dataloader=train_dataloader,
        device=device,
        epochs=args.finetune_epochs,
        lr=args.lr_finetune,
        lambda_pre=args.lambda_pre,
        lambda_aux=args.lambda_aux,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        patience=args.patience
    )
    
    # 保存微调模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'in_channels': in_channels,
            'hidden_channels': args.hidden_dim,
            'num_layers': args.num_layers,
        },
        'phase': 'finetuned',
        'training_config': vars(args)
    }, os.path.join(args.output_dir, 'hgne_finetuned.pt'))
    
    # ============================================================
    # 评估
    # ============================================================
    print("\n" + "="*60)
    print("评估模型...")
    print("="*60 + "\n")
    
    eval_metrics = evaluate_model(model, eval_dataloader, device)
    
    print("[评估结果]")
    for k, v in eval_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    
    # 保存评估结果
    with open(os.path.join(args.output_dir, 'eval_metrics.json'), 'w') as f:
        json.dump(eval_metrics, f, indent=2)
    
    print(f"\n[完成] 模型: {args.output_dir}/hgne_finetuned.pt")
    print(f"[完成] 评估: {args.output_dir}/eval_metrics.json")


if __name__ == '__main__':
    main()