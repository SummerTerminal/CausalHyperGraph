"""
hypergraph_readout.py - 超图读出模块
实现论文 3.5.3 节的超图读出功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


class HypergraphReadout(nn.Module):
    """
    超图读出模块
    论文公式: c = HypergraphReadout(G_sub) = Σ_{v∈V_sub} β_v · z_v
    
    支持三种读出策略：
    1. 注意力池化 (Attention Pooling) - 默认
    2. 最大池化 (Max Pooling)
    3. 平均池化 (Mean Pooling)
    """
    
    def __init__(self, hidden_dim: int = 512, readout_type: str = 'attention'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.readout_type = readout_type
        
        if readout_type == 'attention':
            # 注意力权重计算: β_v = softmax_v(e_q^T W_read z_v)
            self.W_read = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.xavier_uniform_(self.W_read.weight)
        elif readout_type == 'gated':
            # 门控注意力
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            self.W_read = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.xavier_uniform_(self.W_read.weight)
    
    def forward(
        self, 
        z_sub: torch.Tensor,           # [num_sub_nodes, hidden_dim]
        q_proj: torch.Tensor,          # [1, hidden_dim] 或 [hidden_dim]
        node_indices: Optional[List[int]] = None,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            z_sub: 子超图节点嵌入
            q_proj: 投影后的问题编码
            node_indices: 节点索引列表
            return_weights: 是否返回注意力权重
            
        Returns:
            context: 上下文向量 [hidden_dim]
            weights: 注意力权重 [num_sub_nodes] (如果 return_weights=True)
        """
        if q_proj.dim() == 1:
            q_proj = q_proj.unsqueeze(0)  # [1, hidden_dim]
        
        num_nodes = z_sub.shape[0]
        if num_nodes == 0:
            if return_weights:
                return torch.zeros(self.hidden_dim, device=z_sub.device), None
            return torch.zeros(self.hidden_dim, device=z_sub.device)
        
        if self.readout_type == 'attention':
            # β_v = softmax_v(e_q^T W_read z_v)
            # 论文公式: β_v = softmax_v(e_q^T W_read z_v)
            projected = self.W_read(z_sub)  # [num_nodes, hidden_dim]
            scores = torch.mm(q_proj, projected.T) / math.sqrt(self.hidden_dim)  # [1, num_nodes]
            beta = F.softmax(scores, dim=1)  # [1, num_nodes]
            
        elif self.readout_type == 'max':
            # 最大池化
            beta = None
            context = z_sub.max(dim=0)[0]
            if return_weights:
                return context, None
            return context
            
        elif self.readout_type == 'mean':
            # 平均池化
            beta = None
            context = z_sub.mean(dim=0)
            if return_weights:
                return context, None
            return context
            
        elif self.readout_type == 'gated':
            # 门控注意力
            projected = self.W_read(z_sub)  # [num_nodes, hidden_dim]
            q_expanded = q_proj.expand(num_nodes, -1)  # [num_nodes, hidden_dim]
            gate_input = torch.cat([projected, q_expanded], dim=-1)  # [num_nodes, 2*hidden_dim]
            gate_weights = self.gate(gate_input)  # [num_nodes, hidden_dim]
            scores = torch.mm(q_proj, projected.T) / math.sqrt(self.hidden_dim)
            beta = F.softmax(scores, dim=1) * gate_weights.mean(dim=-1, keepdim=True)
            beta = F.softmax(beta, dim=1)
        
        # 计算上下文向量: c = Σ β_v · z_v
        context = torch.mm(beta, z_sub).squeeze(0)  # [hidden_dim]
        
        if return_weights:
            return context, beta.squeeze(0)
        return context, None


class HierarchicalHypergraphReadout(nn.Module):
    """
    层次化超图读出
    论文扩展：先按超边聚合，再按节点聚合
    """
    
    def __init__(self, hidden_dim: int = 512, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 超边级注意力
        self.edge_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )
        
        # 节点级读出
        self.node_readout = HypergraphReadout(hidden_dim, 'attention')
    
    def forward(
        self,
        z_sub: torch.Tensor,
        q_proj: torch.Tensor,
        hyperedge_nodes: List[List[int]],
        node_to_idx: Dict[str, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_sub: 子超图节点嵌入 [num_sub_nodes, hidden_dim]
            q_proj: 问题投影 [1, hidden_dim]
            hyperedge_nodes: 每条超边的节点索引列表
            node_to_idx: 节点ID到索引的映射
            
        Returns:
            context: 最终上下文向量 [hidden_dim]
            edge_weights: 超边级注意力权重
            node_weights: 节点级注意力权重
        """
        if q_proj.dim() == 1:
            q_proj = q_proj.unsqueeze(0)
        
        # Step 1: 聚合每条超边内的节点
        edge_embeddings = []
        valid_edge_indices = []
        
        for edge_idx, node_ids in enumerate(hyperedge_nodes):
            valid_nodes = [node_to_idx[nid] for nid in node_ids if nid in node_to_idx]
            if len(valid_nodes) < 2:
                continue
            edge_emb = z_sub[valid_nodes].mean(dim=0)  # [hidden_dim]
            edge_embeddings.append(edge_emb)
            valid_edge_indices.append(edge_idx)
        
        if not edge_embeddings:
            # 无有效超边，直接读出节点
            context, node_weights = self.node_readout(z_sub, q_proj, return_weights=True)
            return context, torch.tensor([1.0]), node_weights
        
        edge_embs = torch.stack(edge_embeddings)  # [num_edges, hidden_dim]
        
        # Step 2: 超边级注意力
        # 论文扩展: β_e = softmax_e(q^T W_edge e_emb)
        attn_output, edge_weights = self.edge_attention(
            query=q_proj.unsqueeze(1),  # [1, 1, hidden_dim]
            key=edge_embs.unsqueeze(0),  # [1, num_edges, hidden_dim]
            value=edge_embs.unsqueeze(0),
            need_weights=True
        )
        
        edge_context = attn_output.squeeze(1)  # [1, hidden_dim]
        edge_weights = edge_weights.squeeze(0)  # [num_edges]
        
        # Step 3: 节点级注意力（在最重要的超边上）
        top_edge_idx = edge_weights.argmax().item()
        top_nodes = hyperedge_nodes[valid_edge_indices[top_edge_idx]]
        valid_top_nodes = [node_to_idx[nid] for nid in top_nodes if nid in node_to_idx]
        z_top = z_sub[valid_top_nodes]
        
        context, node_weights = self.node_readout(z_top, q_proj, return_weights=True)
        
        # 融合超边级和节点级上下文
        final_context = edge_context.squeeze(0) + context
        final_context = F.normalize(final_context, dim=0)
        
        return final_context, edge_weights, node_weights


class HyperedgeAwareReasoner(nn.Module):
    """
    超边感知推理器
    整合论文 3.5.2 和 3.5.3 节
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        top_k_seed: int = 10,
        M: int = 5,
        readout_type: str = 'attention'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k_seed = top_k_seed
        self.M = M
        
        self.readout = HypergraphReadout(hidden_dim, readout_type)
        self.hierarchical_readout = HierarchicalHypergraphReadout(hidden_dim)
        
        # 超边-问题匹配度计算器
        self.edge_matcher = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def compute_hyperedge_relevance(
        self,
        edge_nodes: List[int],
        z: torch.Tensor,
        q_proj: torch.Tensor,
        H_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算超边与问题的匹配度
        论文公式: m(e, q) = attention(e_q, aggregate(z_v | v ∈ e))
        
        Args:
            edge_nodes: 超边包含的节点索引
            z: 所有节点嵌入
            q_proj: 问题投影
            H_dict: 类型特定的关联矩阵
            
        Returns:
            relevance: 匹配度标量
        """
        if len(edge_nodes) < 2:
            return torch.tensor(0.0, device=z.device)
        
        # 聚合超边内节点
        edge_emb = z[edge_nodes].mean(dim=0, keepdim=True)  # [1, hidden_dim]
        
        # 计算匹配度
        combined = torch.cat([edge_emb, q_proj], dim=-1)  # [1, 2*hidden_dim]
        score = self.edge_matcher(combined).squeeze()  # []
        
        return torch.sigmoid(score)
    
    def retrieve_subgraph(
        self,
        z: torch.Tensor,
        q_proj: torch.Tensor,
        hypergraph: Dict,
        H_dict: Dict[str, torch.Tensor],
        node_to_idx: Dict[str, int]
    ) -> Tuple[torch.Tensor, List[int], List[List[int]], Dict[str, torch.Tensor]]:
        """
        超边感知检索
        论文 3.5.2 节：种子节点 → 超边扩展 → 子超图
        
        Args:
            z: 节点嵌入 [num_nodes, hidden_dim]
            q_proj: 问题投影 [1, hidden_dim]
            hypergraph: 超图数据
            H_dict: 类型感知关联矩阵
            node_to_idx: 节点ID到索引的映射
            
        Returns:
            z_sub: 子超图节点嵌入
            sub_node_indices: 子超图节点索引
            sub_edges: 子超图超边
            edge_relevance: 每条超边的相关度
        """
        num_nodes = z.shape[0]
        
        # ===== 步骤一：种子节点检索 =====
        # 基于语义相似度
        similarities = F.cosine_similarity(q_proj, z, dim=1)  # [num_nodes]
        top_k = min(self.top_k_seed, num_nodes)
        seed_indices = torch.topk(similarities, top_k).indices.tolist()
        seed_set = set(seed_indices)
        
        # ===== 步骤二：超边扩展 =====
        all_edges = hypergraph.get('hyperedges', [])
        if isinstance(all_edges, dict):
            flat_edges = []
            for tau_name, edge_list in all_edges.items():
                for edge in edge_list:
                    edge_copy = edge.copy()
                    edge_copy['type'] = tau_name
                    flat_edges.append(edge_copy)
            all_edges = flat_edges
        
        # 收集包含种子节点的超边
        candidate_edges = []
        edge_relevance_scores = []
        
        for edge in all_edges:
            edge_nodes = edge.get('nodes', [])
            # 转换为索引
            edge_indices = [node_to_idx[nid] for nid in edge_nodes if nid in node_to_idx]
            if len(edge_indices) < 2:
                continue
            
            # 检查是否包含种子节点
            if not (set(edge_indices) & seed_set):
                continue
            
            # 计算超边-问题匹配度
            relevance = self.compute_hyperedge_relevance(
                edge_indices, z, q_proj, H_dict
            )
            candidate_edges.append(edge_indices)
            edge_relevance_scores.append(relevance.item())
        
        # 选择Top-M条相关超边
        if len(candidate_edges) == 0:
            # 无相关超边，回退到种子节点
            z_sub = z[seed_indices]
            return z_sub, seed_indices, [], {}
        
        # 按相关度排序
        sorted_pairs = sorted(
            zip(candidate_edges, edge_relevance_scores),
            key=lambda x: x[1],
            reverse=True
        )
        top_edges = sorted_pairs[:self.M]
        
        # ===== 步骤三：子超图构建 =====
        sub_node_set = set()
        sub_edge_list = []
        edge_relevance_dict = {}
        
        for edge_indices, score in top_edges:
            sub_node_set.update(edge_indices)
            sub_edge_list.append(edge_indices)
            edge_relevance_dict[str(edge_indices)] = score
        
        sub_node_indices = list(sub_node_set)
        z_sub = z[sub_node_indices]
        
        # 构建类型感知的子超图张量
        sub_H_dict = {}
        for tau_name, H_tau in H_dict.items():
            # 过滤子超图中的超边
            sub_nodes_in_H = H_tau[sub_node_indices] if len(sub_node_indices) > 0 else torch.zeros(0, H_tau.size(1))
            sub_H_dict[tau_name] = sub_nodes_in_H
        
        return z_sub, sub_node_indices, sub_edge_list, edge_relevance_dict, sub_H_dict
    
    def forward(
        self,
        z: torch.Tensor,
        q_proj: torch.Tensor,
        hypergraph: Dict,
        H_dict: Dict[str, torch.Tensor],
        node_to_idx: Dict[str, int],
        use_hierarchical: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        完整的推理过程
        
        Returns:
            context: 上下文表示 [hidden_dim]
            info: 包含检索和注意力信息的字典
        """
        # 检索子超图
        z_sub, sub_indices, sub_edges, edge_relevance, sub_H_dict = self.retrieve_subgraph(
            z, q_proj, hypergraph, H_dict, node_to_idx
        )
        
        info = {
            'sub_indices': sub_indices,
            'sub_edges': sub_edges,
            'edge_relevance': edge_relevance,
            'num_sub_nodes': len(sub_indices),
            'num_sub_edges': len(sub_edges)
        }
        
        # 超图读出
        if use_hierarchical and len(sub_edges) > 0:
            context, edge_weights, node_weights = self.hierarchical_readout(
                z_sub, q_proj, sub_edges, node_to_idx
            )
            info['edge_weights'] = edge_weights.tolist() if edge_weights is not None else []
            info['node_weights'] = node_weights.tolist() if node_weights is not None else []
        else:
            context, node_weights = self.readout(z_sub, q_proj, return_weights=True)
            info['node_weights'] = node_weights.tolist() if node_weights is not None else []
            info['edge_weights'] = []
        
        return context, info


# ===== 工厂函数 =====

def create_reasoner(
    hidden_dim: int = 512,
    top_k_seed: int = 10,
    M: int = 5,
    readout_type: str = 'attention'
) -> HyperedgeAwareReasoner:
    """创建超边感知推理器"""
    return HyperedgeAwareReasoner(
        hidden_dim=hidden_dim,
        top_k_seed=top_k_seed,
        M=M,
        readout_type=readout_type
    )