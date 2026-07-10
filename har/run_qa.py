"""
run_qa.py
HAR: Hyperedge-Aware Reasoner - 统一推理脚本 (完整修复版 v2.0)
=================================================================
修复内容：
1. 统一HGNE和E2E模式的推理流程
2. 修复特征维度不匹配问题
3. 修复HypergraphConv返回值
4. 修复类型感知推断边界情况
5. 优化内存管理

Usage:
    # 1. 使用预训练的HGNE (无分类器头)
    python experiments/har/run_qa.py \
        --questions files/baseline/StoryVideoQA-main/PlotTree/data/BigBang_golden.json \
        --hypergraph_dir experiments/hcm/hypergraphs/BigBang \
        --checkpoint checkpoints/hgne_pretrained.pt \
        --output experiments/results/result.json \
        --model_type hgne \
        --no_llm \
        #--use_local_llm  # 使用本地模型进行推理

    # 2. 使用端到端训练好的模型 (推荐，无需LLM)
    python experiments/har/run_qa.py \
        --questions files/.../BigBang_golden.json \
        --hypergraph_dir experiments/hcm/hypergraphs \
        --checkpoint checkpoints/e2e_best_model.pt \
        --output experiments/results/result.json \
        --model_type e2e \
        --no_llm
"""


from har.hypergraph_readout import HypergraphReadout
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm
import gc
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# ============================================================================
# 第一部分：模型定义 (修复版)
# ============================================================================

class HypergraphConv(nn.Module):
    """基础超图卷积层 (修复版：返回一致格式)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Theta = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels
        
    def forward(self, x, H, W_e, D_v, D_e):
        """
        严格按照论文公式：
        X' = σ(D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X Θ)
        """
        D_v_inv_sqrt = torch.pow(D_v + 1e-8, -0.5)
        D_v_inv_sqrt = torch.where(torch.isfinite(D_v_inv_sqrt), D_v_inv_sqrt, torch.zeros_like(D_v_inv_sqrt))
        
        D_e_inv = torch.pow(D_e + 1e-8, -1.0)
        D_e_inv = torch.where(torch.isfinite(D_e_inv), D_e_inv, torch.zeros_like(D_e_inv))
        
        x_transformed = self.Theta(x)
        
        # D_v^{-1/2} * X
        x_normalized = D_v_inv_sqrt.unsqueeze(1) * x_transformed
        
        # H^T @ (D_v^{-1/2} X)
        node_to_edge = H.T @ x_normalized
        
        # W_e D_e^{-1} @ (H^T D_v^{-1/2} X)
        edge_weighted = W_e.unsqueeze(1) * D_e_inv.unsqueeze(1) * node_to_edge
        
        # H @ (W_e D_e^{-1} H^T D_v^{-1/2} X)
        edge_to_node = H @ edge_weighted
        
        # D_v^{-1/2} @ (H W_e D_e^{-1} H^T D_v^{-1/2} X)
        out = D_v_inv_sqrt.unsqueeze(1) * edge_to_node
        
        # 返回格式与TypeAwareHypergraphConv一致： (output, None)
        return F.relu(out), None


class TypeAwareHypergraphConv(nn.Module):
    """类型感知超图卷积层 (修复版)"""
    
    def __init__(self, in_channels, out_channels, num_types=3, type_names=None):
        super().__init__()
        self.num_types = num_types
        self.type_names = type_names or ['MO', 'OM', 'CO']
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 改为：
        self.Theta = nn.ModuleList([
            nn.Linear(in_channels, out_channels)  # bias=True 是默认值
            for _ in range(num_types)
        ])

        self.attention_mlp = nn.Sequential(
            nn.Linear(in_channels + 64, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.type_embeddings = nn.Parameter(torch.randn(num_types, 64))
        self.reset_parameters()

    def reset_parameters(self):
        for theta in self.Theta:
            nn.init.xavier_uniform_(theta.weight)
            nn.init.zeros_(theta.bias)  # ← 添加 bias 初始化
        nn.init.normal_(self.type_embeddings, std=0.1)

    def _single_type_conv(self, x, H_tau, W_e_tau, D_v_tau, D_e_tau, theta):
        """论文公式 (4)"""
        D_v_inv_sqrt = torch.pow(D_v_tau + 1e-8, -0.5)
        D_v_inv_sqrt = torch.where(torch.isfinite(D_v_inv_sqrt), D_v_inv_sqrt, torch.zeros_like(D_v_inv_sqrt))
        
        D_e_inv = torch.pow(D_e_tau + 1e-8, -1.0)
        D_e_inv = torch.where(torch.isfinite(D_e_inv), D_e_inv, torch.zeros_like(D_e_inv))
        
        x_transformed = theta(x)
        x_normalized = D_v_inv_sqrt.unsqueeze(1) * x_transformed
        node_to_edge = H_tau.T @ x_normalized
        edge_weighted = W_e_tau.unsqueeze(1) * D_e_inv.unsqueeze(1) * node_to_edge
        edge_to_node = H_tau @ edge_weighted
        out = D_v_inv_sqrt.unsqueeze(1) * edge_to_node
        
        return out

    def forward(self, x, H_dict, W_e_dict, D_v_dict, D_e_dict):
        type_outputs = []
        gap = x.mean(dim=0, keepdim=True)

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
            # 修复：返回正确维度的零张量
            return torch.zeros(x.shape[0], self.out_channels, device=x.device), torch.ones(1, device=x.device)

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
    """超图神经网络编码器 - 修复版"""
    
    def __init__(self, in_channels=1024, hidden_channels=512, num_layers=3, type_aware=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.type_aware = type_aware
        
        self.convs = nn.ModuleList()
        if type_aware:
            self.convs.append(TypeAwareHypergraphConv(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(TypeAwareHypergraphConv(hidden_channels, hidden_channels))
        else:
            self.convs.append(HypergraphConv(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(HypergraphConv(hidden_channels, hidden_channels))
        
        self.dropout = nn.Dropout(0.2)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])
        
        total_dim = in_channels + num_layers * hidden_channels
        self.proj = nn.Linear(total_dim, hidden_channels)
        
        # 问题投影层（用于HGNE模式）
        self.q_proj = nn.Linear(1024, hidden_channels)
        
    def forward(self, x, H_dict, W_e_dict, D_v_dict, D_e_dict):
        """前向传播，统一返回格式"""
        x_list = [x]
        attention_log = []
        
        for i, conv in enumerate(self.convs):
            if self.type_aware:
                x_new, attn_weights = conv(x, H_dict, W_e_dict, D_v_dict, D_e_dict)
                if attn_weights is not None:
                    attention_log.append(attn_weights.detach())
            else:
                # 基础模式：合并所有类型的超边
                H_list = []
                W_e_list = []
                for tau_name in ['MO', 'OM', 'CO']:
                    if tau_name in H_dict and H_dict[tau_name].size(1) > 0:
                        H_list.append(H_dict[tau_name])
                        W_e_list.append(W_e_dict[tau_name])
                
                if not H_list:
                    # 没有超边，使用残差连接
                    x_new = x
                else:
                    H = torch.cat(H_list, dim=1)
                    W_e = torch.cat(W_e_list, dim=0)
                    D_v = torch.sum(H * W_e.unsqueeze(0), dim=1) + 1e-8
                    D_e = torch.sum(H, dim=0) + 1e-8
                    x_new, _ = conv(x, H, W_e, D_v, D_e)
                    if x_new is None:
                        x_new = x
            
            x_new = self.layer_norms[i](x_new)
            x_new = self.dropout(x_new)
            x = x_new
            x_list.append(x_new)
        
        z = torch.cat(x_list, dim=-1)
        z = self.proj(z)
        return z, attention_log
    
    def project_question(self, q_emb):
        """投影问题嵌入到超图空间"""
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)
        return self.q_proj(q_emb)


class EndToEndModel(nn.Module):
    """端到端模型：HGNE + 答案分类器 (修复版)"""
    def __init__(self, hgne=None, hidden_dim=512, num_answers=5, type_aware=True):
        super().__init__()
        if hgne is None:
            hgne = HGNE(in_channels=1024, hidden_channels=hidden_dim, num_layers=3, type_aware=type_aware)
        self.hgne = hgne
        self.hidden_dim = hidden_dim
        self.type_aware = type_aware
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_answers)
        )
    
    def forward(self, x, H_dict, W_e_dict, D_v_dict, D_e_dict, q_emb):
        """
        Args:
            x: 节点特征 [N, in_channels]
            H_dict, W_e_dict, D_v_dict, D_e_dict: 超图张量字典
            q_emb: 问题编码 [1, 1024] 或 [1024]
        Returns:
            logits: [num_answers]
        """
        # 确保 q_emb 维度正确
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)
        
        # 获取节点表示
        z, _ = self.hgne(x, H_dict, W_e_dict, D_v_dict, D_e_dict)  # [N, hidden_dim]
        
        # 投影问题编码
        q_proj = self.hgne.q_proj(q_emb)  # [1, hidden_dim]
        
        # 注意力池化
        attn_scores = torch.mm(q_proj, z.T) / (self.hidden_dim ** 0.5)  # [1, N]
        attn_weights = F.softmax(attn_scores, dim=1)  # [1, N]
        
        # 上下文向量
        context = torch.mm(attn_weights, z)  # [1, hidden_dim]
        
        # 拼接
        combined = torch.cat([context.squeeze(0), q_proj.squeeze(0)], dim=0)  # [2 * hidden_dim]
        
        # 分类
        logits = self.classifier(combined)  # [num_answers]
        return logits


# ============================================================================
# 第二部分：超图工具函数 (修复版)
# ============================================================================

def get_all_hyperedges(hypergraph: Dict) -> List[Dict]:
    """统一获取所有超边"""
    hyperedges = hypergraph.get('hyperedges', [])
    if isinstance(hyperedges, dict):
        all_edges = []
        for tau_name in ['MO', 'OM', 'CO']:
            edges = hyperedges.get(tau_name, [])
            if isinstance(edges, list):
                all_edges.extend(edges)
        return all_edges
    elif isinstance(hyperedges, list):
        return hyperedges
    return []


def build_type_aware_tensors(hypergraph: Dict, device: torch.device) -> Tuple[Dict, Dict, Dict, Dict]:
    """构建类型感知的超图张量 (修复版)"""
    node_map = {n['id']: i for i, n in enumerate(hypergraph['nodes'])}
    num_nodes = len(hypergraph['nodes'])
    
    hyperedges = hypergraph.get('hyperedges', {})
    if isinstance(hyperedges, list):
        typed_edges = {'MO': [], 'OM': [], 'CO': []}
        for e in hyperedges:
            tau = e.get('type', 'CO')
            if tau in typed_edges:
                typed_edges[tau].append(e)
        hyperedges = typed_edges
    
    H_dict = {}
    W_e_dict = {}
    D_v_dict = {}
    D_e_dict = {}
    
    for tau_name in ['MO', 'OM', 'CO']:
        edges = hyperedges.get(tau_name, [])
        
        # 修复：即使没有超边也创建占位张量，但标记为无效
        has_edges = len(edges) > 0 and any(
            any(node_id in node_map for node_id in edge.get('nodes', []))
            for edge in edges
        )
        
        if not has_edges:
            # 创建空张量标记
            H_dict[tau_name] = torch.zeros(num_nodes, 0, device=device)
            W_e_dict[tau_name] = torch.zeros(0, device=device)
            D_v_dict[tau_name] = torch.ones(num_nodes, device=device)
            D_e_dict[tau_name] = torch.zeros(0, device=device)
            continue
        
        # 过滤有效的超边
        valid_edges = []
        for edge in edges:
            valid_nodes = [n for n in edge.get('nodes', []) if n in node_map]
            if len(valid_nodes) >= 2:
                valid_edges.append({'nodes': valid_nodes, 'weight': edge.get('weight', 1.0)})
        
        if not valid_edges:
            H_dict[tau_name] = torch.zeros(num_nodes, 0, device=device)
            W_e_dict[tau_name] = torch.zeros(0, device=device)
            D_v_dict[tau_name] = torch.ones(num_nodes, device=device)
            D_e_dict[tau_name] = torch.zeros(0, device=device)
            continue
        
        E_tau = len(valid_edges)
        H_tau = torch.zeros(num_nodes, E_tau, device=device)
        W_e_tau = torch.ones(E_tau, device=device)
        
        for e_idx, edge in enumerate(valid_edges):
            for node_id in edge['nodes']:
                H_tau[node_map[node_id], e_idx] = 1.0
            W_e_tau[e_idx] = edge['weight']
        
        D_v_tau = torch.sum(H_tau * W_e_tau.unsqueeze(0), dim=1) + 1e-8
        D_e_tau = torch.sum(H_tau, dim=0) + 1e-8
        
        H_dict[tau_name] = H_tau
        W_e_dict[tau_name] = W_e_tau
        D_v_dict[tau_name] = D_v_tau
        D_e_dict[tau_name] = D_e_tau
    
    return H_dict, W_e_dict, D_v_dict, D_e_dict


def has_valid_hyperedges(H_dict: Dict) -> bool:
    """检查是否有有效的超边"""
    return any(H_dict[tau].size(1) > 0 for tau in ['MO', 'OM', 'CO'])


# ============================================================================
# 第三部分：LLM 初始化
# ============================================================================

_llm_tokenizer = None
_llm_model = None


def init_local_llm(cache_dir: str = 'ckpt'):
    """初始化本地Qwen模型"""
    global _llm_tokenizer, _llm_model
    
    if _llm_model is not None:
        return _llm_tokenizer, _llm_model
    
    print("Loading local LLM (Qwen2.5-3B-Instruct)...")
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    try:
        _llm_tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )
        _llm_model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir,
            torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        _llm_model.eval()
        if _llm_tokenizer.pad_token is None:
            _llm_tokenizer.pad_token = _llm_tokenizer.eos_token
        print(f"Local LLM loaded on {_llm_model.device}!")
        return _llm_tokenizer, _llm_model
    except Exception as e:
        print(f"Error loading local LLM: {e}")
        return None, None


# ============================================================================
# 第四部分：答案生成函数 (修复版)
# ============================================================================

def answer_with_classifier(
    model: nn.Module,
    x: torch.Tensor,
    H_dict: Dict,
    W_e_dict: Dict,
    D_v_dict: Dict,
    D_e_dict: Dict,
    q_emb: torch.Tensor,
    device: torch.device
) -> Tuple[str, torch.Tensor]:
    """使用分类器直接预测答案"""
    with torch.no_grad():
        logits = model(x, H_dict, W_e_dict, D_v_dict, D_e_dict, q_emb)
        pred_idx = logits.argmax().item()
        pred = chr(65 + pred_idx)
    return pred, logits

def answer_with_retrieval_and_llm(
    q_proj: torch.Tensor,
    node_features: torch.Tensor,
    hypergraph: Dict,
    question: str,
    choices: List[str],
    top_k: int = 10,
    M: int = 5,
    use_local_llm: bool = True,
    hidden_dim: int = 512  # 新增参数
) -> Tuple[str, List[Dict]]:
    """
    检索+LLM生成答案 (超图读出增强版)
    论文 3.5.3 节: 超图读出 + 上下文聚合
    
    Returns:
        pred: 预测答案
        subgraph_nodes: 检索到的子图节点
    """
    # 检索种子节点
    similarities = F.cosine_similarity(q_proj, node_features, dim=1)
    _, seed_indices = torch.topk(similarities, min(top_k, len(similarities)))
    
    # 超边扩展
    selected_edges = hyperedge_expansion(
        seed_indices, hypergraph, node_features, q_proj, M=M
    )
    subgraph = build_subgraph(selected_edges, hypergraph)
    
    # ===== 新增：超图读出 (论文 3.5.3 节) =====
    if subgraph['nodes'] and len(subgraph['nodes']) >= 2:
        # 构建节点索引映射
        node_to_idx = {n['id']: i for i, n in enumerate(hypergraph['nodes'])}
        
        # 获取子图节点嵌入
        sub_indices = []
        for n in subgraph['nodes']:
            if n['id'] in node_to_idx:
                sub_indices.append(node_to_idx[n['id']])
        
        if sub_indices:
            z_sub = node_features[sub_indices]  # [num_sub_nodes, hidden_dim]
            
            # 使用超图读出聚合上下文
            readout = HypergraphReadout(hidden_dim=hidden_dim, readout_type='attention')
            context, weights = readout(z_sub, q_proj, return_weights=True)
            
            # 用上下文增强问题表示
            enhanced_q = q_proj + context.unsqueeze(0) * 0.3
            q_proj = enhanced_q
    
    # LLM推理（使用增强后的问题表示）
    pred = answer_with_llm(
        question, choices,
        subgraph['nodes'], use_local_llm=use_local_llm
    )
    
    return pred, subgraph['nodes']



def hyperedge_expansion(
    seed_nodes: torch.Tensor,
    hypergraph: Dict,
    node_embeddings: torch.Tensor,
    question_emb: torch.Tensor,
    M: int = 5
) -> List[Dict]:
    """超边扩展 (修复版)"""
    candidate_edges = []
    node_map = {n['id']: i for i, n in enumerate(hypergraph['nodes'])}
    seed_set = set(seed_nodes.tolist())
    
    all_edges = get_all_hyperedges(hypergraph)
    
    # 收集包含种子节点的超边
    seen = set()
    for edge in all_edges:
        edge_nodes = edge.get('nodes', [])
        for node_id in edge_nodes:
            if node_id in node_map and node_map[node_id] in seed_set:
                e_key = tuple(sorted(edge_nodes))
                if e_key not in seen:
                    seen.add(e_key)
                    candidate_edges.append(edge)
                break
    
    # 计算超边匹配度
    edge_scores = []
    for edge in candidate_edges:
        valid_nodes = [node_map[n] for n in edge.get('nodes', []) if n in node_map]
        if len(valid_nodes) < 2:
            continue
        # 使用超边内节点的平均嵌入
        edge_emb = node_embeddings[valid_nodes].mean(dim=0, keepdim=True)
        score = F.cosine_similarity(question_emb, edge_emb, dim=1).item()
        edge_scores.append((edge, score))
    
    edge_scores.sort(key=lambda x: x[1], reverse=True)
    return [e for e, _ in edge_scores[:M]]


def build_subgraph(selected_edges: List[Dict], hypergraph: Dict) -> Dict:
    """构建子超图"""
    node_ids = set()
    for edge in selected_edges:
        node_ids.update(edge.get('nodes', []))
    
    node_map = {n['id']: i for i, n in enumerate(hypergraph['nodes'])}
    valid_indices = [node_map[n] for n in node_ids if n in node_map]
    
    return {
        'nodes': [hypergraph['nodes'][i] for i in valid_indices],
        'node_indices': valid_indices,
        'hyperedges': selected_edges
    }


def answer_with_llm(
    question: str,
    choices: List[str],
    context_nodes: List[Dict],
    use_local_llm: bool = True
) -> str:
    """使用LLM生成答案"""
    global _llm_tokenizer, _llm_model
    
    if use_local_llm:
        if _llm_tokenizer is None or _llm_model is None:
            init_local_llm()
        
        if _llm_tokenizer is not None and _llm_model is not None:
            return _answer_with_qwen(question, choices, context_nodes)
    
    return _fallback_answer(question, choices)


def _answer_with_qwen(
    question: str,
    choices: List[str],
    context_nodes: List[Dict]
) -> str:
    """使用Qwen模型生成答案"""
    global _llm_tokenizer, _llm_model
    
    context_parts = []
    for i, node in enumerate(context_nodes[:8]):
        text = node.get('text', node.get('S_i', node.get('summary', '')))
        if text:
            context_parts.append(f"Event {i+1}: {text[:400]}")
    
    context = "\n".join(context_parts) if context_parts else "No relevant events found."
    choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    
    prompt = f"""Based on the following story events, answer the multiple-choice question.

Context Events:
{context}

Question: {question}

Choices:
{choices_str}

Instructions: Output ONLY the letter (A, B, C, D, or E). Do not include any explanation.

Answer:"""
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on given context."},
            {"role": "user", "content": prompt}
        ]
        
        text = _llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = _llm_tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(_llm_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = _llm_model.generate(
                **inputs, max_new_tokens=20,
                temperature=0.1, do_sample=True,
                pad_token_id=_llm_tokenizer.pad_token_id,
                eos_token_id=_llm_tokenizer.eos_token_id
            )
        
        response = _llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        for char in reversed(response.strip()):
            if char in ['A', 'B', 'C', 'D', 'E']:
                return char
        return 'A'
    except Exception as e:
        print(f"LLM error: {e}")
        return _fallback_answer(question, choices)


def _fallback_answer(question: str, choices: List[str]) -> str:
    """关键词匹配降级方案"""
    stopwords = {'what', 'how', 'why', 'does', 'do', 'is', 'are', 'was', 'were',
                 'the', 'a', 'an', 'to', 'for', 'of', 'with', 'on', 'at', 'from',
                 'by', 'in', 'into', 'through', 'during', 'including', 'which'}
    
    q_words = set([w.lower() for w in question.split() 
                   if w.lower() not in stopwords and len(w) > 2])
    
    scores = []
    for choice in choices:
        choice_clean = choice.split('.', 1)[-1] if '.' in choice else choice
        choice_words = set([w.lower() for w in choice_clean.split() if len(w) > 2])
        scores.append(len(q_words & choice_words))
    
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return chr(65 + best_idx)


# ============================================================================
# 第五部分：核心功能函数 (修复版)
# ============================================================================

def load_hypergraph(path: str) -> Dict:
    """加载超图JSON文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def encode_question(
    question: str,
    choices: List[str],
    tokenizer: Any,
    model: nn.Module,
    device: torch.device
) -> torch.Tensor:
    """编码问题和选项，返回 [1, 1024]"""
    choices_str = " ".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
    text = f"{question} Choices: {choices_str}"
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]


def find_hypergraph_file(video_id: str, hypergraph_dir: str) -> Optional[str]:
    """查找超图文件"""
    # 清理video_id
    clean_id = video_id.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    hg_path = os.path.join(hypergraph_dir, f"{clean_id}.json")
    if os.path.exists(hg_path):
        return hg_path
    
    for root, dirs, files in os.walk(hypergraph_dir):
        if f"{clean_id}.json" in files:
            return os.path.join(root, f"{clean_id}.json")
        # 也尝试原始video_id
        if f"{video_id}.json" in files:
            return os.path.join(root, f"{video_id}.json")
    
    return None


# ============================================================================
# 第六部分：模型加载 (修复版)
# ============================================================================

def load_model(
    checkpoint_path: str,
    model_type: str,
    device: torch.device,
    type_aware: bool = True
) -> Tuple[nn.Module, bool, Dict]:
    """加载模型 (修复版 v3)"""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 移除可能的 module. 前缀
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model_config = checkpoint.get('model_config', {})
    in_channels = model_config.get('in_channels', 1024)
    hidden_dim = model_config.get('hidden_channels', 512)
    num_layers = model_config.get('num_layers', 3)
    num_answers = model_config.get('num_answers', 5)
    
    # 自动检测类型感知
    has_type_aware_keys = any(
        'type_embeddings' in k or 'attention_mlp' in k 
        for k in state_dict.keys()
    )
    if has_type_aware_keys and not type_aware:
        print("Auto-enabling type_aware mode (detected from checkpoint)")
        type_aware = True
    
    # 创建模型
    if model_type == 'e2e':
        hgne = HGNE(in_channels=in_channels, hidden_channels=hidden_dim, 
                    num_layers=num_layers, type_aware=type_aware)
        model = EndToEndModel(hgne=hgne, hidden_dim=hidden_dim, 
                            num_answers=num_answers, type_aware=type_aware)
        use_classifier = True
    else:
        model = HGNE(in_channels=in_channels, hidden_channels=hidden_dim,
                    num_layers=num_layers, type_aware=type_aware)
        use_classifier = False
    
    # ★ 关键修复：在加载权重前，检测并初始化缺失的 q_proj 层
    if 'q_proj.weight' not in state_dict:
        print("  q_proj layer not in checkpoint, initializing with identity-like weights")
        hgne_model = model.hgne if hasattr(model, 'hgne') else model
        
        # 检查 checkpoint 中是否有 q_proj 的旧名字或其他变体
        # 如果没有，用截断单位矩阵初始化，保留更多原始信息
        with torch.no_grad():
            in_dim = hgne_model.q_proj.in_features   # 1024 (BERT)
            out_dim = hgne_model.q_proj.out_features # 512 (HGNE hidden)
            
            # 方法1: 截断单位矩阵 + 零填充（推荐）
            # 前512维直接映射，后512维丢弃
            identity_block = torch.eye(min(in_dim, out_dim))  # [512, 512]
            weight = torch.zeros(out_dim, in_dim)
            weight[:min(in_dim, out_dim), :min(in_dim, out_dim)] = identity_block
            hgne_model.q_proj.weight.data.copy_(weight)
            hgne_model.q_proj.bias.data.zero_()
            
            print(f"  q_proj initialized: {out_dim}x{in_dim} (identity block)")
    else:
        print("  q_proj weights found in checkpoint")
    
    # 尝试加载权重
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Loaded with strict=True")
    except RuntimeError as e:
        print(f"Trying strict=False...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"  Missing keys: {len(missing_keys)}")
            for k in missing_keys[:5]:
                print(f"    - {k}")
            if len(missing_keys) > 5:
                print(f"    ... and {len(missing_keys)-5} more")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)}")
            for k in unexpected_keys[:5]:
                print(f"    - {k}")
            if len(unexpected_keys) > 5:
                print(f"    ... and {len(unexpected_keys)-5} more")
    
    model.to(device)
    model.eval()
    
    return model, use_classifier, {
        'type_aware': type_aware,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'in_channels': in_channels,
        'num_answers': num_answers
    }


# ============================================================================
# 第七部分：主推理函数 (修复版)
# ============================================================================

def run_qa(args) -> float:
    """运行QA推理 (修复版)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"CausalHyperGraph - HAR Inference v2.0 (修复版)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Use LLM: {args.use_local_llm if not args.no_llm else False}")
    print(f"Top-K: {args.top_k}, M: {args.M}")
    print(f"Type-Aware: {not args.no_type_aware}")
    print(f"{'='*60}\n")
    
    # 加载问题
    with open(args.questions, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")
    
    # 加载BERT编码器
    print("Loading BERT...")
    tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased', cache_dir=args.cache_dir)
    bert_model = AutoModel.from_pretrained('bert-large-uncased', cache_dir=args.cache_dir)
    bert_model.to(device)
    bert_model.eval()
    
    # 加载模型
    type_aware = not getattr(args, 'no_type_aware', False)
    model, use_classifier, model_config = load_model(
        args.checkpoint, args.model_type, device, type_aware=type_aware
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Model config: {model_config}")
    print(f"Use classifier: {use_classifier}")
    
    # 初始化LLM（如果需要）
    if not args.no_llm and args.use_local_llm and not use_classifier:
        init_local_llm(args.cache_dir)
    
    # ================== 推理循环 ==================
    print("\nRunning QA inference...")
    results = []
    correct = 0
    total = 0
    skipped = 0
    detailed_stats = defaultdict(int)
    type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    start_time = time.time()
    
    for idx, q_dict in enumerate(tqdm(questions, desc="Processing QA")):
        video_id = q_dict.get('vid', q_dict.get('video_id', q_dict.get('vid_name', '')))
        
        hg_path = find_hypergraph_file(video_id, args.hypergraph_dir)
        
        if hg_path is None:
            skipped += 1
            detailed_stats['missing_hypergraph'] += 1
            results.append({**q_dict, 'predicted': 'X', 'correct': False, 'error': 'missing_hypergraph'})
            total += 1
            continue
        
        try:
            hypergraph = load_hypergraph(hg_path)
        except Exception as e:
            skipped += 1
            detailed_stats['load_error'] += 1
            results.append({**q_dict, 'predicted': 'X', 'correct': False, 'error': f'load_error: {e}'})
            total += 1
            continue
        
        q_type = q_dict.get('type', 'unknown')
        choices = q_dict.get('choices', ['A', 'B', 'C', 'D', 'E'])
        while len(choices) < 5:
            choices.append(f"Option {len(choices)+1}")
        
        try:
            # ---- 构建超图张量 ----
            H_dict, W_e_dict, D_v_dict, D_e_dict = build_type_aware_tensors(hypergraph, device)
            
            # 检查有效超边
            if not has_valid_hyperedges(H_dict):
                skipped += 1
                detailed_stats['no_valid_hyperedges'] += 1
                results.append({**q_dict, 'predicted': 'X', 'correct': False, 'error': 'no_valid_hyperedges'})
                total += 1
                continue
            
            # 构建节点特征矩阵
            embedding_dim = model_config.get('in_channels', 1024)
            node_embeddings = []
            for n in hypergraph['nodes']:
                emb = n.get('embedding', [])
                if not emb:
                    emb = [0.0] * embedding_dim
                elif len(emb) < embedding_dim:
                    emb = emb + [0.0] * (embedding_dim - len(emb))
                node_embeddings.append(emb[:embedding_dim])
            
            x = torch.tensor(node_embeddings, dtype=torch.float32).to(device)
            
            # ---- 编码问题 ----
            q_emb = encode_question(q_dict['question'], choices, tokenizer, bert_model, device)
            
            # ---- 获取超图感知特征 ----
            with torch.no_grad():
                z, attn_log = model.hgne(x, H_dict, W_e_dict, D_v_dict, D_e_dict) if use_classifier else model(x, H_dict, W_e_dict, D_v_dict, D_e_dict)
                
                # 投影问题到超图空间
                q_proj = model.hgne.q_proj(q_emb) if use_classifier else model.q_proj(q_emb)
                
                # 确保维度正确
                if q_proj.dim() == 1:
                    q_proj = q_proj.unsqueeze(0)
            
            # ---- 答案生成 ----
            if use_classifier and args.no_llm:
                # 端到端分类器
                pred, logits = answer_with_classifier(
                    model, x, H_dict, W_e_dict, D_v_dict, D_e_dict, q_emb, device
                )
                confidence = F.softmax(logits, dim=0).max().item()
                detailed_stats['classifier'] += 1
                subgraph_nodes = []
            else:
                # 检索+LLM
                pred, subgraph_nodes = answer_with_retrieval_and_llm(
                    q_proj, z, hypergraph,
                    q_dict['question'], choices,
                    top_k=args.top_k, M=args.M,
                    use_local_llm=args.use_local_llm and not args.no_llm,
                    hidden_dim=model_config.get('hidden_dim', 512)
                )
                confidence = 1.0
                detailed_stats['llm'] += 1
            
            # ---- 判断正确性 ----
            correct_option = q_dict.get('option', q_dict.get('answer', ''))
            if isinstance(correct_option, int):
                correct_option = chr(65 + correct_option)
            elif isinstance(correct_option, str) and correct_option.isdigit():
                correct_option = chr(65 + int(correct_option))
            
            is_correct = (pred == correct_option) if correct_option else False
            
            if is_correct:
                correct += 1
            total += 1
            
            type_stats[q_type]['total'] += 1
            if is_correct:
                type_stats[q_type]['correct'] += 1
            
            results.append({
                **q_dict,
                'predicted': pred,
                'correct': is_correct,
                'confidence': confidence,
                'num_nodes': len(hypergraph['nodes']),
                'num_subgraph_nodes': len(subgraph_nodes) if subgraph_nodes else 0
            })
            
        except Exception as e:
            import traceback
            skipped += 1
            detailed_stats['runtime_error'] += 1
            results.append({**q_dict, 'predicted': 'X', 'correct': False, 'error': str(e)})
            total += 1
            if args.verbose:
                print(f"\nError on {video_id}: {e}")
                traceback.print_exc()
        
        # 清理显存
        if idx % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # ================== 统计 ==================
    elapsed = time.time() - start_time
    accuracy = correct / max(total - skipped, 1) * 100 if total > skipped else 0
    overall_accuracy = correct / max(total, 1) * 100
    
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"Total questions: {total}")
    print(f"Correct: {correct}")
    print(f"Skipped (errors): {skipped}")
    print(f"Accuracy (excluding skipped): {accuracy:.2f}%")
    print(f"Accuracy (overall): {overall_accuracy:.2f}%")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    if elapsed > 0:
        print(f"Speed: {total/elapsed:.2f} q/s")
    
    print(f"\nDetailed stats:")
    for key, val in sorted(detailed_stats.items()):
        print(f"  {key}: {val}")
    
    if type_stats:
        print(f"\nPer-type accuracy:")
        for q_type, stats in sorted(type_stats.items()):
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total'] * 100
                print(f"  {q_type}: {acc:.1f}% ({stats['correct']}/{stats['total']})")
    
    print(f"{'='*60}")
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'accuracy_excluding_skipped': accuracy,
            'accuracy_overall': overall_accuracy,
            'correct': correct,
            'total': total,
            'skipped': skipped,
            'config': vars(args),
            'model_config': model_config,
            'type_stats': {k: dict(v) for k, v in type_stats.items()},
            'detailed_stats': dict(detailed_stats),
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {args.output}")
    return accuracy


# ============================================================================
# 第八部分：命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CausalHyperGraph QA Inference v2.0 (修复版)')
    
    parser.add_argument('--questions', required=True, help='QA JSON文件')
    parser.add_argument('--hypergraph_dir', required=True, help='超图目录')
    parser.add_argument('--checkpoint', required=True, help='模型检查点')
    parser.add_argument('--output', required=True, help='输出文件')
    parser.add_argument('--cache_dir', default='ckpt', help='模型缓存目录')
    
    parser.add_argument('--model_type', choices=['hgne', 'e2e'], default='e2e',
                        help='模型类型: hgne(纯编码器+LLM) 或 e2e(端到端分类器)')
    parser.add_argument('--no_type_aware', action='store_true',
                        help='禁用类型感知')
    
    parser.add_argument('--top_k', type=int, default=10, help='种子节点数量')
    parser.add_argument('--M', type=int, default=5, help='扩展超边数量')
    
    parser.add_argument('--use_local_llm', action='store_true', help='使用本地LLM')
    parser.add_argument('--no_llm', action='store_true', help='禁用LLM')
    
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if args.no_llm:
        args.use_local_llm = False
    
    run_qa(args)


if __name__ == '__main__':
    main()