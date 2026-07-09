"""
train_with_qa_head.py - 整合QA分类头的两阶段训练
实现论文 3.6 节完整训练流程
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import random
import numpy as np
from collections import defaultdict

# 导入已有模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hgne.train_hgne import (
    HGNE, TypeAwareHypergraphConv, 
    TypeAwareContrastiveLoss, HyperedgePredictionLoss
)
from har.hypergraph_readout import HyperedgeAwareReasoner, create_reasoner
from har.question_type_infer import QuestionTypeInferer, TypeAwareAuxiliaryLoss


class EndToEndHGNE(nn.Module):
    """
    端到端超图问答模型
    整合：HGNE编码器 + 超边感知推理器 + QA分类头
    """
    
    def __init__(
        self,
        hgne: HGNE,
        reasoner: HyperedgeAwareReasoner,
        hidden_dim: int = 512,
        num_answers: int = 5,
        num_types: int = 3
    ):
        super().__init__()
        self.hgne = hgne
        self.reasoner = reasoner
        self.hidden_dim = hidden_dim
        self.num_answers = num_answers
        
        # 问题投影
        self.q_proj = nn.Linear(1024, hidden_dim)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        
        # 答案分类器
        # 论文公式: p(a_i | q, G_sub) = softmax(MLP(concat(c, e_q, e_{a_i})))
        self.answer_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_answers)
        )
        
        # 类型感知辅助损失
        self.type_aux_loss = TypeAwareAuxiliaryLoss(hidden_dim, num_types)
        
        # 冻结状态
        self.freeze_hgne = False
    
    def forward(
        self,
        x: torch.Tensor,
        H_dict: Dict[str, torch.Tensor],
        W_e_dict: Dict[str, torch.Tensor],
        D_v_dict: Dict[str, torch.Tensor],
        D_e_dict: Dict[str, torch.Tensor],
        q_emb: torch.Tensor,
        hypergraph: Dict,
        node_to_idx: Dict[str, int],
        return_all: bool = False
    ):
        """
        Args:
            x: 节点特征 [num_nodes, 1024]
            H_dict, W_e_dict, D_v_dict, D_e_dict: 超图张量
            q_emb: 问题编码 [1, 1024]
            hypergraph: 超图数据
            node_to_idx: 节点ID到索引映射
            return_all: 是否返回所有中间结果
            
        Returns:
            logits: 答案logits [num_answers]
            info: 中间信息字典
        """
        # 投影问题
        q_proj = self.q_proj(q_emb)  # [1, hidden_dim]
        
        # 1. HGNE编码
        with torch.set_grad_enabled(not self.freeze_hgne):
            z, attn_log = self.hgne(x, H_dict, W_e_dict, D_v_dict, D_e_dict)
        
        # 2. 超边感知推理
        context, reasoner_info = self.reasoner(
            z, q_proj, hypergraph, H_dict, node_to_idx
        )
        
        # 3. 答案分类
        combined = torch.cat([context.unsqueeze(0), q_proj], dim=-1)  # [1, 2*hidden_dim]
        logits = self.answer_classifier(combined).squeeze(0)  # [num_answers]
        
        if return_all:
            return {
                'logits': logits,
                'context': context,
                'q_proj': q_proj,
                'z': z,
                'attn_log': attn_log,
                'reasoner_info': reasoner_info
            }
        
        return logits, reasoner_info
    
    def set_freeze_hgne(self, freeze: bool = True):
        """冻结/解冻HGNE"""
        self.freeze_hgne = freeze
        for param in self.hgne.parameters():
            param.requires_grad = not freeze


class QADatasetWithType(Dataset):
    """
    带类型标签的QA数据集
    支持辅助损失 L_aux
    """
    
    def __init__(
        self,
        questions: List[Dict],
        hypergraph_dir: str,
        tokenizer,
        bert_model,
        device,
        type_inferer: Optional[QuestionTypeInferer] = None,
        preload_hg: bool = True
    ):
        self.hypergraph_dir = hypergraph_dir
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.device = device
        self.type_inferer = type_inferer or QuestionTypeInferer()
        
        # 预加载超图
        self.hypergraphs = {}
        self.valid_questions = []
        
        for q in tqdm(questions, desc="Loading hypergraphs"):
            vid = q.get('vid', q.get('video_id', ''))
            if not vid:
                continue
            
            if preload_hg and vid not in self.hypergraphs:
                hg_path = self._find_hg_file(vid)
                if hg_path:
                    try:
                        with open(hg_path, 'r', encoding='utf-8') as f:
                            hg = json.load(f)
                        self.hypergraphs[vid] = self._process_hypergraph(hg)
                    except Exception as e:
                        continue
                else:
                    continue
            
            if vid in self.hypergraphs:
                # 推断问题类型
                q_type, type_scores = self.type_inferer.infer_type(q['question'])
                q['_type'] = q_type
                q['_type_scores'] = type_scores
                q['_type_label'] = self.type_inferer.get_type_label(q['question'])
                self.valid_questions.append(q)
        
        print(f"✅ Loaded {len(self.valid_questions)} valid questions")
        self._print_type_distribution()
    
    def _find_hg_file(self, vid):
        """查找超图文件"""
        for root, dirs, files in os.walk(self.hypergraph_dir):
            for f in files:
                if f.endswith('.json') and not f.startswith('_'):
                    file_vid = os.path.splitext(f)[0]
                    if file_vid == vid or file_vid.lower() == vid.lower():
                        return os.path.join(root, f)
        return None
    
    def _process_hypergraph(self, hg):
        """处理超图"""
        nodes = hg.get('nodes', [])
        hyperedges_raw = hg.get('hyperedges', {})
        
        # 统一超边格式
        if isinstance(hyperedges_raw, dict):
            typed_edges = {}
            for tau_name in ['MO', 'OM', 'CO']:
                typed_edges[tau_name] = hyperedges_raw.get(tau_name, [])
        else:
            typed_edges = {'MO': [], 'OM': [], 'CO': []}
            for edge in hyperedges_raw:
                tau = edge.get('type', 'CO')
                if tau in typed_edges:
                    typed_edges[tau].append(edge)
        
        # 节点索引
        node_id_to_idx = {n['id']: i for i, n in enumerate(nodes)}
        
        # 节点特征
        embeddings = []
        for n in nodes:
            emb = n.get('embedding', [0.0] * 1024)
            if len(emb) != 1024:
                emb = [0.0] * 1024
            embeddings.append(emb)
        
        num_nodes = len(nodes)
        
        # 构建张量
        H_dict = {}
        W_e_dict = {}
        D_v_dict = {}
        D_e_dict = {}
        
        for tau_name in ['MO', 'OM', 'CO']:
            edges = typed_edges.get(tau_name, [])
            if len(edges) == 0:
                H_dict[tau_name] = torch.zeros(num_nodes, 1)
                W_e_dict[tau_name] = torch.ones(1)
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
        
        return {
            'x': torch.tensor(embeddings, dtype=torch.float32),
            'H_dict': H_dict,
            'W_e_dict': W_e_dict,
            'D_v_dict': D_v_dict,
            'D_e_dict': D_e_dict,
            'num_nodes': num_nodes,
            'node_id_to_idx': node_id_to_idx,
            'hyperedges_raw': typed_edges,
            'nodes': nodes
        }
    
    def _print_type_distribution(self):
        """打印问题类型分布"""
        type_counts = defaultdict(int)
        for q in self.valid_questions:
            type_counts[q.get('_type', 'unknown')] += 1
        
        print("  Question type distribution:")
        for t, c in sorted(type_counts.items()):
            print(f"    {t}: {c} ({c/len(self.valid_questions)*100:.1f}%)")
    
    def __len__(self):
        return len(self.valid_questions)
    
    def __getitem__(self, idx):
        q = self.valid_questions[idx]
        vid = q.get('vid', q.get('video_id', ''))
        
        hg_data = self.hypergraphs.get(vid)
        if hg_data is None:
            return None
        
        # 编码问题
        choices = q.get('choices', ['A', 'B', 'C', 'D', 'E'])
        choices_str = " ".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
        text = f"{q['question']} Choices: {choices_str}"
        
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        
        with torch.no_grad():
            outputs = self.bert_model(**{k: v.to(self.device) for k, v in inputs.items()})
        q_emb = outputs.last_hidden_state[:, 0, :].cpu().squeeze(0)
        
        # 答案索引
        correct = q.get('option', q.get('answer', ''))
        if isinstance(correct, int):
            answer_idx = correct
        elif isinstance(correct, str):
            if correct.isdigit():
                answer_idx = int(correct)
            else:
                answer_idx = ord(correct.upper()) - 65
        else:
            answer_idx = 0
        answer_idx = max(0, min(4, answer_idx))
        
        return {
            'x': hg_data['x'],
            'H_dict': hg_data['H_dict'],
            'W_e_dict': hg_data['W_e_dict'],
            'D_v_dict': hg_data['D_v_dict'],
            'D_e_dict': hg_data['D_e_dict'],
            'num_nodes': hg_data['num_nodes'],
            'node_id_to_idx': hg_data['node_id_to_idx'],
            'hyperedges_raw': hg_data['hyperedges_raw'],
            'nodes': hg_data['nodes'],
            'q_emb': q_emb,
            'answer_idx': answer_idx,
            'vid': vid,
            'question': q['question'],
            'choices': choices,
            'type_label': q.get('_type_label', 3),  # 3 = unknown
            'type_scores': q.get('_type_scores', {})
        }


def collate_fn_with_type(batch):
    """批处理函数"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return batch


def train_with_qa_head(
    model: EndToEndHGNE,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4,
    lambda_pre: float = 0.3,
    lambda_aux: float = 0.1,
    lambda_cont: float = 1.0,
    gradient_accumulation_steps: int = 4,
    warmup_epochs: int = 3,
    patience: int = 5,
    output_dir: str = 'checkpoints'
):
    """
    完整的训练流程
    
    论文公式: L = L_fine + λ₁·L_pre + λ₂·L_aux + λ₃·L_cont
    """
    
    # 损失函数
    contrastive_loss = TypeAwareContrastiveLoss(temperature=0.1).to(device)
    pre_loss_fn = HyperedgePredictionLoss(hidden_dim=model.hidden_dim).to(device)
    ce_loss = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 学习率调度
    steps_per_epoch = max(1, len(train_dataloader) // gradient_accumulation_steps)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=warmup_steps / max(total_steps, 1),
        anneal_strategy='cos'
    )
    
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    history = []
    
    print(f"\n{'='*60}")
    print(f"开始训练 (论文 3.6.2 节)")
    print(f"{'='*60}")
    print(f"  λ₁ (预训练): {lambda_pre}")
    print(f"  λ₂ (辅助): {lambda_aux}")
    print(f"  λ₃ (对比): {lambda_cont}")
    print(f"  梯度累积: {gradient_accumulation_steps}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        model.train()
        
        epoch_losses = defaultdict(float)
        epoch_correct = 0
        epoch_total = 0
        accum_count = 0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            if batch is None or len(batch) == 0:
                continue
            
            batch_loss = 0
            batch_correct = 0
            batch_size = len(batch)
            
            for sample in batch:
                try:
                    # 移动到GPU
                    x = sample['x'].to(device, non_blocking=True)
                    H_dict = {k: v.to(device, non_blocking=True) for k, v in sample['H_dict'].items()}
                    W_e_dict = {k: v.to(device, non_blocking=True) for k, v in sample['W_e_dict'].items()}
                    D_v_dict = {k: v.to(device, non_blocking=True) for k, v in sample['D_v_dict'].items()}
                    D_e_dict = {k: v.to(device, non_blocking=True) for k, v in sample['D_e_dict'].items()}
                    q_emb = sample['q_emb'].to(device, non_blocking=True)
                    answer_idx = torch.tensor(sample['answer_idx']).to(device, non_blocking=True)
                    type_label = sample.get('type_label', 3)
                except Exception as e:
                    continue
                
                if x.shape[0] < 2:
                    continue
                
                # 前向传播
                with autocast(enabled=torch.cuda.is_available()):
                    outputs = model(
                        x, H_dict, W_e_dict, D_v_dict, D_e_dict,
                        q_emb, sample, sample['node_id_to_idx'],
                        return_all=True
                    )
                    
                    logits = outputs['logits']
                    context = outputs['context']
                    q_proj = outputs['q_proj']
                    z = outputs['z']
                    
                    # 1. 问答损失 L_fine
                    loss_fine = ce_loss(logits.unsqueeze(0), answer_idx.unsqueeze(0))
                    
                    # 2. 超边预测损失 L_pre
                    H_all = torch.cat([H_dict[k] for k in ['MO', 'OM', 'CO'] if k in H_dict], dim=1)
                    if H_all.size(1) > 0:
                        loss_pre = pre_loss_fn(z, H_all, sample['num_nodes'])
                    else:
                        loss_pre = torch.tensor(0.0, device=device)
                    
                    # 3. 对比损失 L_cont
                    pos_masks = {}
                    for tau_name in ['MO', 'OM', 'CO']:
                        if tau_name in H_dict:
                            H_tau = H_dict[tau_name]
                            if H_tau.size(1) > 0:
                                pos_mask = (H_tau @ H_tau.T) > 0
                                pos_mask.fill_diagonal_(False)
                                pos_masks[tau_name] = pos_mask
                    if pos_masks:
                        loss_cont, _ = contrastive_loss(z, pos_masks)
                    else:
                        loss_cont = torch.tensor(0.0, device=device)
                    
                    # 4. 类型辅助损失 L_aux
                    if type_label < 3:
                        loss_aux, _ = model.type_aux_loss(context, q_proj.squeeze(0), type_label)
                    else:
                        loss_aux = torch.tensor(0.0, device=device)
                    
                    # 总损失
                    loss = (loss_fine + 
                           lambda_pre * loss_pre + 
                           lambda_aux * loss_aux + 
                           lambda_cont * loss_cont)
                
                # 反向传播
                if not torch.isnan(loss) and not torch.isinf(loss):
                    scaled_loss = loss / (batch_size * gradient_accumulation_steps)
                    scaler.scale(scaled_loss).backward()
                    
                    batch_loss += loss.item()
                    pred = logits.argmax()
                    batch_correct += (pred == answer_idx).item()
                    
                    # 记录损失
                    epoch_losses['total'] += loss.item()
                    epoch_losses['fine'] += loss_fine.item()
                    epoch_losses['pre'] += loss_pre.item()
                    epoch_losses['cont'] += loss_cont.item()
                    epoch_losses['aux'] += loss_aux.item()
                    
                accum_count += 1
                
                if accum_count % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scheduler.step()
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            
            if batch_size > 0:
                epoch_correct += batch_correct
                epoch_total += batch_size
                
                if epoch_total > 0:
                    pbar.set_postfix({
                        'loss': f"{epoch_losses['total']/max(epoch_total,1):.4f}",
                        'acc': f"{epoch_correct/epoch_total*100:.1f}%",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                    })
        
        # 处理剩余梯度
        if accum_count % gradient_accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # 计算平均损失
        for k in epoch_losses:
            epoch_losses[k] /= max(epoch_total, 1)
        
        train_acc = epoch_correct / max(epoch_total, 1) * 100
        
        # 验证
        val_acc, val_loss = evaluate_with_qa_head(model, val_dataloader, device)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_losses['total'],
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'losses': dict(epoch_losses)
        })
        
        print(f"\nEpoch {epoch+1}: "
              f"Train Loss={epoch_losses['total']:.4f}, "
              f"Train Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, "
              f"Val Acc={val_acc:.2f}%")
        
        # 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'history': history
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"  ✅ Best model saved (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[早停] {patience} 个epoch无改善")
                break
        
        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'history': history
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    return model, best_val_acc, history


@torch.no_grad()
def evaluate_with_qa_head(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    ce_loss = nn.CrossEntropyLoss()
    
    for batch in dataloader:
        if batch is None or len(batch) == 0:
            continue
        
        for sample in batch:
            try:
                x = sample['x'].to(device)
                H_dict = {k: v.to(device) for k, v in sample['H_dict'].items()}
                W_e_dict = {k: v.to(device) for k, v in sample['W_e_dict'].items()}
                D_v_dict = {k: v.to(device) for k, v in sample['D_v_dict'].items()}
                D_e_dict = {k: v.to(device) for k, v in sample['D_e_dict'].items()}
                q_emb = sample['q_emb'].to(device)
                answer_idx = torch.tensor(sample['answer_idx']).to(device)
            except Exception:
                continue
            
            if x.shape[0] < 2:
                continue
            
            logits, _ = model(
                x, H_dict, W_e_dict, D_v_dict, D_e_dict,
                q_emb, sample, sample['node_id_to_idx']
            )
            
            loss = ce_loss(logits.unsqueeze(0), answer_idx.unsqueeze(0))
            total_loss += loss.item()
            pred = logits.argmax()
            correct += (pred == answer_idx).item()
            total += 1
    
    model.train()
    return correct / max(total, 1) * 100, total_loss / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description='端到端HGNE训练 - 完整版')
    
    parser.add_argument('--questions', required=True, help='QA数据文件')
    parser.add_argument('--hypergraph_dir', required=True, help='超图目录')
    parser.add_argument('--hgne_checkpoint', required=True, help='HGNE预训练检查点')
    parser.add_argument('--output_dir', default='checkpoints/e2e_full')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation', type=int, default=4)
    parser.add_argument('--patience', type=int, default=5)
    
    # 损失权重
    parser.add_argument('--lambda_pre', type=float, default=0.3)
    parser.add_argument('--lambda_aux', type=float, default=0.1)
    parser.add_argument('--lambda_cont', type=float, default=1.0)
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--top_k_seed', type=int, default=10)
    parser.add_argument('--M', type=int, default=5)
    
    parser.add_argument('--cache_dir', default='ckpt')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"端到端HGNE训练 - 完整版")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载数据
    from train_e2e import smart_load_json
    questions = smart_load_json(args.questions)
    print(f"\n加载 {len(questions)} 个问题")
    
    # 加载BERT
    print("\n加载BERT...")
    tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased', cache_dir=args.cache_dir)
    bert_model = AutoModel.from_pretrained('bert-large-uncased', cache_dir=args.cache_dir)
    bert_model.to(device)
    bert_model.eval()
    
    # 加载HGNE
    print(f"\n加载HGNE: {args.hgne_checkpoint}")
    hgne = HGNE(
        in_channels=1024, 
        hidden_channels=args.hidden_dim, 
        num_layers=args.num_layers
    )
    checkpoint = torch.load(args.hgne_checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    hgne.load_state_dict(state_dict, strict=False)
    hgne.to(device)
    
    # 创建推理器
    reasoner = create_reasoner(
        hidden_dim=args.hidden_dim,
        top_k_seed=args.top_k_seed,
        M=args.M
    )
    
    # 创建模型
    model = EndToEndHGNE(
        hgne=hgne,
        reasoner=reasoner,
        hidden_dim=args.hidden_dim,
        num_answers=5
    )
    model.to(device)
    
    # 创建类型推断器
    type_inferer = QuestionTypeInferer()
    
    # 创建数据集
    dataset = QADatasetWithType(
        questions=questions,
        hypergraph_dir=args.hypergraph_dir,
        tokenizer=tokenizer,
        bert_model=bert_model,
        device=device,
        type_inferer=type_inferer
    )
    
    # 划分数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    print(f"训练集: {train_size}, 验证集: {val_size}")
    
    dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn_with_type, num_workers=0
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_with_type, num_workers=0
    )
    
    # 训练
    model, best_acc, history = train_with_qa_head(
        model=model,
        train_dataloader=dataloader,
        val_dataloader=val_dataloader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        lambda_pre=args.lambda_pre,
        lambda_aux=args.lambda_aux,
        lambda_cont=args.lambda_cont,
        gradient_accumulation_steps=args.gradient_accumulation,
        patience=args.patience,
        output_dir=args.output_dir
    )
    
    # 保存结果
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"结果保存至: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()