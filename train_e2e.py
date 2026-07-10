# experiments/train_e2e.py
"""
能跑

端到端微调 - 匹配旧版预训练权重 (hidden_dim=512)/256
v7版本是用512训练的



cd /d/SDIPCT/papers/experiment_v0/
source venvs/env_chg_main/Scripts/activate


python experiments/train_e2e.py \
    --questions data/questions.json \
    --hypergraph_dir experiments/hcm/hypergraphs \
    --hgne_checkpoint checkpoints/hgne_pretrained_v7.pt \
    --epochs 30 \
    --batch_size 32 \
    --gradient_accumulation 4 \
    --lr 5e-5 \
    --output_dir checkpoints/e2e_final \
    --seed 42
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import warnings
import numpy as np

warnings.filterwarnings('ignore')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def smart_load_json(file_path):
    for enc in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return json.load(f)
        except:
            continue
    return None


# ===== 模型定义（与旧版train_hgne.py完全一致） =====

class TypeAwareHypergraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_types=3, type_names=None):
        super().__init__()
        self.num_types = num_types
        self.type_names = type_names or ['MO', 'OM', 'CO']
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.ModuleList([
            nn.Linear(in_channels, out_channels) for _ in range(num_types)
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
            nn.init.zeros_(theta.bias)
        nn.init.normal_(self.type_embeddings, std=0.1)

    def _single_type_conv(self, x, H_tau, W_e_tau, D_v_tau, D_e_tau, theta):
        D_v_inv_sqrt = torch.pow(D_v_tau + 1e-8, -0.5).clamp(min=0)
        D_e_inv = torch.pow(D_e_tau + 1e-8, -1.0).clamp(min=0)
        out = theta(x)
        out.mul_(D_v_inv_sqrt.unsqueeze(1))
        out = torch.mm(H_tau.T, out)
        out.mul_(W_e_tau.unsqueeze(1))
        out.mul_(D_e_inv.unsqueeze(1))
        out = torch.mm(H_tau, out)
        out.mul_(D_v_inv_sqrt.unsqueeze(1))
        return out

    def forward(self, x, H_dict, W_e_dict, D_v_dict, D_e_dict):
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
        attn_inputs = torch.cat([
            torch.cat([gap, self.type_embeddings[tau_idx].unsqueeze(0)], dim=-1)
            for tau_idx, _ in type_outputs
        ], dim=0)
        attn_weights = F.softmax(self.attention_mlp(attn_inputs).squeeze(-1), dim=0)
        out = torch.zeros_like(type_outputs[0][1])
        for idx, (_, tau_out) in enumerate(type_outputs):
            out.add_(attn_weights[idx] * tau_out)
        return F.relu_(out), attn_weights


class HGNE(nn.Module):
    def __init__(self, in_channels=1024, hidden_channels=256, num_layers=3):
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
        for i, conv in enumerate(self.convs):
            x_new, _ = conv(x, H_dict, W_e_dict, D_v_dict, D_e_dict)
            x_new = self.layer_norms[i](x_new)
            x_new = self.dropout(x_new)
            x = x_new
            x_list.append(x_new)
        z = torch.cat(x_list, dim=-1)
        z = self.proj(z)
        return z


class QAClassifier(nn.Module):
    def __init__(self, hidden_dim=256, num_answers=5):
        super().__init__()
        self.q_proj = nn.Linear(1024, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.2, batch_first=False)
        # self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.2, batch_first=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_answers)
        )

    def forward(self, node_embeddings, question_embedding):
        if question_embedding.dim() == 1:
            question_embedding = question_embedding.unsqueeze(0)
        q = self.q_proj(question_embedding)
        node_embeddings_unsq = node_embeddings.unsqueeze(1)
        q_unsq = q.unsqueeze(0)
        attn_output, _ = self.cross_attn(query=q_unsq, key=node_embeddings_unsq, value=node_embeddings_unsq)
        context = attn_output.squeeze(0).squeeze(0)
        combined = torch.cat([context, q.squeeze(0)], dim=0)
        return self.classifier(combined)


class EndToEndModel(nn.Module):
    def __init__(self, hgne, hidden_dim=256, num_answers=5):
        super().__init__()
        self.hgne = hgne
        self.qa_head = QAClassifier(hidden_dim, num_answers)
        self.freeze_hgne = False

    def forward(self, x, H_dict, W_e_dict, D_v_dict, D_e_dict, q_emb):
        with torch.set_grad_enabled(not self.freeze_hgne):
            z = self.hgne(x, H_dict, W_e_dict, D_v_dict, D_e_dict)
        return self.qa_head(z, q_emb)

    def set_freeze_hgne(self, freeze=True):
        self.freeze_hgne = freeze
        for param in self.hgne.parameters():
            param.requires_grad = not freeze


# ===== 数据集 =====

class TypeAwareQADataset(Dataset):
    def __init__(self, questions, hypergraph_dir, tokenizer, bert_model, device, preload_hg=True):
        self.hypergraph_dir = hypergraph_dir
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.device = device
        self.hypergraphs = {}
        self.valid_questions = []
        
        for q in tqdm(questions, desc="Loading hypergraphs"):
            vid = q['vid']
            if preload_hg and vid not in self.hypergraphs:
                hg_path = self._find_hg(vid)
                if hg_path:
                    try:
                        with open(hg_path, 'r', encoding='utf-8') as f:
                            hg = json.load(f)
                        self.hypergraphs[vid] = self._process_hg(hg)
                    except:
                        continue
            if vid in self.hypergraphs:
                self.valid_questions.append(q)
        
        print(f"Loaded {len(self.valid_questions)} valid questions")
    
    def _find_hg(self, vid):
        for root, _, files in os.walk(self.hypergraph_dir):
            for f in files:
                if f.endswith('.json') and not f.startswith('_'):
                    if os.path.splitext(f)[0] == vid:
                        return os.path.join(root, f)
        return None
    
    def _process_hg(self, hg):
        nodes = hg['nodes']
        hedges = hg.get('hyperedges', {})
        
        typed = {}
        if isinstance(hedges, dict):
            for t in ['MO', 'OM', 'CO']:
                typed[t] = hedges.get(t, [])
        
        embs = []
        for n in nodes:
            e = n.get('embedding', [0.0]*1024)
            if len(e) != 1024:
                e = [0.0]*1024
            embs.append(e)
        
        nid2idx = {n['id']: i for i, n in enumerate(nodes)}
        num_nodes = len(nodes)
        
        H_dict, W_e_dict = {}, {}
        for t in ['MO', 'OM', 'CO']:
            edges = typed.get(t, [])
            ne = max(1, len(edges))
            H = torch.zeros(num_nodes, ne)
            We = torch.ones(ne)
            for ei, edge in enumerate(edges):
                for nid in edge.get('nodes', []):
                    if nid in nid2idx:
                        H[nid2idx[nid], ei] = 1.0
                if 'weight' in edge:
                    We[ei] = float(edge['weight'])
            H_dict[t] = H
            W_e_dict[t] = We
        
        return {
            'x': torch.tensor(embs, dtype=torch.float32),
            'H_dict': H_dict,
            'W_e_dict': W_e_dict,
            'num_nodes': num_nodes,
            'nid2idx': nid2idx
        }
    
    def __len__(self):
        return len(self.valid_questions)
    
    def __getitem__(self, idx):
        q = self.valid_questions[idx]
        vid = q['vid']
        hg = self.hypergraphs[vid]
        
        choices_str = " ".join([f"({chr(65+i)}) {c}" for i, c in enumerate(q['choices'])])
        text = f"{q['question']} Choices: {choices_str}"
        
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        with torch.no_grad():
            outputs = self.bert_model(**{k: v.to(self.device) for k, v in inputs.items()})
        q_emb = outputs.last_hidden_state[:, 0, :].cpu().squeeze(0)
        
        # 计算度数
        D_v_dict, D_e_dict = {}, {}
        for t in ['MO', 'OM', 'CO']:
            H = hg['H_dict'][t]
            We = hg['W_e_dict'][t]
            D_v_dict[t] = torch.sum(H * We.unsqueeze(0), dim=1) + 1e-8
            D_e_dict[t] = torch.sum(H, dim=0) + 1e-8
        
        answer_idx = ord(q['option']) - 65
        answer_idx = max(0, min(4, answer_idx))
        
        return {
            'x': hg['x'],
            'H_dict': hg['H_dict'],
            'W_e_dict': hg['W_e_dict'],
            'D_v_dict': D_v_dict,
            'D_e_dict': D_e_dict,
            'q_emb': q_emb,
            'answer_idx': answer_idx,
            'vid': vid
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return batch if batch else None


# ===== 训练函数 =====
def train(model, train_loader, val_loader, epochs, device, output_dir, lr=1e-4, patience=5, grad_accum=4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # ★ 改用 CosineAnnealingLR（与 train_hgne.py 一致）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    
    best_acc, patience_cnt = 0, 0
    history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        optimizer.zero_grad()
        acc_grad = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            if batch is None:
                continue
            
            for s in batch:
                try:
                    x = s['x'].to(device)
                    Hd = {k: v.to(device) for k, v in s['H_dict'].items()}
                    Wed = {k: v.to(device) for k, v in s['W_e_dict'].items()}
                    Dvd = {k: v.to(device) for k, v in s['D_v_dict'].items()}
                    Ded = {k: v.to(device) for k, v in s['D_e_dict'].items()}
                    qe = s['q_emb'].to(device)
                    ai = torch.tensor(s['answer_idx']).to(device)
                except:
                    continue
                
                if x.shape[0] < 2:
                    continue
                
                logits = model(x, Hd, Wed, Dvd, Ded, qe)
                loss = criterion(logits.unsqueeze(0), ai.unsqueeze(0))
                loss = loss / grad_accum
                loss.backward()
                
                total_loss += loss.item() * grad_accum
                correct += (logits.argmax() == ai).item()
                total += 1
                acc_grad += 1
                
                if acc_grad % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            if total > 0:
                pbar.set_postfix({'loss': f'{total_loss/total:.4f}', 'acc': f'{correct/total*100:.1f}%'})
        
        # ★ 处理剩余梯度
        if acc_grad % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # ★ 每个epoch结束后调用 scheduler.step()
        scheduler.step()
        
        train_acc = correct / max(total, 1) * 100
        
        # 验证
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                for s in batch:
                    try:
                        x = s['x'].to(device)
                        Hd = {k: v.to(device) for k, v in s['H_dict'].items()}
                        Wed = {k: v.to(device) for k, v in s['W_e_dict'].items()}
                        Dvd = {k: v.to(device) for k, v in s['D_v_dict'].items()}
                        Ded = {k: v.to(device) for k, v in s['D_e_dict'].items()}
                        qe = s['q_emb'].to(device)
                        ai = torch.tensor(s['answer_idx']).to(device)
                    except:
                        continue
                    if x.shape[0] < 2:
                        continue
                    logits = model(x, Hd, Wed, Dvd, Ded, qe)
                    val_loss += criterion(logits.unsqueeze(0), ai.unsqueeze(0)).item()
                    val_correct += (logits.argmax() == ai).item()
                    val_total += 1
        
        val_acc = val_correct / max(val_total, 1) * 100
        
        history.append({
            'epoch': epoch+1, 'train_loss': total_loss/max(total,1),
            'train_acc': train_acc, 'val_loss': val_loss/max(val_total,1), 'val_acc': val_acc
        })
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience_cnt = 0
            torch.save(model.state_dict(), f'{output_dir}/best_model.pt')
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stop at epoch {epoch+1}")
                break
    
    return model, best_acc, history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions', required=True)
    parser.add_argument('--hypergraph_dir', required=True)
    parser.add_argument('--hgne_checkpoint', required=True)
    parser.add_argument('--output_dir', default='checkpoints/e2e_final')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--gradient_accumulation', type=int, default=4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--cache_dir', default='ckpt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载问题
    questions = smart_load_json(args.questions)
    print(f"Questions: {len(questions)}")
    
    # 加载BERT
    tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased', cache_dir=args.cache_dir)
    bert_model = AutoModel.from_pretrained('bert-large-uncased', cache_dir=args.cache_dir)
    bert_model.to(device).eval()
    
    # 加载HGNE（关键：hidden_channels=256！）
    print(f"Loading HGNE: {args.hgne_checkpoint}")
    hgne = HGNE(in_channels=1024, hidden_channels=512, num_layers=3)  # ★ 512匹配预训练权重
    
    ckpt = torch.load(args.hgne_checkpoint, map_location=device)
    sd = ckpt.get('model_state_dict', ckpt)
    if all(k.startswith('module.') for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}
    
    missing, unexpected = hgne.load_state_dict(sd, strict=False)
    print(f"Loaded HGNE: missing={len(missing)}, unexpected={len(unexpected)}")
    hgne.to(device)
    
    # 创建模型
    model = EndToEndModel(hgne, hidden_dim=512, num_answers=5)  # ★ 512
    model.to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # 数据集
    dataset = TypeAwareQADataset(questions, args.hypergraph_dir, tokenizer, bert_model, device)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Train: {train_size}, Val: {val_size}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 训练
    model, best_acc, history = train(
        model, train_loader, val_loader, args.epochs, device, args.output_dir,
        lr=args.lr, patience=args.patience, grad_accum=args.gradient_accumulation
    )
    
    # 保存
    with open(f'{args.output_dir}/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    torch.save({'model_state_dict': model.state_dict(), 'val_acc': best_acc}, f'{args.output_dir}/final.pt')
    print(f"\nDone! Best Val Acc: {best_acc:.2f}%")


if __name__ == '__main__':
    main()