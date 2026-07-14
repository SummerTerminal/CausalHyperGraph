# train_hgne.py (full)
"""
Implements the two-stage training from paper Section 3.6: pretrain (20 eps) + finetune (50 eps)
"""
import os, json, argparse, random, time, platform
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
from typing import Dict, Optional, List
import numpy as np

# 共享模型
from models import HGNE, TypeAwareHypergraphConv
from har.question_type_infer import QuestionTypeInferer

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ===== GPU optim =====
def setup_gpu_optimizations():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {name}, {mem_gb:.1f} GB")
        torch.cuda.set_per_process_memory_fraction(0.9)
        return mem_gb
    return 0

def get_memory_usage():
    if not torch.cuda.is_available():
        return {}
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return {'allocated': alloc, 'reserved': reserved, 'total': total,
            'free': total - reserved, 'utilization': (alloc / total) * 100}

# ===== Losses (paper 3.6) =====

class TypeAwareContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature

    def _single_type_loss(self, z, pos_mask):
        N = z.shape[0]
        if N < 2 or pos_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device), 0
        z_n = F.normalize(z, dim=1)
        sim = torch.mm(z_n, z_n.T) / self.temp
        # stable
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
        exp_sim = torch.exp(sim)
        pos_exp = exp_sim * pos_mask.float()
        numer = pos_exp.sum(dim=1)
        neg_mask = torch.ones(N, N, device=z.device) - torch.eye(N, device=z.device)
        denom = (exp_sim * neg_mask).sum(dim=1)
        has_pos = pos_mask.sum(dim=1) > 0
        valid = has_pos.sum().item()
        if valid == 0:
            return torch.tensor(0.0, device=z.device), 0
        ratio = numer[has_pos] / (denom[has_pos] + 1e-8)
        loss = -torch.log(ratio + 1e-8).mean()
        return loss, valid

    def forward(self, z, pos_masks):
        total = torch.tensor(0.0, device=z.device)
        type_losses = {}
        w = 0.0
        for tau, pm in pos_masks.items():
            l, n = self._single_type_loss(z, pm)
            if n > 0 and not torch.isnan(l):
                total += l
                w += 1.0
                type_losses[tau] = l.item()
        if w > 0:
            total = total / w
        return total, type_losses

class HyperedgePredictionLoss(nn.Module):
    """predict whether a set of nodes forms a real hyperedge"""
    def __init__(self, hidden_dim=512, num_negs=5):
        super().__init__()
        self.num_negs = num_negs
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1)
        )

    def _sample_neg(self, z, n_nodes, edge_sz, n_negs):
        dev = z.device
        negs = []
        for _ in range(n_negs):
            idx = torch.randint(0, n_nodes, (edge_sz,), device=dev)
            # ensure uniqueness
            while len(torch.unique(idx)) < edge_sz:
                idx = torch.randint(0, n_nodes, (edge_sz,), device=dev)
            negs.append(idx)
        return negs

    def forward(self, z, H_all, n_nodes):
        E = H_all.size(1)
        if E == 0 or n_nodes < 2:
            return torch.tensor(0.0, device=z.device)
        dev = z.device
        sz = H_all.sum(dim=0).clamp(min=1)
        e_emb = (H_all.T @ z) / sz.unsqueeze(1)   # [E, d]
        valid_mask = sz >= 2
        if not valid_mask.any():
            return torch.tensor(0.0, device=dev)
        pos_scores = self.readout_mlp(e_emb[valid_mask]).squeeze(-1)
        n_pos = pos_scores.shape[0]
        all_neg = []
        for i in range(n_pos):
            esz = int(sz[valid_mask][i].item())
            neg_idxs = self._sample_neg(z, n_nodes, esz, self.num_negs)
            for idxs in neg_idxs:
                neg_emb = z[idxs].mean(dim=0)
                all_neg.append(self.readout_mlp(neg_emb.unsqueeze(0)))
        neg_scores = torch.cat(all_neg).squeeze(-1)
        pos_exp = pos_scores.repeat_interleave(self.num_negs)
        all_scores = torch.cat([pos_exp, neg_scores])
        all_lbl = torch.cat([torch.ones(n_pos * self.num_negs, device=dev),
                            torch.zeros(len(neg_scores), device=dev)])
        return F.binary_cross_entropy_with_logits(all_scores, all_lbl)

class TypePredictionLoss(nn.Module):
    def __init__(self, hidden_dim=512, n_types=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_types)
        )
        self.type_names = ['MO', 'OM', 'CO']

    def forward(self, z, typed_edges, nid2idx=None, qtype_label=None):
        all_emb, all_lbl = [], []
        for tau_idx, tau_n in enumerate(self.type_names):
            for edge in typed_edges.get(tau_n, []):
                nodes = edge.get('nodes', [])
                if len(nodes) < 2: continue
                if nid2idx:
                    idxs = [nid2idx.get(n) for n in nodes if n in nid2idx]
                else:
                    continue
                if len(idxs) < 2: continue
                all_emb.append(z[idxs].mean(dim=0))
                all_lbl.append(tau_idx)
        if not all_emb:
            return torch.tensor(0.0, device=z.device)
        emb_t = torch.stack(all_emb)
        lbl_t = torch.tensor(all_lbl, device=z.device)
        cnt = torch.bincount(lbl_t)
        if len(cnt) == 3:
            w = 1.0 / (cnt.float() + 1e-8)
            w = w / w.sum() * 3
        else:
            w = torch.ones(3, device=z.device)
        main_loss = F.cross_entropy(self.classifier(emb_t), lbl_t, weight=w)
        if qtype_label is not None and qtype_label < 3:
            ctx = emb_t.mean(dim=0, keepdim=True)
            logits = self.classifier(ctx)
            tgt = torch.tensor([qtype_label], device=z.device)
            aux = F.cross_entropy(logits, tgt)
            return main_loss + aux
        return main_loss

# ===== Dataset =====
class TypeAwareHypergraphDataset(Dataset):
    def __init__(self, hg_dir, split='train', eval_split=0.1, seed=42):
        self.hg_dir = hg_dir
        self.split = split
        files = []
        for root, _, fs in os.walk(hg_dir):
            for f in fs:
                if f.endswith('.json') and not f.startswith('_') and not f.startswith('.'):
                    files.append(os.path.join(root, f))
        print(f"[Data] found {len(files)} hg files")
        random.seed(seed)
        random.shuffle(files)
        n_eval = max(1, int(eval_split * len(files)))
        if split == 'train':
            files = files[n_eval:]
        else:
            files = files[:n_eval]
        print(f"[Data] {split} set: {len(files)} hypergraphs")
        self.samples = []
        loaded = 0
        for fp in files:
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    hg = json.load(f)
                proc = self._process(hg, fp)
                if proc is not None:
                    proc = self._to_gpu_data(proc)
                    self.samples.append(proc)
                    loaded += 1
            except Exception as e:
                print(f"[Warn] skip {fp}: {e}")
        print(f"[Data] loaded {loaded} hypergraphs")
        if self.samples:
            self._print_stats()

    def _to_gpu_data(self, s):
        total_e = sum(v.size(1) for v in s['H_dict'].values())
        if total_e > 0:
            Hall = torch.zeros(s['num_nodes'], total_e)
            g_idx = 0
            for tau in ['MO','OM','CO']:
                if tau in s['H_dict']:
                    et = s['H_dict'][tau].size(1)
                    Hall[:, g_idx:g_idx+et] = s['H_dict'][tau]
                    g_idx += et
            s['H_all'] = Hall
        else:
            s['H_all'] = torch.zeros(s['num_nodes'], 1)
        return s

    def _process(self, hg, fp):
        if not hg.get('nodes'):
            return None
        vid = hg.get('video_id', os.path.splitext(os.path.basename(fp))[0])
        raw_edges = hg.get('hyperedges', {})
        if isinstance(raw_edges, dict):
            typed = {}
            for tau in ['MO','OM','CO']:
                elist = raw_edges.get(tau, [])
                if isinstance(elist, list):
                    typed[tau] = [e for e in elist if isinstance(e, dict)]
                else:
                    typed[tau] = []
        elif isinstance(raw_edges, list):
            typed = defaultdict(list)
            for e in raw_edges:
                if isinstance(e, dict):
                    typed[e.get('type','CO')].append(e)
            for tau in ['MO','OM','CO']:
                if tau not in typed:
                    typed[tau] = []
        else:
            typed = {'MO':[], 'OM':[], 'CO':[]}
        total_cnt = sum(len(v) for v in typed.values())
        if total_cnt == 0:
            return None

        nid2idx = {n['id']: i for i, n in enumerate(hg['nodes'])}
        N = len(hg['nodes'])
        # get embedding dim from first valid node
        emb_dim = 1024
        for n in hg['nodes']:
            emb = n.get('embedding', [])
            if emb and len(emb) > 0:
                emb_dim = len(emb)
                break
        embs = []
        for n in hg['nodes']:
            emb = n.get('embedding', [])
            if not emb or len(emb) == 0:
                emb = [0.0] * emb_dim
            embs.append(emb)
        x = torch.tensor(embs, dtype=torch.float32)

        Hd, Wd, Dvd, Ded = {}, {}, {}, {}
        for tau in ['MO','OM','CO']:
            edges = typed.get(tau, [])
            if not edges: continue
            Et = len(edges)
            Ht = torch.zeros(N, Et)
            Wt = torch.ones(Et)
            for ei, edge in enumerate(edges):
                for nid in edge.get('nodes', []):
                    if nid in nid2idx:
                        Ht[nid2idx[nid], ei] = 1.0
                if 'weight' in edge:
                    Wt[ei] = float(edge['weight'])
            Dvt = torch.sum(Ht * Wt.unsqueeze(0), dim=1) + 1e-8
            Det = torch.sum(Ht, dim=0) + 1e-8
            Hd[tau] = Ht
            Wd[tau] = Wt
            Dvd[tau] = Dvt
            Ded[tau] = Det

        pos_masks = {}
        for tau, Ht in Hd.items():
            if Ht.size(1) > 0:
                pm = (Ht @ Ht.T) > 0
                pm.fill_diagonal_(False)
                pos_masks[tau] = pm

        total_e = sum(v.size(1) for v in Hd.values())
        return {
            'x': x,
            'H_dict': Hd, 'W_e_dict': Wd, 'D_v_dict': Dvd, 'D_e_dict': Ded,
            'H_all': None,
            'pos_masks': pos_masks,
            'typed_edges': typed,
            'node_id_to_idx': nid2idx,
            'num_nodes': N,
            'num_edges': total_e,
            'video_id': vid,
        }

    def _print_stats(self):
        mo = sum(s['H_dict'].get('MO',torch.zeros(1)).size(1) for s in self.samples)
        om = sum(s['H_dict'].get('OM',torch.zeros(1)).size(1) for s in self.samples)
        co = sum(s['H_dict'].get('CO',torch.zeros(1)).size(1) for s in self.samples)
        avg_n = np.mean([s['num_nodes'] for s in self.samples])
        avg_e = np.mean([s['num_edges'] for s in self.samples])
        print(f"[Data] MO:{mo} OM:{om} CO:{co}")
        print(f"[Data] avg nodes:{avg_n:.1f}  edges:{avg_e:.1f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    return batch

# ===== Two-phase training =====

def pretrain_phase(model, dl, dev, epochs=20, lr=2e-5,
                   grad_acc=2, patience=5):
    print("\n" + "="*60)
    print("Phase 1: Pretrain - hyperedge prediction")
    print("="*60)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    pre_loss_fn = HyperedgePredictionLoss(hidden_dim=model.hidden_channels).to(dev)
    best_loss = float('inf')
    best_ep = 0
    pat_cnt = 0
    ema = None

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        cnt = 0
        opt_idx = 0
        buf = []
        for sample in dl:
            if isinstance(sample, list): buf.extend(sample)
            else: buf.append(sample)
            while len(buf) >= 1:
                b = buf[:1]; buf = buf[1:]
                s = b[0]
                if not isinstance(s, dict) or s['num_nodes'] < 3: continue
                x = s['x'].to(dev, non_blocking=True)
                Hd = {k:v.to(dev, non_blocking=True) for k,v in s['H_dict'].items()}
                Wd = {k:v.to(dev, non_blocking=True) for k,v in s['W_e_dict'].items()}
                Dvd = {k:v.to(dev, non_blocking=True) for k,v in s['D_v_dict'].items()}
                Ded = {k:v.to(dev, non_blocking=True) for k,v in s['D_e_dict'].items()}
                Hall = s['H_all'].to(dev, non_blocking=True)

                if opt_idx % grad_acc == 0:
                    opt.zero_grad(set_to_none=True)

                with autocast(enabled=torch.cuda.is_available()):
                    z, _ = model(x, Hd, Wd, Dvd, Ded)
                    loss = pre_loss_fn(z, Hall, s['num_nodes'])

                if not torch.isnan(loss) and not torch.isinf(loss):
                    scaled = loss / grad_acc
                    scaler.scale(scaled).backward()
                    ep_loss += loss.item()
                    cnt += 1
                    opt_idx += 1

                if opt_idx % grad_acc == 0 and opt_idx > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
        # leftover
        if buf:
            opt.zero_grad(set_to_none=True)
            for s in buf:
                if not isinstance(s, dict) or s['num_nodes'] < 3: continue
                x = s['x'].to(dev, non_blocking=True)
                Hd = {k:v.to(dev, non_blocking=True) for k,v in s['H_dict'].items()}
                Wd = {k:v.to(dev, non_blocking=True) for k,v in s['W_e_dict'].items()}
                Dvd = {k:v.to(dev, non_blocking=True) for k,v in s['D_v_dict'].items()}
                Ded = {k:v.to(dev, non_blocking=True) for k,v in s['D_e_dict'].items()}
                Hall = s['H_all'].to(dev, non_blocking=True)
                with autocast(enabled=torch.cuda.is_available()):
                    z, _ = model(x, Hd, Wd, Dvd, Ded)
                    loss = pre_loss_fn(z, Hall, s['num_nodes'])
                if not torch.isnan(loss) and not torch.isinf(loss):
                    scaler.scale(loss).backward()
                    ep_loss += loss.item()
                    cnt += 1
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()
        avg_loss = ep_loss / max(cnt, 1)
        cur_lr = sched.get_last_lr()[0]
        ema = avg_loss if ema is None else 0.95*ema + 0.05*avg_loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ep = ep+1
            best_sd = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            pat_cnt = 0
        else:
            pat_cnt += 1
        mem = get_memory_usage()
        gpu_u = mem.get('utilization',0) if mem else 0
        print(f"Pretrain Ep {ep+1:2d}/{epochs} | Loss:{avg_loss:.4f} (EMA:{ema:.4f}) LR:{cur_lr:.2e} GPU:{gpu_u:.1f}% Best:{best_loss:.4f} (Ep{best_ep})")
        if pat_cnt >= patience:
            print(f"\n[EarlyStop] {patience} epochs no improve")
            break
    model.load_state_dict(best_sd)
    print(f"\n[Pretrain done] Best loss {best_loss:.4f} (Ep {best_ep})")
    return model

def finetune_phase(model, dl, dev, epochs=50, lr=1e-4,
                   lam_pre=0.3, lam_aux=0.1, grad_acc=2, patience=5):
    print("\n" + "="*60)
    print("Phase 2: Finetune - QA optimization (full loss)")
    print("="*60)
    ti = QuestionTypeInferer()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    cont_loss = TypeAwareContrastiveLoss(temperature=0.1)
    pre_loss_fn = HyperedgePredictionLoss(hidden_dim=model.hidden_channels).to(dev)
    aux_loss_fn = TypePredictionLoss(hidden_dim=model.hidden_channels).to(dev)

    best_loss = float('inf')
    best_ep = 0
    pat_cnt = 0
    ema = None

    for ep in range(epochs):
        model.train()
        ep_losses = defaultdict(float)
        cnt = 0
        opt_idx = 0
        buf = []
        for sample in dl:
            if isinstance(sample, list): buf.extend(sample)
            else: buf.append(sample)
            while len(buf) >= 1:
                b = buf[:1]; buf = buf[1:]
                s = b[0]
                if not isinstance(s, dict) or s['num_nodes'] < 3: continue
                x = s['x'].to(dev, non_blocking=True)
                Hd = {k:v.to(dev, non_blocking=True) for k,v in s['H_dict'].items()}
                Wd = {k:v.to(dev, non_blocking=True) for k,v in s['W_e_dict'].items()}
                Dvd = {k:v.to(dev, non_blocking=True) for k,v in s['D_v_dict'].items()}
                Ded = {k:v.to(dev, non_blocking=True) for k,v in s['D_e_dict'].items()}
                pos_masks = {k:v.to(dev, non_blocking=True) for k,v in s['pos_masks'].items()}
                Hall = s['H_all'].to(dev, non_blocking=True)
                # question type (aux)
                q_text = s.get('question','')
                t_label = None
                if q_text:
                    _, _ = ti.infer_type(q_text)
                    t_label = ti.get_label(q_text)
                    if t_label == 3: t_label = None
                if opt_idx % grad_acc == 0:
                    opt.zero_grad(set_to_none=True)

                with autocast(enabled=torch.cuda.is_available()):
                    z, _ = model(x, Hd, Wd, Dvd, Ded)
                    l_cont, type_ls = cont_loss(z, pos_masks)
                    l_pre = pre_loss_fn(z, Hall, s['num_nodes'])
                    l_aux = aux_loss_fn(z, s['typed_edges'], s['node_id_to_idx'],
                                        qtype_label=t_label)
                    loss = l_cont + lam_pre * l_pre + lam_aux * l_aux

                if not torch.isnan(loss) and not torch.isinf(loss):
                    scaled = loss / grad_acc
                    scaler.scale(scaled).backward()
                    ep_losses['total'] += loss.item()
                    ep_losses['cont'] += l_cont.item()
                    ep_losses['pre'] += l_pre.item()
                    ep_losses['aux'] += l_aux.item()
                    for k,v in type_ls.items():
                        ep_losses[f'type_{k}'] += v
                    cnt += 1
                    opt_idx += 1
                if opt_idx % grad_acc == 0 and opt_idx > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
        # leftover
        if buf:
            opt.zero_grad(set_to_none=True)
            for s in buf:
                if not isinstance(s, dict) or s['num_nodes'] < 3: continue
                x = s['x'].to(dev, non_blocking=True)
                Hd = {k:v.to(dev, non_blocking=True) for k,v in s['H_dict'].items()}
                Wd = {k:v.to(dev, non_blocking=True) for k,v in s['W_e_dict'].items()}
                Dvd = {k:v.to(dev, non_blocking=True) for k,v in s['D_v_dict'].items()}
                Ded = {k:v.to(dev, non_blocking=True) for k,v in s['D_e_dict'].items()}
                pos_masks = {k:v.to(dev, non_blocking=True) for k,v in s['pos_masks'].items()}
                Hall = s['H_all'].to(dev, non_blocking=True)
                q_text = s.get('question','')
                t_label = None
                if q_text:
                    _, _ = ti.infer_type(q_text)
                    t_label = ti.get_label(q_text)
                    if t_label == 3: t_label = None
                with autocast(enabled=torch.cuda.is_available()):
                    z, _ = model(x, Hd, Wd, Dvd, Ded)
                    l_cont, type_ls = cont_loss(z, pos_masks)
                    l_pre = pre_loss_fn(z, Hall, s['num_nodes'])
                    l_aux = aux_loss_fn(z, s['typed_edges'], s['node_id_to_idx'],
                                        qtype_label=t_label)
                    loss = l_cont + lam_pre * l_pre + lam_aux * l_aux
                if not torch.isnan(loss) and not torch.isinf(loss):
                    scaler.scale(loss).backward()
                    ep_losses['total'] += loss.item()
                    ep_losses['cont'] += l_cont.item()
                    ep_losses['pre'] += l_pre.item()
                    ep_losses['aux'] += l_aux.item()
                    for k,v in type_ls.items():
                        ep_losses[f'type_{k}'] += v
                    cnt += 1
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()
        for k in ep_losses:
            ep_losses[k] /= max(cnt,1)
        avg_loss = ep_losses['total']
        cur_lr = sched.get_last_lr()[0]
        ema = avg_loss if ema is None else 0.95*ema + 0.05*avg_loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ep = ep+1
            best_sd = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            pat_cnt = 0
        else:
            pat_cnt += 1
        mem = get_memory_usage()
        gpu_u = mem.get('utilization',0) if mem else 0
        print(f"Finetune Ep {ep+1:2d}/{epochs} | Loss:{avg_loss:.4f} EMA:{ema:.4f} "
              f"Cont:{ep_losses['cont']:.4f} Pre:{ep_losses['pre']:.4f} Aux:{ep_losses['aux']:.4f} "
              f"LR:{cur_lr:.2e} GPU:{gpu_u:.1f}% Best:{best_loss:.4f} Ep{best_ep}")
        if pat_cnt >= patience:
            print(f"\n[EarlyStop] {patience} epochs no improve")
            break
    model.load_state_dict(best_sd)
    print(f"\n[Finetune done] Best loss {best_loss:.4f} (Ep {best_ep})")
    return model

# ===== Evaluation =====
@torch.no_grad()
def evaluate_model(model, dl, dev):
    model.eval()
    cont_loss = TypeAwareContrastiveLoss(0.1)
    pre_loss_fn = HyperedgePredictionLoss(hidden_dim=model.hidden_channels).to(dev).eval()
    aux_loss_fn = TypePredictionLoss(hidden_dim=model.hidden_channels).to(dev).eval()
    metrics = defaultdict(float)
    valid_cnt = 0
    for sample in dl:
        if isinstance(sample, list): s = sample[0] if sample else None
        else: s = sample
        if not isinstance(s, dict) or s['num_nodes'] < 3: continue
        x = s['x'].to(dev, non_blocking=True)
        Hd = {k:v.to(dev, non_blocking=True) for k,v in s['H_dict'].items()}
        Wd = {k:v.to(dev, non_blocking=True) for k,v in s['W_e_dict'].items()}
        Dvd = {k:v.to(dev, non_blocking=True) for k,v in s['D_v_dict'].items()}
        Ded = {k:v.to(dev, non_blocking=True) for k,v in s['D_e_dict'].items()}
        pos_masks = {k:v.to(dev, non_blocking=True) for k,v in s['pos_masks'].items()}
        Hall = s['H_all'].to(dev, non_blocking=True)
        with autocast(enabled=torch.cuda.is_available()):
            z, attn_log = model(x, Hd, Wd, Dvd, Ded)
            l_cont, _ = cont_loss(z, pos_masks)
            metrics['contrastive_loss'] += l_cont.item()
            # edge pred acc
            E = Hall.size(1)
            if E > 0:
                sz = Hall.sum(dim=0)
                valid_e = sz >= 2
                if valid_e.any():
                    e_emb = (Hall.T @ z) / sz.unsqueeze(1).clamp(min=1)
                    valid_emb = e_emb[valid_e]
                    neg_embs = []
                    for i in range(valid_emb.shape[0]):
                        enodes = Hall[:, valid_e][:, i].nonzero(as_tuple=True)[0]
                        if len(enodes) < 2: continue
                        repl = torch.randint(0, s['num_nodes'], (1,)).item()
                        new_nodes = enodes.clone()
                        rp = torch.randint(0, len(new_nodes), (1,)).item()
                        new_nodes[rp] = repl
                        while len(torch.unique(new_nodes)) < len(new_nodes):
                            repl = torch.randint(0, s['num_nodes'], (1,)).item()
                            new_nodes[rp] = repl
                        neg_embs.append(z[new_nodes].mean(dim=0))
                    if neg_embs:
                        neg_t = torch.stack(neg_embs)
                        pos_s = pre_loss_fn.readout_mlp(valid_emb).squeeze(-1)
                        neg_s = pre_loss_fn.readout_mlp(neg_t).squeeze(-1)
                        metrics['edge_pred_acc'] += (pos_s > neg_s).float().mean().item()
            # type pred acc
            if s.get('typed_edges'):
                all_emb, all_lbl = [], []
                for ti, tn in enumerate(['MO','OM','CO']):
                    for edge in s['typed_edges'].get(tn, []):
                        nodes = edge.get('nodes',[])
                        if len(nodes)<2: continue
                        idxs = [s['node_id_to_idx'].get(n) for n in nodes if n in s['node_id_to_idx']]
                        if len(idxs)<2: continue
                        all_emb.append(z[idxs].mean(dim=0))
                        all_lbl.append(ti)
                if all_emb:
                    emb_t = torch.stack(all_emb)
                    lbl_t = torch.tensor(all_lbl, device=dev)
                    logits = aux_loss_fn.classifier(emb_t)
                    preds = logits.argmax(dim=1)
                    metrics['type_pred_acc'] += (preds == lbl_t).float().mean().item()
            # cosine sim for positive pairs
            sim_sum, sim_cnt = 0.0, 0
            for pm in pos_masks.values():
                if pm.sum() > 0:
                    z_n = F.normalize(z, dim=1)
                    sim = torch.mm(z_n, z_n.T)
                    sim_sum += sim[pm].mean().item()
                    sim_cnt += 1
            if sim_cnt > 0:
                metrics['pos_cos_sim'] += sim_sum / sim_cnt
            # attention
            if attn_log:
                avg_attn = torch.stack(attn_log).mean(dim=0)
                for i, nm in enumerate(['MO','OM','CO']):
                    if i < len(avg_attn):
                        metrics[f'attn_{nm}'] += avg_attn[i].item()
        valid_cnt += 1
    for k in metrics:
        metrics[k] /= max(valid_cnt, 1)
    metrics['num_samples'] = valid_cnt
    return dict(metrics)

# ===== Main =====
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hypergraph_dir', required=True)
    p.add_argument('--output_dir', default='checkpoints')
    p.add_argument('--pretrain_epochs', type=int, default=20)
    p.add_argument('--finetune_epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr_pretrain', type=float, default=2e-5)
    p.add_argument('--lr_finetune', type=float, default=1e-4)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--num_layers', type=int, default=3)
    p.add_argument('--lambda_pre', type=float, default=0.3)
    p.add_argument('--lambda_aux', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval_split', type=float, default=0.1)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--gradient_accumulation_steps', type=int, default=8)
    a = p.parse_args()

    setup_gpu_optimizations()
    random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = TypeAwareHypergraphDataset(a.hypergraph_dir, 'train', a.eval_split, a.seed)
    eval_ds = TypeAwareHypergraphDataset(a.hypergraph_dir, 'eval', a.eval_split, a.seed)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn,
                          num_workers=a.num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    eval_dl = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    in_ch = train_ds.samples[0]['x'].shape[1] if train_ds.samples else 1024
    model = HGNE(in_channels=in_ch, hidden_channels=a.hidden_dim, num_layers=a.num_layers)
    model.to(dev)
    print(f"\n[Model] params: {sum(p.numel() for p in model.parameters()):,}")
    os.makedirs(a.output_dir, exist_ok=True)
    with open(os.path.join(a.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(a), f, indent=2)

    # pretrain
    model = pretrain_phase(model, train_dl, dev,
                           epochs=a.pretrain_epochs, lr=a.lr_pretrain,
                           grad_acc=a.gradient_accumulation_steps, patience=a.patience)
    torch.save({'model_state_dict': model.state_dict(),
                'model_config': {'in_channels': in_ch, 'hidden_channels': a.hidden_dim, 'num_layers': a.num_layers},
                'phase': 'pretrained'},
               os.path.join(a.output_dir, 'hgne_pretrained.pt'))

    # finetune
    model = finetune_phase(model, train_dl, dev,
                           epochs=a.finetune_epochs, lr=a.lr_finetune,
                           lam_pre=a.lambda_pre, lam_aux=a.lambda_aux,
                           grad_acc=a.gradient_accumulation_steps, patience=a.patience)
    torch.save({'model_state_dict': model.state_dict(),
                'model_config': {'in_channels': in_ch, 'hidden_channels': a.hidden_dim, 'num_layers': a.num_layers},
                'phase': 'finetuned', 'training_config': vars(a)},
               os.path.join(a.output_dir, 'hgne_finetuned.pt'))

    # eval
    print("\n"+"="*60+"\nEvaluating model...\n"+"="*60)
    metrics = evaluate_model(model, eval_dl, dev)
    print("[Eval results]")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    with open(os.path.join(a.output_dir, 'eval_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Done] model: {a.output_dir}/hgne_finetuned.pt")

if __name__ == '__main__':
    main()