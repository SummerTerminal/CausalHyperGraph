# train_e2e.py (modified)
"""
End-to-end fine-tuning: L = L_fine + λ₁L_pre + λ₂L_aux (paper)
"""
import os, json, argparse, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import warnings, numpy as np
from models import HGNE, EndToEndModel
from hgne.train_hgne import HyperedgePredictionLoss
from har.question_type_infer import QuestionTypeInferer

warnings.filterwarnings('ignore')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def smart_load_json(fp):
    for enc in ['utf-8','utf-8-sig','latin-1','cp1252']:
        try:
            with open(fp,'r',encoding=enc) as f: return json.load(f)
        except: continue
    return None

class TypeAwareQADataset(Dataset):
    def __init__(self, questions, hg_dir, tok, bert, dev, preload=True):
        self.hg_dir = hg_dir; self.tok = tok; self.bert = bert; self.dev = dev
        self.type_inferer = QuestionTypeInferer()
        self.hgs = {}; self.valid_qs = []
        for q in tqdm(questions, desc="Loading hgs"):
            vid = q['vid']
            if preload and vid not in self.hgs:
                p = self._find(vid)
                if p:
                    try:
                        with open(p,'r',encoding='utf-8') as f: hg = json.load(f)
                        self.hgs[vid] = self._process_hg(hg)
                    except: continue
            if vid in self.hgs: self.valid_qs.append(q)
        print(f"Valid: {len(self.valid_qs)}")

    def _find(self, vid):
        for r,_,fs in os.walk(self.hg_dir):
            for f in fs:
                if f.endswith('.json') and not f.startswith('_') and os.path.splitext(f)[0]==vid:
                    return os.path.join(r,f)
        return None

    def _process_hg(self, hg):
        nds = hg['nodes']; hed = hg.get('hyperedges',{})
        typed = {'MO':[],'OM':[],'CO':[]}
        if isinstance(hed,dict):
            for t in ['MO','OM','CO']: typed[t]=hed.get(t,[])
        else:
            for e in hed: typed[e.get('type','CO')].append(e)

        embs = []
        for n in nds:
            e = n.get('embedding',[0.0]*1024)
            if len(e)!=1024: e=[0.0]*1024
            embs.append(e)
        nid2idx = {n['id']:i for i,n in enumerate(nds)}
        N = len(nds)
        Hd = {t: torch.zeros(N,0) for t in ['MO','OM','CO']}
        Wd = {t: torch.zeros(0) for t in ['MO','OM','CO']}
        for t in ['MO','OM','CO']:
            edges = typed.get(t,[])
            if not edges: continue
            ne = len(edges)
            Ht = torch.zeros(N,ne)
            Wt = torch.ones(ne)
            for ei,edge in enumerate(edges):
                for nid in edge.get('nodes',[]):
                    if nid in nid2idx: Ht[nid2idx[nid],ei]=1.0
                if 'weight' in edge: Wt[ei]=float(edge['weight'])
            Hd[t] = Ht; Wd[t] = Wt
        return {'x':torch.tensor(embs,dtype=torch.float32),
                'H_dict':Hd,'W_e_dict':Wd,'num_nodes':N,
                'nid2idx':nid2idx,'hyperedges_raw':typed}

    def __len__(self): return len(self.valid_qs)

    def __getitem__(self, idx):
        q = self.valid_qs[idx]; vid = q['vid']; hg = self.hgs[vid]
        choices = q.get('choices',['A','B','C','D','E'])
        while len(choices)<5: choices.append(f"Opt {len(choices)+1}")
        c_str = " ".join([f"({chr(65+i)}) {c}" for i,c in enumerate(choices)])
        txt = f"{q['question']} Choices: {c_str}"
        inp = self.tok(txt, return_tensors='pt', truncation=True, max_length=256)
        with torch.no_grad():
            out = self.bert(**{k:v.to(self.dev) for k,v in inp.items()})
        q_emb = out.last_hidden_state[:,0,:].cpu().squeeze(0)

        Dvd, Ded = {}, {}
        for t in ['MO','OM','CO']:
            H = hg['H_dict'][t]; We = hg['W_e_dict'][t]
            Dvd[t] = torch.sum(H*We.unsqueeze(0), dim=1) + 1e-8
            Ded[t] = torch.sum(H, dim=0) + 1e-8

        # fix: handle various answer formats
        opt = q.get('option')
        if isinstance(opt,str) and len(opt)==1 and opt.isalpha():
            ans_idx = ord(opt.upper())-65
        elif isinstance(opt,int): ans_idx = opt
        elif isinstance(opt,str) and opt.isdigit(): ans_idx = int(opt)
        else: ans_idx = 0
        ans_idx = max(0,min(4,ans_idx))

        tl = self.type_inferer.get_label(q['question'])
        return {'x':hg['x'],'H_dict':hg['H_dict'],'W_e_dict':hg['W_e_dict'],
                'D_v_dict':Dvd,'D_e_dict':Ded,'q_emb':q_emb,'answer_idx':ans_idx,
                'vid':vid,'question':q['question'],'choices':choices,
                'type_label':tl,'hyperedges_raw':hg['hyperedges_raw'],
                'nid2idx':hg['nid2idx'],'num_nodes':hg['num_nodes']}

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return batch if batch else None

def train_with_full_loss(model, train_loader, val_loader, epochs, dev, out_dir,
                         lr=1e-4, lambda_pre=0.3, lambda_aux=0.1, patience=5, grad_accum=4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    ce = nn.CrossEntropyLoss()
    pre_loss_fn = HyperedgePredictionLoss(hidden_dim=model.hidden_dim).to(dev)
    best_acc, pat_cnt = 0,0
    hist = []

    for ep in range(epochs):
        model.train()
        tot_loss, corr, tot = 0,0,0
        opt.zero_grad()
        ag = 0
        pbar = tqdm(train_loader, desc=f"Ep {ep+1}/{epochs}")
        for batch in pbar:
            if batch is None: continue
            for s in batch:
                try:
                    x = s['x'].to(dev); Hd = {k:v.to(dev) for k,v in s['H_dict'].items()}
                    Wed = {k:v.to(dev) for k,v in s['W_e_dict'].items()}
                    Dvd = {k:v.to(dev) for k,v in s['D_v_dict'].items()}
                    Ded = {k:v.to(dev) for k,v in s['D_e_dict'].items()}
                    qe = s['q_emb'].to(dev); ai = torch.tensor(s['answer_idx']).to(dev)
                    hg_raw = {'hyperedges':s['hyperedges_raw']}
                    n2i = s['nid2idx']; tl = s.get('type_label',3)
                except: continue
                if x.shape[0]<2: continue

                out = model(x, Hd, Wed, Dvd, Ded, qe,
                            hypergraph=hg_raw, node_to_idx=n2i,
                            return_all=True, type_label=tl)
                l_fine = ce(out['logits'].unsqueeze(0), ai.unsqueeze(0))

                Hall = torch.cat([Hd[k] for k in ['MO','OM','CO'] if k in Hd], dim=1)
                if Hall.size(1)>0:
                    l_pre = pre_loss_fn(out['z'], Hall, s['num_nodes'])
                else: l_pre = torch.tensor(0.0, device=dev)

                l_aux = out['aux_loss']
                loss = l_fine + lambda_pre*l_pre + lambda_aux*l_aux
                loss = loss / grad_accum
                loss.backward()
                tot_loss += loss.item()*grad_accum
                corr += (out['logits'].argmax()==ai).item()
                tot += 1; ag += 1

                if ag % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); opt.zero_grad()
            if tot>0: pbar.set_postfix({'loss':f'{tot_loss/tot:.4f}','acc':f'{corr/tot*100:.1f}%'})

        if ag % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        sched.step()
        train_acc = corr/max(tot,1)*100

        model.eval()
        v_loss, v_corr, v_tot = 0,0,0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                for s in batch:
                    try:
                        x = s['x'].to(dev); Hd = {k:v.to(dev) for k,v in s['H_dict'].items()}
                        Wed = {k:v.to(dev) for k,v in s['W_e_dict'].items()}
                        Dvd = {k:v.to(dev) for k,v in s['D_v_dict'].items()}
                        Ded = {k:v.to(dev) for k,v in s['D_e_dict'].items()}
                        qe = s['q_emb'].to(dev); ai = torch.tensor(s['answer_idx']).to(dev)
                        hg_raw = {'hyperedges':s['hyperedges_raw']}
                        n2i = s['nid2idx']; tl = s.get('type_label',3)
                    except: continue
                    if x.shape[0]<2: continue
                    logits,_ = model(x, Hd, Wed, Dvd, Ded, qe,
                                     hypergraph=hg_raw, node_to_idx=n2i,
                                     type_label=tl)
                    v_loss += ce(logits.unsqueeze(0), ai.unsqueeze(0)).item()
                    v_corr += (logits.argmax()==ai).item()
                    v_tot += 1
        val_acc = v_corr/max(v_tot,1)*100

        hist.append({'epoch':ep+1,'train_loss':tot_loss/max(tot,1),'train_acc':train_acc,
                     'val_loss':v_loss/max(v_tot,1),'val_acc':val_acc})
        print(f"Ep {ep+1}: Train {train_acc:.2f}% Val {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc; pat_cnt=0
            torch.save(model.state_dict(), f'{out_dir}/best_model.pt')
        else:
            pat_cnt+=1
            if pat_cnt>=patience:
                print(f"Early stop at ep {ep+1}"); break
    return model, best_acc, hist

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--questions', required=True)
    p.add_argument('--hypergraph_dir', required=True)
    p.add_argument('--hgne_checkpoint', required=True)
    p.add_argument('--output_dir', default='checkpoints/e2e_final')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--gradient_accumulation', type=int, default=4)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--lambda_pre', type=float, default=0.3)
    p.add_argument('--lambda_aux', type=float, default=0.1)
    p.add_argument('--cache_dir', default='ckpt')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}")
    os.makedirs(a.output_dir, exist_ok=True)

    qs = smart_load_json(a.questions)
    print(f"Questions: {len(qs)}")

    tok = AutoTokenizer.from_pretrained('bert-large-uncased', cache_dir=a.cache_dir)
    bert = AutoModel.from_pretrained('bert-large-uncased', cache_dir=a.cache_dir)
    bert.to(dev).eval()

    print(f"Loading HGNE: {a.hgne_checkpoint}")
    hgne = HGNE(in_channels=1024, hidden_channels=512, num_layers=3)
    ck = torch.load(a.hgne_checkpoint, map_location=dev)
    sd = ck.get('model_state_dict', ck)
    if all(k.startswith('module.') for k in sd): sd = {k[7:]:v for k,v in sd.items()}
    miss, unexp = hgne.load_state_dict(sd, strict=False)
    print(f"HGNE loaded: missing={len(miss)}, unexpected={len(unexp)}")
    hgne.to(dev)

    model = EndToEndModel(hgne, hidden_dim=512, num_answers=5, use_har=True)
    model.to(dev)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    ds = TypeAwareQADataset(qs, a.hypergraph_dir, tok, bert, dev)
    tr_sz = int(0.8*len(ds)); vl_sz = len(ds)-tr_sz
    tr_ds, vl_ds = torch.utils.data.random_split(ds, [tr_sz,vl_sz],
                        generator=torch.Generator().manual_seed(a.seed))
    print(f"Train: {tr_sz}, Val: {vl_sz}")
    tr_ldr = DataLoader(tr_ds, batch_size=a.batch_size, shuffle=True, collate_fn=collate_fn)
    vl_ldr = DataLoader(vl_ds, batch_size=a.batch_size, shuffle=False, collate_fn=collate_fn)

    model, best_acc, hist = train_with_full_loss(model, tr_ldr, vl_ldr, a.epochs, dev, a.output_dir,
                                                 lr=a.lr, lambda_pre=a.lambda_pre, lambda_aux=a.lambda_aux,
                                                 patience=a.patience, grad_accum=a.gradient_accumulation)

    with open(f'{a.output_dir}/history.json','w') as f: json.dump(hist,f,indent=2)
    torch.save({'model_state_dict':model.state_dict(),'val_acc':best_acc}, f'{a.output_dir}/final.pt')
    print(f"\nDone! Best val acc: {best_acc:.2f}%")

if __name__ == '__main__':
    main()