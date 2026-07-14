# run_qa.py  --  HAR inference (shared model)
"""
HAR: Hyperedge-Aware Reasoner - unified inference.
Uses the shared model from models.py
"""
import os, json, argparse, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm
import gc, time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

from models import HGNE, EndToEndModel
from har.hypergraph_readout import HypergraphReadout

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'   # 避免警告

# ===== 工具函数（没动） =====
def get_all_hyperedges(hypergraph: Dict) -> List[Dict]:
    # flatten hyperedges from dict-of-lists or raw list
    hes = hypergraph.get('hyperedges', [])
    if isinstance(hes, dict):
        all_e = []
        for tau in ['MO', 'OM', 'CO']:
            elist = hes.get(tau, [])
            if isinstance(elist, list):
                all_e.extend(elist)
        return all_e
    elif isinstance(hes, list):
        return hes
    return []

def build_type_aware_tensors(hg: Dict, device: torch.device) -> Tuple[Dict, Dict, Dict, Dict]:
    # 构建每类超边的 incidence, degree 等
    nmap = {n['id']: i for i, n in enumerate(hg['nodes'])}
    N = len(hg['nodes'])
    hedges = hg.get('hyperedges', {})
    if isinstance(hedges, list):
        typed = {'MO': [], 'OM': [], 'CO': []}
        for e in hedges:
            t = e.get('type', 'CO')
            if t in typed:
                typed[t].append(e)
        hedges = typed

    Hd, Wd, Dvd, Ded = {}, {}, {}, {}
    for tau in ['MO', 'OM', 'CO']:
        edges = hedges.get(tau, [])
        valid = len(edges) > 0 and any(
            any(nid in nmap for nid in e.get('nodes', []))
            for e in edges
        )
        if not valid:
            Hd[tau] = torch.zeros(N, 0, device=device)
            Wd[tau] = torch.zeros(0, device=device)
            Dvd[tau] = torch.ones(N, device=device)
            Ded[tau] = torch.zeros(0, device=device)
            continue
        # gather valid edges
        valid_es = []
        for e in edges:
            nodes = [n for n in e.get('nodes', []) if n in nmap]
            if len(nodes) >= 2:
                valid_es.append({'nodes': nodes, 'weight': e.get('weight', 1.0)})
        if not valid_es:
            Hd[tau] = torch.zeros(N, 0, device=device)
            Wd[tau] = torch.zeros(0, device=device)
            Dvd[tau] = torch.ones(N, device=device)
            Ded[tau] = torch.zeros(0, device=device)
            continue
        E = len(valid_es)
        H = torch.zeros(N, E, device=device)
        W = torch.ones(E, device=device)
        for ei, e in enumerate(valid_es):
            for nid in e['nodes']:
                H[nmap[nid], ei] = 1.0
            W[ei] = e['weight']
        Dv = torch.sum(H * W.unsqueeze(0), dim=1) + 1e-8
        De = torch.sum(H, dim=0) + 1e-8
        Hd[tau] = H
        Wd[tau] = W
        Dvd[tau] = Dv
        Ded[tau] = De
    return Hd, Wd, Dvd, Ded

def has_valid_hyperedges(H_dict: Dict) -> bool:
    return any(H_dict[tau].size(1) > 0 for tau in ['MO', 'OM', 'CO'])

def load_hypergraph(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def encode_question(question: str, choices: List[str], tokenizer, model, device):
    # 简单的拼接 question + choices
    cs = " ".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
    text = f"{question} Choices: {cs}"
    inp = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp)
    return out.last_hidden_state[:, 0, :]

def find_hg_file(vid: str, hg_dir: str) -> Optional[str]:
    # 尝试找对应的超图 json
    clean = vid.replace('/', '_').replace('\\', '_').replace(':', '_')
    p = os.path.join(hg_dir, f"{clean}.json")
    if os.path.exists(p):
        return p
    for root, dirs, files in os.walk(hg_dir):
        if f"{clean}.json" in files:
            return os.path.join(root, f"{clean}.json")
        if f"{vid}.json" in files:
            return os.path.join(root, f"{vid}.json")
    return None

# ===== LLM stuff =====
_llm_tok = None
_llm_model = None

def init_llm(cache_dir='ckpt'):
    global _llm_tok, _llm_model
    if _llm_model is not None:
        return _llm_tok, _llm_model
    print("Loading local LLM (Qwen2.5-3B-Instruct)...")
    mname = "Qwen/Qwen2.5-3B-Instruct"
    try:
        _llm_tok = AutoTokenizer.from_pretrained(mname, cache_dir=cache_dir, trust_remote_code=True)
        _llm_model = AutoModelForCausalLM.from_pretrained(
            mname, cache_dir=cache_dir,
            torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        _llm_model.eval()
        if _llm_tok.pad_token is None:
            _llm_tok.pad_token = _llm_tok.eos_token
        print(f"LLM loaded on {_llm_model.device}.")
        return _llm_tok, _llm_model
    except Exception as e:
        print(f"LLM load error: {e}")
        return None, None

def answer_with_classifier(model, x, Hd, Wd, Dvd, Ded, q_emb, device):
    with torch.no_grad():
        logits, _ = model(x, Hd, Wd, Dvd, Ded, q_emb,
                          hypergraph=None, node_to_idx=None, type_label=None)
        pred_idx = logits.argmax().item()
        pred = chr(65 + pred_idx)
    return pred, logits

def hyperedge_expansion(seeds, hg, node_embs, q_emb, M=5):
    cand = []
    nmap = {n['id']: i for i, n in enumerate(hg['nodes'])}
    seed_set = set(seeds.tolist())
    all_e = get_all_hyperedges(hg)
    seen = set()
    for e in all_e:
        enodes = e.get('nodes', [])
        for nid in enodes:
            if nid in nmap and nmap[nid] in seed_set:
                key = tuple(sorted(enodes))
                if key not in seen:
                    seen.add(key)
                    cand.append(e)
                break
    scores = []
    for e in cand:
        valid = [nmap[n] for n in e.get('nodes', []) if n in nmap]
        if len(valid) < 2:
            continue
        e_emb = node_embs[valid].mean(dim=0, keepdim=True)
        s = F.cosine_similarity(q_emb, e_emb, dim=1).item()
        scores.append((e, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [e for e, _ in scores[:M]]

def build_subgraph(selected_edges, hg):
    nids = set()
    for e in selected_edges:
        nids.update(e.get('nodes', []))
    nmap = {n['id']: i for i, n in enumerate(hg['nodes'])}
    valid_idx = [nmap[n] for n in nids if n in nmap]
    return {
        'nodes': [hg['nodes'][i] for i in valid_idx],
        'node_indices': valid_idx,
        'hyperedges': selected_edges
    }

def answer_with_llm(question, choices, context_nodes, use_local_llm=True):
    global _llm_tok, _llm_model
    if use_local_llm:
        if _llm_tok is None or _llm_model is None:
            init_llm()
        if _llm_tok is not None and _llm_model is not None:
            return _answer_qwen(question, choices, context_nodes)
    return _fallback(question, choices)

def _answer_qwen(question, choices, context_nodes):
    global _llm_tok, _llm_model
    ctx_parts = []
    for i, node in enumerate(context_nodes[:8]):
        txt = node.get('text', node.get('S_i', node.get('summary', '')))
        if txt:
            ctx_parts.append(f"Event {i+1}: {txt[:400]}")
    ctx = "\n".join(ctx_parts) if ctx_parts else "No relevant events."
    cho = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    prompt = f"""Based on the following story events, answer the multiple-choice question.

Context Events:
{ctx}

Question: {question}

Choices:
{cho}

Instructions: Output ONLY the letter (A, B, C, D, or E). Do not include any explanation.

Answer:"""
    try:
        msgs = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on given context."},
            {"role": "user", "content": prompt}
        ]
        text = _llm_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = _llm_tok(text, return_tensors="pt", truncation=True, max_length=2048)
        inp = {k: v.to(_llm_model.device) for k, v in inp.items()}
        with torch.no_grad():
            out = _llm_model.generate(
                **inp, max_new_tokens=20,
                temperature=0.1, do_sample=True,
                pad_token_id=_llm_tok.pad_token_id,
                eos_token_id=_llm_tok.eos_token_id
            )
        resp = _llm_tok.decode(out[0], skip_special_tokens=True)
        # 找最后一个字母
        for ch in reversed(resp.strip()):
            if ch in ['A', 'B', 'C', 'D', 'E']:
                return ch
        return 'A'
    except Exception as e:
        print(f"LLM error: {e}")
        return _fallback(question, choices)

def _fallback(question, choices):
    # 基于词重叠的简单匹配
    stop = {'what', 'how', 'why', 'does', 'do', 'is', 'are', 'was', 'were',
            'the', 'a', 'an', 'to', 'for', 'of', 'with', 'on', 'at', 'from',
            'by', 'in', 'into', 'through', 'during', 'including', 'which'}
    qw = set([w.lower() for w in question.split() if w.lower() not in stop and len(w) > 2])
    scores = []
    for c in choices:
        cc = c.split('.', 1)[-1] if '.' in c else c
        cw = set([w.lower() for w in cc.split() if len(w) > 2])
        scores.append(len(qw & cw))
    best = max(range(len(scores)), key=lambda i: scores[i])
    return chr(65 + best)

def answer_with_retrieval_and_llm(q_proj, node_feats, hg, question, choices,
                                  top_k=10, M=5, use_local_llm=True, hidden_dim=512):
    sims = F.cosine_similarity(q_proj, node_feats, dim=1)
    _, seeds = torch.topk(sims, min(top_k, len(sims)))
    sel_edges = hyperedge_expansion(seeds, hg, node_feats, q_proj, M=M)
    sub = build_subgraph(sel_edges, hg)
    if sub['nodes'] and len(sub['nodes']) >= 2:
        nmap = {n['id']: i for i, n in enumerate(hg['nodes'])}
        sub_idx = [nmap[n['id']] for n in sub['nodes'] if n['id'] in nmap]
        if sub_idx:
            z_sub = node_feats[sub_idx]
            readout = HypergraphReadout(hidden_dim=hidden_dim, readout_type='attention')
            ctx, _ = readout(z_sub, q_proj, return_weights=True)
            q_proj = q_proj + ctx.unsqueeze(0) * 0.3   # 简单融合
    pred = answer_with_llm(question, choices, sub['nodes'], use_local_llm=use_local_llm)
    return pred, sub['nodes']

# ===== 模型加载（修复前缀） =====
def load_model(ckpt_path, model_type, device, type_aware=True):
    print(f"Loading model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get('model_state_dict', ckpt)
    # 去掉可能的 module. 前缀
    if all(k.startswith('module.') for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}

    cfg = ckpt.get('model_config', {})
    in_ch = cfg.get('in_channels', 1024)
    hid = cfg.get('hidden_channels', 512)
    nlayers = cfg.get('num_layers', 3)
    nans = cfg.get('num_answers', 5)

    hgne = HGNE(in_channels=in_ch, hidden_channels=hid, num_layers=nlayers)
    # 如果 q_proj 不在 checkpoint 里，做一个近似初始化
    if 'q_proj.weight' not in sd and 'hgne.q_proj.weight' not in sd:
        print("  q_proj missing, init with identity-like")
        with torch.no_grad():
            d = min(1024, hid)
            hgne.q_proj.weight.data[:, :d] = torch.eye(d)
            hgne.q_proj.bias.data.zero_()

    if model_type == 'e2e':
        model = EndToEndModel(hgne, hidden_dim=hid, num_answers=nans, use_har=True)
        use_clf = True
        # 检测是否需要加 'hgne.' 前缀
        if any(k.startswith('conv') for k in sd.keys()):
            nsd = {}
            for k, v in sd.items():
                nsd[f'hgne.{k}'] = v
            sd = nsd
            print("  Added 'hgne.' prefix for E2E model.")
    else:
        model = hgne
        use_clf = False

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  missing keys: {missing[:5]}...")
    if unexpected:
        print(f"  unexpected keys: {unexpected[:5]}...")
    model.to(device)
    model.eval()
    return model, use_clf, {
        'type_aware': True,
        'hidden_dim': hid,
        'num_layers': nlayers,
        'in_channels': in_ch,
        'num_answers': nans
    }

# ===== 主推理 =====
def run_qa(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\nCausalHyperGraph HAR Inference\n{'='*60}")
    print(f"Device: {dev}, model: {args.model_type}")

    with open(args.questions, 'r', encoding='utf-8') as f:
        qs = json.load(f)
    print(f"Loaded {len(qs)} questions")

    tok = AutoTokenizer.from_pretrained('bert-large-uncased', cache_dir=args.cache_dir)
    bert = AutoModel.from_pretrained('bert-large-uncased', cache_dir=args.cache_dir)
    bert.to(dev).eval()

    model, use_clf, mdl_cfg = load_model(
        args.checkpoint, args.model_type, dev, type_aware=not args.no_type_aware
    )
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Use classifier: {use_clf}")

    if not args.no_llm and args.use_local_llm and not use_clf:
        init_llm(args.cache_dir)

    results = []
    correct, total, skipped = 0, 0, 0
    det_stats = defaultdict(int)
    type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    t0 = time.time()

    for idx, q in enumerate(tqdm(qs, desc="QA")):
        vid = q.get('vid', q.get('video_id', q.get('vid_name', '')))
        hg_path = find_hg_file(vid, args.hypergraph_dir)
        if hg_path is None:
            skipped += 1
            det_stats['missing_hg'] += 1
            results.append({**q, 'predicted': 'X', 'correct': False, 'error': 'missing_hg'})
            total += 1
            continue
        try:
            hg = load_hypergraph(hg_path)
        except Exception as e:
            skipped += 1
            det_stats['load_err'] += 1
            results.append({**q, 'predicted': 'X', 'correct': False, 'error': f'load: {e}'})
            total += 1
            continue

        q_type = q.get('type', 'unknown')
        choices = q.get('choices', ['A', 'B', 'C', 'D', 'E'])
        while len(choices) < 5:
            choices.append(f"Opt {len(choices)+1}")

        try:
            Hd, Wd, Dvd, Ded = build_type_aware_tensors(hg, dev)
            if not has_valid_hyperedges(Hd):
                skipped += 1
                det_stats['no_valid_edges'] += 1
                results.append({**q, 'predicted': 'X', 'correct': False, 'error': 'no_valid_edges'})
                total += 1
                continue

            emb_dim = mdl_cfg.get('in_channels', 1024)
            node_embs = []
            for n in hg['nodes']:
                emb = n.get('embedding', [])
                if not emb:
                    emb = [0.0] * emb_dim
                elif len(emb) < emb_dim:
                    emb = emb + [0.0] * (emb_dim - len(emb))
                node_embs.append(emb[:emb_dim])
            x = torch.tensor(node_embs, dtype=torch.float32).to(dev)

            q_emb = encode_question(q['question'], choices, tok, bert, dev)

            with torch.no_grad():
                if use_clf:
                    logits, _ = model(x, Hd, Wd, Dvd, Ded, q_emb,
                                      hypergraph=None, node_to_idx=None, type_label=None)
                    z, qp = None, None
                else:
                    z, _ = model(x, Hd, Wd, Dvd, Ded)
                    qp = model.project_question(q_emb)
                    logits = None

            if use_clf and args.no_llm:
                pred, _ = answer_with_classifier(model, x, Hd, Wd, Dvd, Ded, q_emb, dev)
                conf = 1.0
                det_stats['classifier'] += 1
                sub_nodes = []
            else:
                pred, sub_nodes = answer_with_retrieval_and_llm(
                    qp, z, hg, q['question'], choices,
                    top_k=args.top_k, M=args.M,
                    use_local_llm=args.use_local_llm and not args.no_llm,
                    hidden_dim=mdl_cfg.get('hidden_dim', 512)
                )
                conf = 1.0
                det_stats['llm'] += 1

            gt = q.get('option', q.get('answer', ''))
            if isinstance(gt, int):
                gt = chr(65 + gt)
            elif isinstance(gt, str) and gt.isdigit():
                gt = chr(65 + int(gt))
            is_correct = (pred == gt) if gt else False
            if is_correct:
                correct += 1
            total += 1
            type_stats[q_type]['total'] += 1
            if is_correct:
                type_stats[q_type]['correct'] += 1

            results.append({**q, 'predicted': pred, 'correct': is_correct,
                            'confidence': conf, 'num_nodes': len(hg['nodes']),
                            'num_sub_nodes': len(sub_nodes) if sub_nodes else 0})
        except Exception as e:
            skipped += 1
            det_stats['runtime_err'] += 1
            results.append({**q, 'predicted': 'X', 'correct': False, 'error': str(e)})
            total += 1
            if args.verbose:
                import traceback
                traceback.print_exc()

        if idx % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    elapsed = time.time() - t0
    acc_excl = correct / max(total - skipped, 1) * 100 if total > skipped else 0
    acc_overall = correct / max(total, 1) * 100

    print(f"\n{'='*60}\nResults\n{'='*60}")
    print(f"Total: {total}, Correct: {correct}, Skipped: {skipped}")
    print(f"Acc (excl skip): {acc_excl:.2f}%")
    print(f"Acc (overall): {acc_overall:.2f}%")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'accuracy_excluding_skipped': acc_excl,
            'accuracy_overall': acc_overall,
            'correct': correct,
            'total': total,
            'skipped': skipped,
            'config': vars(args),
            'model_config': mdl_cfg,
            'type_stats': {k: dict(v) for k, v in type_stats.items()},
            'detailed_stats': dict(det_stats),
            'results': results
        }, f, ensure_ascii=False, indent=2)

    print(f"Saved to {args.output}")
    return acc_excl


def main():
    parser = argparse.ArgumentParser(description='CausalHyperGraph QA Inference')
    parser.add_argument('--questions', required=True)
    parser.add_argument('--hypergraph_dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache_dir', default='ckpt')
    parser.add_argument('--model_type', choices=['hgne', 'e2e'], default='e2e')
    parser.add_argument('--no_type_aware', action='store_true')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--M', type=int, default=5)
    parser.add_argument('--use_local_llm', action='store_true')
    parser.add_argument('--no_llm', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    a = parser.parse_args()
    if a.no_llm:
        a.use_local_llm = False
    run_qa(a)

if __name__ == '__main__':
    main()