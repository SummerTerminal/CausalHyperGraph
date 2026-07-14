"""
build_special_testset.py - 构建 MTO-Test 和 OMT-Test 专项集 (paper 4.1.1)
"""

import os, json, re, argparse, random
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional


class SpecTestBuilder:
    """builder for MTO and OMT specialized test sets"""
    def __init__(self, hg_dir: str):
        self.hg_dir = hg_dir
        self.hgs = {}           # vid -> hypergraph dict
        self._load_all()

    def _load_all(self):
        # 把目录下所有json超图加载进来
        for root, _, files in os.walk(self.hg_dir):
            for f in files:
                if f.endswith('.json') and not f.startswith('_'):
                    vid = os.path.splitext(f)[0]
                    p = os.path.join(root, f)
                    try:
                        with open(p, 'r', encoding='utf-8') as fp:
                            self.hgs[vid] = json.load(fp)
                    except Exception as e:
                        print(f"skip {f}: {e}")
        print(f"loaded {len(self.hgs)} hypergraphs")

    def _has_mo(self, hg: Dict) -> bool:
        edges = hg.get('hyperedges', {})
        if isinstance(edges, dict):
            return len(edges.get('MO', [])) > 0
        return any(e.get('type') == 'MO' for e in edges)

    def _has_om(self, hg: Dict) -> bool:
        edges = hg.get('hyperedges', {})
        if isinstance(edges, dict):
            return len(edges.get('OM', [])) > 0
        return any(e.get('type') == 'OM' for e in edges)

    def _count_mo(self, hg: Dict) -> int:
        edges = hg.get('hyperedges', {})
        if isinstance(edges, dict):
            return len(edges.get('MO', []))
        return sum(1 for e in edges if e.get('type') == 'MO')

    def _count_om(self, hg: Dict) -> int:
        edges = hg.get('hyperedges', {})
        if isinstance(edges, dict):
            return len(edges.get('OM', []))
        return sum(1 for e in edges if e.get('type') == 'OM')

    def _edge_sz(self, edge: Dict) -> int:
        return len(edge.get('nodes', []))

    def _large_edges(self, hg: Dict, etype: str, min_sz=3) -> List[Dict]:
        edges = hg.get('hyperedges', {})
        out = []
        if isinstance(edges, dict):
            for e in edges.get(etype, []):
                if self._edge_sz(e) >= min_sz:
                    out.append(e)
        else:
            for e in edges:
                if e.get('type') == etype and self._edge_sz(e) >= min_sz:
                    out.append(e)
        return out

    def _is_mto_q(self, q: str) -> bool:
        # check if question needs multi-cause reasoning
        pats = [r'why did', r'why does', r'what caused', r'what led to',
                r'what made', r'what resulted in', r'what contributed to',
                r'what were the reasons', r'what factors', r'due to what',
                r'why was', r'why were', r'what brought about',
                r'what combination', r'what together', r'what jointly']
        return any(re.search(p, q.lower()) for p in pats)

    def _is_omt_q(self, q: str) -> bool:
        # one-cause-multiple-effects keywords
        pats = [r'what happened after', r'what resulted from', r'consequences of',
                r'effects of', r'impact of', r'what followed', r'what came after',
                r'what did.*cause', r'what chain of events', r'what repercussions',
                r'what was the aftermath', r'what were the results',
                r'what changes occurred', r'what subsequent events']
        return any(re.search(p, q.lower()) for p in pats)

    def filter_qs(self, questions: List[Dict], ttype: str = 'mto') -> List[Dict]:
        """
        filter questions based on hypergraph structure and keyword matching
        """
        filtered = []
        for q in questions:
            vid = q.get('vid', q.get('video_id', ''))
            if vid not in self.hgs:
                continue
            hg = self.hgs[vid]

            if ttype == 'mto':
                if not self._has_mo(hg) or not self._is_mto_q(q.get('question', '')):
                    continue
                cnt = self._count_mo(hg)
                if cnt < 2:
                    continue
                large = self._large_edges(hg, 'MO', min_sz=3)
                if not large:
                    continue
                q['_mo_cnt'] = cnt
                q['_max_mo_sz'] = max(self._edge_sz(e) for e in large)
                q['_ttype'] = 'mto'
                filtered.append(q)

            elif ttype == 'omt':
                if not self._has_om(hg) or not self._is_omt_q(q.get('question', '')):
                    continue
                cnt = self._count_om(hg)
                if cnt < 2:
                    continue
                large = self._large_edges(hg, 'OM', min_sz=3)
                if not large:
                    continue
                q['_om_cnt'] = cnt
                q['_max_om_sz'] = max(self._edge_sz(e) for e in large)
                q['_ttype'] = 'omt'
                filtered.append(q)

        return filtered

    def build(self, questions: List[Dict], max_samples=2400, seed=42):
        """build both MTO and OMT testsets"""
        random.seed(seed)
        mto_c = self.filter_qs(questions, 'mto')
        omt_c = self.filter_qs(questions, 'omt')
        print(f"MTO candidates: {len(mto_c)}, OMT candidates: {len(omt_c)}")

        # 挑复杂的：超边越大的越优先
        mto_c.sort(key=lambda x: x.get('_max_mo_sz', 0), reverse=True)
        omt_c.sort(key=lambda x: x.get('_max_om_sz', 0), reverse=True)

        half = max_samples // 2
        mto_sel = mto_c[:min(half, len(mto_c))]
        omt_sel = omt_c[:min(half, len(omt_c))]
        random.shuffle(mto_sel)
        random.shuffle(omt_sel)
        return {'mto': mto_sel, 'omt': omt_sel}


def build_file(qpath: str, hg_dir: str, out_dir: str, max_samples=2400, seed=42):
    with open(qpath, 'r', encoding='utf-8') as f:
        qs = json.load(f)
    print(f"loaded {len(qs)} questions")

    builder = SpecTestBuilder(hg_dir)
    tset = builder.build(qs, max_samples, seed)
    os.makedirs(out_dir, exist_ok=True)

    mto_f = os.path.join(out_dir, 'mto_testset.json')
    omt_f = os.path.join(out_dir, 'omt_testset.json')
    with open(mto_f, 'w', encoding='utf-8') as f:
        json.dump(tset['mto'], f, ensure_ascii=False, indent=2)
    with open(omt_f, 'w', encoding='utf-8') as f:
        json.dump(tset['omt'], f, ensure_ascii=False, indent=2)

    # quick stats
    n_mto = len(tset['mto'])
    n_omt = len(tset['omt'])
    avg_mo = sum(q.get('_max_mo_sz', 0) for q in tset['mto']) / max(n_mto, 1)
    avg_om = sum(q.get('_max_om_sz', 0) for q in tset['omt']) / max(n_omt, 1)
    stats = {'total_mto': n_mto, 'total_omt': n_omt, 'avg_mo_size': avg_mo, 'avg_om_size': avg_om}
    with open(os.path.join(out_dir, 'testset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}\nDone!\n{'='*60}")
    print(f"MTO-Test: {n_mto} samples, avg MO size {avg_mo:.1f}")
    print(f"OMT-Test: {n_omt} samples, avg OM size {avg_om:.1f}")
    print(f"saved to {out_dir}")
    return tset


def main():
    parser = argparse.ArgumentParser(description='build causal test sets')
    parser.add_argument('--questions', required=True)
    parser.add_argument('--hypergraph_dir', required=True)
    parser.add_argument('--output_dir', default='experiments/special_testsets')
    parser.add_argument('--max_samples', type=int, default=2400)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    build_file(args.questions, args.hypergraph_dir, args.output_dir, args.max_samples, args.seed)


if __name__ == '__main__':
    main()