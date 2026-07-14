# build_hypergraph.py
"""
build_hypergraph.py - 构建超图，用于论文 3.3 节
修复：严格因果、密度控制、动态置信度、ID比较、空列表保护等。
"""

import json, os, argparse, time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass, field

# ---- 自适应时间窗口 ----

def load_video_lengths(video_length_path: str) -> Dict[str, float]:
    """load video durations from json"""
    if not os.path.exists(video_length_path):
        print(f"  ⚠️  Video length file not found: {video_length_path}")
        return {}
    with open(video_length_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_adaptive_time_window(video_id: str, video_length=None, video_lengths=None, default_window=600):
    """
    根据视频类型/时长自适应窗口，详见论文 A.1
    """
    if video_length is None and video_lengths:
        video_length = video_lengths.get(video_id)
    vid_lower = video_id.lower()

    # 判断类型
    if any(kw in vid_lower for kw in ['got','game of thrones']):
        window = 1200   # 史诗类
    elif any(kw in vid_lower for kw in ['bigbang','big bang','friends']):
        window = 900    # 喜剧类
    elif any(kw in vid_lower for kw in ['imdb','douban']):
        if video_length:
            if video_length > 9000:      # >2.5h
                window = 1200
            elif video_length > 6000:    # >1.67h
                window = 900
            elif video_length > 4000:    # >1.1h
                window = 600
            else:
                window = 300
        else:
            window = 600
    else:
        window = default_window
    return window

# ---- small helpers ----
def is_valid_string(s: str) -> bool:
    """check if string is non-empty and not some placeholder"""
    if not s or not isinstance(s, str):
        return False
    stripped = s.strip()
    if not stripped:
        return False
    # 过滤明显的无效值
    if stripped.lower() in {'nan','none','null','na','-','--','n/a',''}:
        return False
    return True

def cosine_similarity(a: np.ndarray, b: np.ndarray, eps=1e-8) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ===== Event / Hyperedge data classes =====

@dataclass
class Event:
    id: str
    time_seconds: Optional[float] = None
    start_time: str = ""
    end_time: str = ""
    characters: List[str] = field(default_factory=list)  # P_i
    action: str = ""
    location: str = ""
    summary: str = ""   # S_i
    text: str = ""
    embedding: Optional[List[float]] = None

@dataclass
class Hyperedge:
    type: str          # 'MO', 'OM', 'CO'
    nodes: List[str]
    source_nodes: List[str] = field(default_factory=list)   # causes
    target_nodes: List[str] = field(default_factory=list)   # effects
    weight: float = 1.0
    character: str = ""
    causality_score: float = 0.0

# ===== Causality Verifier (MHCV) =====

class CausalityVerifier:
    """
    multi-stage heuristic causality verifier.
    阶段: temporal -> char continuity -> keywords -> embeddings
    """
    CAUSAL_KW = {
        'strong': [
            'because', 'therefore', 'thus', 'hence', 'consequently',
            'as a result', 'due to', 'owing to', 'so that', 'in order to',
            'lead to', 'result in', 'cause', 'trigger', 'bring about',
            'kill', 'die', 'death', 'murder', 'attack', 'fight',
            'betray', 'revenge', 'discover', 'reveal', 'confess',
            'decide', 'determine', 'force', 'compel', 'persuade'
        ],
        'weak': [
            'then', 'after', 'before', 'when', 'while', 'during',
            'leave', 'arrive', 'enter', 'exit', 'return',
            'agree', 'refuse', 'accept', 'deny', 'admit',
            'begin', 'start', 'stop', 'end', 'finish'
        ],
        'emotional': [
            'angry','furious','upset','happy','sad','afraid',
            'scared','worried','excited','surprised','shocked',
            'disappointed','frustrated','relieved','anxious','nervous'
        ]
    }

    def __init__(self, use_embeddings=True, sim_thresh=0.5, time_window=600, min_confidence_non_strict=0.65):
        self.use_emb = use_embeddings
        self.sim_thresh = sim_thresh
        self.time_win = time_window
        self.min_conf_ns = min_confidence_non_strict

    def verify_causality(self, src_events: List[Event], tgt_event: Event, strict=True) -> Tuple[bool, float]:
        conf = 0.0

        # stage 1: temporal order (must pass)
        if not self._check_temporal_order(src_events, tgt_event):
            return False, 0.0
        conf += 0.2

        # stage 2: character continuity (must for single cause)
        char_cont = self._check_character_continuity(src_events, tgt_event)
        if len(src_events) == 1 and not char_cont:
            return False, conf
        if char_cont:
            conf += 0.15

        # stage 3: causal keywords (must have at least some)
        kw_score = self._check_causal_keywords(src_events, tgt_event)
        if kw_score <= 0:
            return False, conf
        conf += kw_score * 0.30

        # stage 4: semantic similarity
        if self.use_emb:
            emb_score = self._check_semantic_similarity(src_events, tgt_event)
            if emb_score <= 0:
                return False, conf
            conf += emb_score * 0.35

        # decision
        if strict:
            is_causal = conf >= 0.50
        else:
            is_causal = conf >= self.min_conf_ns

        return is_causal, min(conf, 1.0)

    def _check_temporal_order(self, src_events: List[Event], tgt_event: Event) -> bool:
        """利用时间戳或ID顺序（数字比较）判断前后"""
        tgt_time = tgt_event.time_seconds

        def _num(e: Event) -> int:
            try:
                return int(e.id.split('_')[1]) if '_' in e.id else 0
            except:
                return 0

        tgt_num = _num(tgt_event)

        if tgt_time is not None:
            for s in src_events:
                st = s.time_seconds
                if st is not None:
                    if st > tgt_time:
                        return False
                else:
                    # fallback to id numbers
                    if _num(s) >= tgt_num:
                        return False
            return True
        else:
            # both no timestamps -> strict id order
            for s in src_events:
                if _num(s) >= tgt_num:
                    return False
            return True

    def _check_character_continuity(self, src_events, tgt_event) -> bool:
        tgt_chars = {c for c in tgt_event.characters if is_valid_string(c)}
        if not tgt_chars:
            return False
        for s in src_events:
            src_chars = {c for c in s.characters if is_valid_string(c)}
            if src_chars & tgt_chars:
                return True
        return False

    def _check_causal_keywords(self, src_events, tgt_event) -> float:
        s_text = " ".join([e.text.lower() for e in src_events])
        t_text = tgt_event.text.lower()
        comb = s_text + " " + t_text

        strong = sum(1 for kw in self.CAUSAL_KW['strong'] if kw in comb)
        weak = sum(1 for kw in self.CAUSAL_KW['weak'] if kw in comb)
        # emotion words
        emo_t = sum(1 for kw in self.CAUSAL_KW['emotional'] if kw in t_text)
        emo_s = sum(1 for kw in self.CAUSAL_KW['emotional'] if kw in s_text)
        emo = emo_t + 0.5 * emo_s

        score = (strong * 0.5 + weak * 0.2 + emo * 0.15) / 3.0
        return min(score, 1.0)

    def _check_semantic_similarity(self, src_events, tgt_event) -> float:
        src_embs = []
        for e in src_events:
            if e.embedding is not None and len(e.embedding) > 0:
                src_embs.append(e.embedding)
        if not src_embs or tgt_event.embedding is None or len(tgt_event.embedding)==0:
            return 0.0
        avg_src = np.mean(src_embs, axis=0)
        tgt_emb = np.array(tgt_event.embedding)
        sim = cosine_similarity(avg_src, tgt_emb)
        if sim > self.sim_thresh:
            return min((sim - self.sim_thresh) / (1 - self.sim_thresh), 1.0)
        return 0.0

    def _check_synergy(self, src_events: List[Event]) -> float:
        """检查多个源事件之间的协同"""
        if len(src_events) < 2:
            return 0.0
        synergy = 0.0
        n_checks = 0

        # time proximity
        times = [e.time_seconds for e in src_events if e.time_seconds is not None]
        if len(times) >= 2:
            span = max(times) - min(times)
            if span <= self.time_win / 3:
                synergy += 0.4
            else:
                synergy += 0.1
            n_checks += 1

        # location overlap
        locs = [e.location for e in src_events if is_valid_string(e.location)]
        if len(set(locs)) < len(locs) and len(locs) > 0:
            synergy += 0.3
        n_checks += 1

        # character interaction
        all_chars = []
        for e in src_events:
            all_chars.extend([c for c in e.characters if is_valid_string(c)])
        cnt = {}
        for c in all_chars:
            cnt[c] = cnt.get(c,0)+1
        multi = sum(1 for v in cnt.values() if v>1)
        if multi > 0:
            synergy += 0.3
        n_checks += 1

        if n_checks > 0:
            synergy /= n_checks
        return min(synergy, 1.0)

# ===== Hypergraph Builder =====

class HypergraphBuilder:
    """三阶段构建：候选->验证->密度控制"""

    def __init__(self, time_window=600, mo_size=3, om_size=3,
                 co_window_size=None, co_max_size=50,
                 similarity_threshold=0.5, strict_causality=True,
                 min_events_for_causal=4, video_id=None, video_lengths=None,
                 max_mo_ratio=0.3, max_om_ratio=0.3, enable_debug=False):
        # adaptive time window
        if video_id and video_lengths:
            vlen = video_lengths.get(video_id)
            self.time_window = get_adaptive_time_window(video_id, vlen, video_lengths, time_window)
            if self.time_window != time_window:
                print(f"  📐 Adaptive window for {video_id}: {self.time_window}s (default {time_window}s, len {vlen}s)")
        else:
            self.time_window = time_window

        self.mo_size = mo_size
        self.om_size = om_size
        self.co_win_sz = co_window_size
        self.co_max_sz = co_max_size
        self.min_ev_causal = min_events_for_causal
        self.max_mo_ratio = max_mo_ratio
        self.max_om_ratio = max_om_ratio
        self.debug = enable_debug

        self.verifier = CausalityVerifier(
            use_embeddings=True, sim_thresh=similarity_threshold,
            time_window=self.time_window, min_confidence_non_strict=0.65
        )
        self.strict = strict_causality

    def build(self, events: List[Event]) -> Tuple[List[Hyperedge],List[Hyperedge],List[Hyperedge]]:
        E_CO = self._build_CO(events)
        E_MO, E_OM = self._build_causal(events)
        E_MO = self._dedup(E_MO)
        E_OM = self._dedup(E_OM)
        E_CO = self._dedup(E_CO)
        return E_MO, E_OM, E_CO

    def _build_CO(self, events):
        # 角色共现超边
        char_ev = defaultdict(list)
        all_chars_found = set()
        for e in events:
            for c in e.characters:
                all_chars_found.add(c)
                if is_valid_string(c):
                    char_ev[c].append(e.id)

        if self.debug:
            invalid = all_chars_found - set(char_ev.keys())
            print(f"  [CO debug] total chars found: {len(all_chars_found)}, valid: {len(char_ev)}, invalid: {len(invalid)}")
            if invalid and len(invalid) <= 10:
                print(f"    filtered: {invalid}")

        E_CO = []
        for ch, nids in char_ev.items():
            if len(nids) < 2:
                continue
            nids_sorted = sorted(nids)
            win = self.co_win_sz if self.co_win_sz is not None else self.co_max_sz
            if len(nids_sorted) > win:
                step = max(1, win // 2)
                for i in range(0, len(nids_sorted)-1, step):
                    chunk = nids_sorted[i:i+win]
                    if len(chunk) >= 2:
                        E_CO.append(Hyperedge(type='CO', nodes=chunk, character=ch, weight=1.0))
            else:
                E_CO.append(Hyperedge(type='CO', nodes=nids_sorted, character=ch, weight=1.0))

        if self.debug:
            print(f"  [CO debug] built {len(E_CO)} CO edges")
        return E_CO

    def _build_causal(self, events):
        if len(events) < self.min_ev_causal:
            return [], []
        E_MO = self._build_MO(events, 'global', check_char_overlap=False)
        E_OM = self._build_OM(events, 'global', check_char_overlap=False)
        return E_MO, E_OM

    def _build_MO(self, events, char, check_char_overlap=False):
        E_MO = []
        n = len(events)
        candidates = []

        for i in range(n):
            effect = events[i]
            causes = []
            # 向前收集时间窗口内的前置事件
            for j in range(i-1, -1, -1):
                cev = events[j]
                if check_char_overlap:
                    cev_chars = {c for c in cev.characters if is_valid_string(c)}
                    eff_chars = {c for c in effect.characters if is_valid_string(c)}
                    if cev_chars and eff_chars and not cev_chars & eff_chars:
                        continue
                if not self._within_window([cev], effect):
                    break
                causes.append(cev)
            causes.reverse()
            if len(causes) < 2:
                continue

            # 独立验证每个因
            indep = []
            for c in causes:
                ok, conf = self.verifier.verify_causality([c], effect, strict=self.strict)
                if ok:
                    indep.append((c, conf))
            # 子集协同验证
            verified = []
            if len(indep) >= 2:
                indep.sort(key=lambda x: x[1], reverse=True)
                subset_causes = [x[0] for x in indep]
                synergy_pairs = []
                for ii in range(len(subset_causes)):
                    for jj in range(ii+1, len(subset_causes)):
                        sub = [subset_causes[ii], subset_causes[jj]]
                        syn = self.verifier._check_synergy(sub)
                        if syn >= 0.3:
                            base = (indep[ii][1] + indep[jj][1]) / 2
                            enh = base * (1 + syn * 0.5)
                            synergy_pairs.append((sub, enh))
                if synergy_pairs:
                    synergy_pairs.sort(key=lambda x: x[1], reverse=True)
                    best_sub, best_score = synergy_pairs[0]
                    for cau in best_sub:
                        verified.append((cau, best_score / len(best_sub)))
                else:
                    verified = indep[:self.mo_size]
            else:
                verified = indep

            cause_final = [vc[0] for vc in verified]
            cause_ids = [e.id for e in cause_final]
            if len(cause_ids) < 2:
                continue

            avg_conf = float(np.mean([vc[1] for vc in verified])) if verified else 0.0
            candidates.append((cause_ids, effect.id, avg_conf))

        # density control for MO
        max_mo = max(1, int(n * self.max_mo_ratio))
        if len(candidates) > max_mo:
            candidates.sort(key=lambda x: x[2], reverse=True)
            dropped = len(candidates) - max_mo
            candidates = candidates[:max_mo]
            if self.debug:
                print(f"  [MO dens] kept {max_mo}/{max_mo+dropped} (ratio {max_mo/n:.2f})")

        for cids, eid, conf in candidates:
            E_MO.append(Hyperedge(
                type='MO',
                nodes=cids + [eid],
                source_nodes=cids,
                target_nodes=[eid],
                weight=conf,
                character=char,
                causality_score=conf
            ))
        return E_MO

    def _build_OM(self, events, char, check_char_overlap=False):
        E_OM = []
        n = len(events)
        candidates = []

        for i in range(n):
            cause = events[i]
            effects = []
            for j in range(i+1, n):
                eff = events[j]
                if check_char_overlap:
                    c_chars = {c for c in cause.characters if is_valid_string(c)}
                    e_chars = {c for c in eff.characters if is_valid_string(c)}
                    if c_chars and e_chars and not c_chars & e_chars:
                        continue
                if not self._within_window([cause], eff):
                    break
                effects.append(eff)
            if len(effects) < 2:
                continue

            verified = []
            for ef in effects:
                ok, conf = self.verifier.verify_causality([cause], ef, strict=self.strict)
                if ok:
                    verified.append((ef, conf))
            if len(verified) < 2:
                continue

            verified.sort(key=lambda x: x[1], reverse=True)
            if len(verified) > self.om_size:
                verified = verified[:self.om_size]

            eff_final = [v[0] for v in verified]
            eff_ids = [e.id for e in eff_final]
            if len(eff_ids) < 2:
                continue

            avg_conf = float(np.mean([v[1] for v in verified])) if verified else 0.0
            candidates.append((cause.id, eff_ids, avg_conf))

        # density control for OM
        max_om = max(1, int(n * self.max_om_ratio))
        if len(candidates) > max_om:
            candidates.sort(key=lambda x: x[2], reverse=True)
            dropped = len(candidates) - max_om
            candidates = candidates[:max_om]
            if self.debug:
                print(f"  [OM dens] kept {max_om}/{max_om+dropped} (ratio {max_om/n:.2f})")

        for cid, eids, conf in candidates:
            E_OM.append(Hyperedge(
                type='OM',
                nodes=[cid] + eids,
                source_nodes=[cid],
                target_nodes=eids,
                weight=conf,
                character=char,
                causality_score=conf
            ))
        return E_OM

    def _within_window(self, src_events, tgt_event):
        tgt_time = tgt_event.time_seconds
        if tgt_time is None:
            return all(e.id < tgt_event.id for e in src_events)
        for s in src_events:
            st = s.time_seconds
            if st is None: continue
            if st > tgt_time: return False
            if (tgt_time - st) > self.time_window: return False
        # all have no time? use id order
        if all(e.time_seconds is None for e in src_events):
            return all(e.id < tgt_event.id for e in src_events)
        return True

    def _dedup(self, edges: List[Hyperedge]) -> List[Hyperedge]:
        seen = set()
        out = []
        for e in edges:
            if e.type in ('MO','OM'):
                key = (e.type, tuple(sorted(e.source_nodes)), tuple(sorted(e.target_nodes)))
            else:
                key = (e.type, tuple(sorted(e.nodes)))
            if key not in seen:
                seen.add(key)
                out.append(e)
        return out

# ===== serialization helper =====

def hyperedge_to_dict(edge: Hyperedge) -> dict:
    return {
        'type': edge.type,
        'nodes': edge.nodes,
        'source_nodes': edge.source_nodes,
        'target_nodes': edge.target_nodes,
        'weight': edge.weight,
        'character': edge.character,
        'causality_score': edge.causality_score
    }

def build_hypergraph(
        events: List[dict],
        time_window=600, mo_size=3, om_size=3,
        co_window_size=None, co_max_size=50,
        similarity_threshold=0.5, strict_causality=True,
        min_events_for_causal=4, video_id=None,
        video_lengths=None, use_adaptive=True,
        max_mo_ratio=0.3, max_om_ratio=0.3, enable_debug=False
) -> Tuple[List[dict], List[dict], List[dict], dict]:
    """External interface to build hypergraph from event dicts"""
    ev_objs = [Event(
        id=e.get('id',''),
        time_seconds=e.get('time_seconds'),
        start_time=e.get('start_time',''),
        end_time=e.get('end_time',''),
        characters=e.get('P_i',[]),
        action=e.get('A_i',''),
        location=e.get('L_i',''),
        summary=e.get('S_i',''),
        text=e.get('text',''),
        embedding=e.get('embedding')
    ) for e in events]

    builder = HypergraphBuilder(
        time_window=time_window, mo_size=mo_size, om_size=om_size,
        co_window_size=co_window_size, co_max_size=co_max_size,
        similarity_threshold=similarity_threshold, strict_causality=strict_causality,
        min_events_for_causal=min_events_for_causal,
        video_id=video_id if use_adaptive else None,
        video_lengths=video_lengths if use_adaptive else None,
        max_mo_ratio=max_mo_ratio, max_om_ratio=max_om_ratio,
        enable_debug=enable_debug
    )

    E_MO, E_OM, E_CO = builder.build(ev_objs)

    MO_dicts = [hyperedge_to_dict(e) for e in E_MO]
    OM_dicts = [hyperedge_to_dict(e) for e in E_OM]
    CO_dicts = [hyperedge_to_dict(e) for e in E_CO]

    # compute stats
    mo_scores = [e.causality_score for e in E_MO]
    om_scores = [e.causality_score for e in E_OM]
    mo_sizes = [len(e.source_nodes) for e in E_MO]
    om_sizes = [len(e.target_nodes) for e in E_OM]

    stats = {
        'num_events': len(events),
        'num_mo_edges': len(MO_dicts),
        'num_om_edges': len(OM_dicts),
        'num_co_edges': len(CO_dicts),
        'total_hyperedges': len(MO_dicts)+len(OM_dicts)+len(CO_dicts),
        'mo_avg_causality_score': float(np.mean(mo_scores)) if mo_scores else 0,
        'om_avg_causality_score': float(np.mean(om_scores)) if om_scores else 0,
        'mo_avg_weight': float(np.mean([e.weight for e in E_MO])) if E_MO else 0,
        'om_avg_weight': float(np.mean([e.weight for e in E_OM])) if E_OM else 0,
        'mo_avg_size': float(np.mean(mo_sizes)) if mo_sizes else 0,
        'om_avg_size': float(np.mean(om_sizes)) if om_sizes else 0,
        'mo_min_size': int(np.min(mo_sizes)) if mo_sizes else 0,
        'mo_max_size': int(np.max(mo_sizes)) if mo_sizes else 0,
        'om_min_size': int(np.min(om_sizes)) if om_sizes else 0,
        'om_max_size': int(np.max(om_sizes)) if om_sizes else 0,
        'strict_causality': strict_causality,
        'time_window': builder.time_window,
        'mo_size_limit': mo_size,
        'om_size_limit': om_size,
        'max_mo_ratio': max_mo_ratio,
        'max_om_ratio': max_om_ratio,
        'co_window_size': co_window_size,
        'co_max_size': co_max_size,
    }
    return MO_dicts, OM_dicts, CO_dicts, stats

# ---- 命令行入口 ----
def main():
    parser = argparse.ArgumentParser(description='CausalHyperGraph builder (论文3.3)')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--time_window', type=int, default=600)
    parser.add_argument('--mo_size', type=int, default=3)
    parser.add_argument('--om_size', type=int, default=3)
    parser.add_argument('--co_window_size', type=int, default=None)
    parser.add_argument('--co_max_size', type=int, default=50)
    parser.add_argument('--similarity_threshold', type=float, default=0.5)
    parser.add_argument('--strict', action='store_true', default=True)
    parser.add_argument('--no_strict', action='store_true')
    parser.add_argument('--min_events', type=int, default=4)
    parser.add_argument('--max_mo_ratio', type=float, default=0.3)
    parser.add_argument('--max_om_ratio', type=float, default=0.3)
    parser.add_argument('--enable_debug', action='store_true')
    parser.add_argument('--video_length', type=str, default=None)
    parser.add_argument('--video_id', type=str, default=None)
    parser.add_argument('--no_adaptive_window', action='store_true')
    args = parser.parse_args()

    strict = not args.no_strict if args.no_strict else args.strict

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    events = data.get('events', data.get('nodes', []))
    if not events:
        print("❌ No events found")
        return

    vlengths = None
    if not args.no_adaptive_window:
        if args.video_length and os.path.exists(args.video_length):
            vlengths = load_video_lengths(args.video_length)
    vid = args.video_id
    if not vid and not args.no_adaptive_window:
        vid = Path(args.input).stem

    print(f"\n{'='*60}")
    print(f"Hypergraph Construction Config")
    print(f"{'='*60}")
    print(f"  Events: {len(events)}")
    print(f"  Time window: {args.time_window}s, adaptive: {'on' if not args.no_adaptive_window else 'off'}")
    print(f"  Strict causality: {strict}")
    print(f"  MO/OM size limits: {args.mo_size}/{args.om_size}, ratio: {args.max_mo_ratio}/{args.max_om_ratio}")
    print(f"{'='*60}\n")

    t0 = time.time()
    MO, OM, CO, stats = build_hypergraph(
        events=events, time_window=args.time_window,
        mo_size=args.mo_size, om_size=args.om_size,
        co_window_size=args.co_window_size, co_max_size=args.co_max_size,
        similarity_threshold=args.similarity_threshold,
        strict_causality=strict,
        min_events_for_causal=args.min_events,
        video_id=vid if not args.no_adaptive_window else None,
        video_lengths=vlengths if not args.no_adaptive_window else None,
        max_mo_ratio=args.max_mo_ratio,
        max_om_ratio=args.max_om_ratio,
        enable_debug=args.enable_debug
    )
    elapsed = time.time() - t0

    hg_out = {
        'nodes': events,
        'hyperedges': MO + OM + CO,
        'stats': stats
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(hg_out, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}\nResults\n{'='*60}")
    print(f"  Build time: {elapsed:.1f}s")
    print(f"  MO edges: {stats['num_mo_edges']} (ratio {stats['num_mo_edges']/len(events):.2f})")
    print(f"    avg weight: {stats['mo_avg_weight']:.3f}, avg size: {stats['mo_avg_size']:.1f}")
    print(f"  OM edges: {stats['num_om_edges']} (ratio {stats['num_om_edges']/len(events):.2f})")
    print(f"    avg weight: {stats['om_avg_weight']:.3f}, avg size: {stats['om_avg_size']:.1f}")
    print(f"  CO edges: {stats['num_co_edges']}")
    print(f"  Total: {stats['total_hyperedges']}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()