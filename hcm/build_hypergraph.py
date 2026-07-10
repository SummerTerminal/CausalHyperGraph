"""
build_hypergraph.py - 超图构建模块

修复内容：
1. 严格因果验证（strict_causality=True 默认）
2. 密度控制（max_mo_ratio/max_om_ratio）
3. 动态置信度权重替代硬编码
4. 角色连续性从加分项改为必要门槛
5. 空列表保护消除 NumPy 警告
6. 角色提取修复 + CO 诊断日志
7. 清理冗余代码
"""

import json
import os
import argparse
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass, field


# ===== 自适应时间窗口 =====

def load_video_lengths(video_length_path: str) -> Dict[str, float]:
    """加载视频时长信息"""
    if not os.path.exists(video_length_path):
        print(f"  ⚠️  Video length file not found: {video_length_path}")
        return {}
    
    with open(video_length_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_adaptive_time_window(
    video_id: str,
    video_length: Optional[float] = None,
    video_lengths: Optional[Dict[str, float]] = None,
    default_window: int = 600
) -> int:
    """
    根据视频类型和时长自适应调整时间窗口
    
    论文 A.1 节：
    - 剧情片 Δt=10分钟
    - 悬疑片 Δt=5分钟
    - 喜剧片 Δt=15分钟
    - 史诗片 Δt=20分钟
    """
    if video_length is None and video_lengths:
        video_length = video_lengths.get(video_id)
    
    video_id_lower = video_id.lower()
    
    if any(keyword in video_id_lower for keyword in ['got', 'game of thrones']):
        window = 1200  # 史诗/奇幻剧集：20分钟
    elif any(keyword in video_id_lower for keyword in ['bigbang', 'big bang', 'friends']):
        window = 900   # 喜剧剧集：15分钟
    elif any(keyword in video_id_lower for keyword in ['imdb', 'douban']):
        if video_length:
            if video_length > 9000:      # >2.5小时：史诗
                window = 1200
            elif video_length > 6000:    # >1.67小时：剧情
                window = 900
            elif video_length > 4000:    # >1.1小时：标准
                window = 600
            else:                         # 短片
                window = 300
        else:
            window = 600
    else:
        window = default_window
    
    return window


# ===== 工具函数 =====

def is_valid_string(s: str) -> bool:
    """检查字符串是否有效"""
    if not s or not isinstance(s, str):
        return False
    s_stripped = s.strip()
    if not s_stripped:
        return False
    # 放宽：只过滤明确的无效值，保留单字符角色名
    invalid_values = {'nan', 'none', 'null', 'na', '-', '--', 'n/a', ''}
    if s_stripped.lower() in invalid_values:
        return False
    return True


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """计算余弦相似度（数值稳定版本）"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a < eps or norm_b < eps:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))


# ===== 数据结构 =====

@dataclass
class Event:
    """事件数据结构（符合论文定义 v_i = (t_i, P_i, A_i, L_i, S_i)）"""
    id: str
    time_seconds: Optional[float] = None
    start_time: str = ""
    end_time: str = ""
    characters: List[str] = field(default_factory=list)       # P_i
    action: str = ""                                           # A_i
    location: str = ""                                         # L_i
    summary: str = ""                                          # S_i
    text: str = ""
    embedding: Optional[List[float]] = None


@dataclass
class Hyperedge:
    """超边数据结构"""
    type: str  # 'MO', 'OM', 'CO'
    nodes: List[str]
    source_nodes: List[str] = field(default_factory=list)  # 因节点
    target_nodes: List[str] = field(default_factory=list)  # 果节点
    weight: float = 1.0  # 动态置信度权重
    character: str = ""
    causality_score: float = 0.0  # 因果必要性分数


# ===== 因果验证器 =====

class CausalityVerifier:
    """
    多阶段启发式因果验证器（Multi-stage Heuristic Causality Verifier, MHCV）
    
    阶段1: 时间顺序检查（必须通过）
    阶段2: 角色连续性检查（单因必要门槛，多因放宽）
    阶段3: 因果关键词匹配（必须通过）
    阶段4: 嵌入语义相似度检查（必须通过）
    """
    
    CAUSAL_KEYWORDS = {
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
            'angry', 'furious', 'upset', 'happy', 'sad', 'afraid',
            'scared', 'worried', 'excited', 'surprised', 'shocked',
            'disappointed', 'frustrated', 'relieved', 'anxious', 'nervous'
        ]
    }
    
    def __init__(
        self, 
        use_embeddings: bool = True, 
        similarity_threshold: float = 0.5, 
        time_window: int = 600,
        min_confidence_for_non_strict: float = 0.65
    ):
        self.use_embeddings = use_embeddings
        self.similarity_threshold = similarity_threshold
        self.time_window = time_window
        self.min_confidence_for_non_strict = min_confidence_for_non_strict
    
    def verify_causality(
        self, 
        source_events: List[Event], 
        target_event: Event,
        strict_mode: bool = True
    ) -> Tuple[bool, float]:
        confidence = 0.0
    
        # ===== 阶段1：时间顺序检查（必须通过） =====
        if not self._check_temporal_order(source_events, target_event):
            return False, 0.0
        confidence += 0.2
        
        # ===== 阶段2：角色连续性检查 =====
        # 单因情况必须有角色连续性；多因情况放宽（允许跨角色协同）
        char_continuity = self._check_character_continuity(source_events, target_event)
        if len(source_events) == 1 and not char_continuity:
            # 单因无角色关联，拒绝
            return False, confidence
        if char_continuity:
            confidence += 0.15
        
        # ===== 阶段3：语义因果关键词匹配（必须通过） =====
        keyword_score = self._check_causal_keywords(source_events, target_event)
        if keyword_score > 0:
            confidence += keyword_score * 0.30
        else:
            # 无因果关键词，直接拒绝
            return False, confidence
        
        # ===== 阶段4：嵌入语义相似度检查（必须通过） =====
        if self.use_embeddings:
            embedding_score = self._check_semantic_similarity(source_events, target_event)
            if embedding_score > 0:
                confidence += embedding_score * 0.35
            else:
                # 语义不相关，直接拒绝
                return False, confidence
        
        # ===== 最终判断 =====
        if strict_mode:
            is_causal = confidence >= 0.50
        else:
            is_causal = confidence >= self.min_confidence_for_non_strict
        
        return is_causal, min(confidence, 1.0)

    def _check_temporal_order(self, source_events: List[Event], target_event: Event) -> bool:
        """检查所有因事件的时间是否早于果事件"""
        target_time = target_event.time_seconds
        
        if target_time is None:
            for src in source_events:
                if src.id >= target_event.id:
                    return False
            return True
        
        for src in source_events:
            src_time = src.time_seconds
            if src_time is None:
                continue
            if src_time > target_time:
                return False
        
        return True
    
    def _check_character_continuity(
        self, source_events: List[Event], target_event: Event
    ) -> bool:
        """检查源事件和目标事件是否有角色重叠"""
        target_chars = {c for c in target_event.characters if is_valid_string(c)}
        if not target_chars:
            return False
        
        for src in source_events:
            src_chars = {c for c in src.characters if is_valid_string(c)}
            if src_chars & target_chars:
                return True
        
        return False
    
    def _check_causal_keywords(
        self, source_events: List[Event], target_event: Event
    ) -> float:
        """基于因果关键词的筛选"""
        combined_source_text = " ".join([e.text.lower() for e in source_events])
        target_text = target_event.text.lower()
        combined_text = combined_source_text + " " + target_text
        
        strong_count = sum(1 for kw in self.CAUSAL_KEYWORDS['strong'] if kw in combined_text)
        weak_count = sum(1 for kw in self.CAUSAL_KEYWORDS['weak'] if kw in combined_text)
        
        emotional_count = sum(1 for kw in self.CAUSAL_KEYWORDS['emotional'] if kw in target_text)
        emotional_in_source = sum(1 for kw in self.CAUSAL_KEYWORDS['emotional'] 
                                  if kw in combined_source_text)
        emotional_count += emotional_in_source * 0.5
        
        score = (strong_count * 0.5 + weak_count * 0.2 + emotional_count * 0.15) / 3.0
        
        return min(score, 1.0)
    
    def _check_semantic_similarity(
        self, source_events: List[Event], target_event: Event
    ) -> float:
        """基于嵌入向量的语义因果相似度"""
        source_embeddings = []
        for e in source_events:
            if e.embedding is not None and len(e.embedding) > 0:
                source_embeddings.append(e.embedding)
        
        if not source_embeddings or target_event.embedding is None:
            return 0.0
        if len(target_event.embedding) == 0:
            return 0.0
        
        avg_source_embedding = np.mean(source_embeddings, axis=0)
        target_embedding = np.array(target_event.embedding)
        
        similarity = cosine_similarity(avg_source_embedding, target_embedding)
        
        if similarity > self.similarity_threshold:
            return min((similarity - self.similarity_threshold) / (1 - self.similarity_threshold), 1.0)
        
        return 0.0

    def _check_synergy(self, source_events: List[Event]) -> float:
        """检查多个源事件之间是否存在协同关系"""
        if len(source_events) < 2:
            return 0.0
        
        synergy_score = 0.0
        num_checks = 0
        
        # 检查1: 时间接近性
        times = [e.time_seconds for e in source_events if e.time_seconds is not None]
        if len(times) >= 2:
            time_span = max(times) - min(times)
            max_allowed_span = self.time_window / 3
            if time_span <= max_allowed_span:
                synergy_score += 0.4
            else:
                synergy_score += 0.1
            num_checks += 1
        
        # 检查2: 场景共享性
        locations = [e.location for e in source_events if is_valid_string(e.location)]
        if len(set(locations)) < len(locations) and len(locations) > 0:
            synergy_score += 0.3
        num_checks += 1
        
        # 检查3: 角色交互性
        all_chars = []
        for e in source_events:
            all_chars.extend([c for c in e.characters if is_valid_string(c)])
        char_counts = {}
        for c in all_chars:
            char_counts[c] = char_counts.get(c, 0) + 1
        multi_event_chars = sum(1 for count in char_counts.values() if count > 1)
        if multi_event_chars > 0:
            synergy_score += 0.3
        num_checks += 1
        
        if num_checks > 0:
            synergy_score /= num_checks
        
        return min(synergy_score, 1.0)


# ===== 超图构建器 =====

class HypergraphBuilder:
    """
    超图构建器
    
    三阶段构建流程：
    1. 候选事件聚合（时间窗口 + 自适应窗口）
    2. 因果必要性验证（MHCV，成对协同验证）
    3. 超边构建 + 密度控制
    """
    
    def __init__(
        self,
        time_window: int = 600,
        mo_size: int = 3,
        om_size: int = 3,
        co_window_size: Optional[int] = None,
        co_max_size: int = 50,
        similarity_threshold: float = 0.5,
        strict_causality: bool = True,  # 默认严格
        min_events_for_causal: int = 4,
        video_id: Optional[str] = None,
        video_lengths: Optional[Dict[str, float]] = None,
        max_mo_ratio: float = 0.3,    # 新增：密度控制
        max_om_ratio: float = 0.3,     # 新增：密度控制
        enable_debug: bool = False,    # 新增：调试日志
    ):
        # 自适应时间窗口
        if video_id and video_lengths:
            video_length = video_lengths.get(video_id)
            self.time_window = get_adaptive_time_window(
                video_id, video_length, video_lengths, time_window
            )
            if self.time_window != time_window:
                print(f"  📐 Adaptive time window for {video_id}: {self.time_window}s "
                      f"(default: {time_window}s, length: {video_length}s)")
        else:
            self.time_window = time_window
        
        self.mo_size = mo_size
        self.om_size = om_size
        self.co_window_size = co_window_size
        self.co_max_size = co_max_size
        self.min_events_for_causal = min_events_for_causal
        self.max_mo_ratio = max_mo_ratio
        self.max_om_ratio = max_om_ratio
        self.enable_debug = enable_debug
        
        self.verifier = CausalityVerifier(
            use_embeddings=True,
            similarity_threshold=similarity_threshold,
            time_window=self.time_window,
            min_confidence_for_non_strict=0.65
        )
        self.strict_causality = strict_causality
    
    def build(self, events: List[Event]) -> Tuple[List[Hyperedge], List[Hyperedge], List[Hyperedge]]:
        """构建完整超图"""
        E_CO = self._build_co_hyperedges(events)
        E_MO, E_OM = self._build_causal_hyperedges(events)
        
        E_MO = self._deduplicate_edges(E_MO)
        E_OM = self._deduplicate_edges(E_OM)
        E_CO = self._deduplicate_edges(E_CO)
        
        return E_MO, E_OM, E_CO
    
    def _build_co_hyperedges(self, events: List[Event]) -> List[Hyperedge]:
        """构建角色共现超边（CO）"""
        char_events = defaultdict(list)
        all_chars_found = set()
        
        for e in events:
            for char in e.characters:
                all_chars_found.add(char)
                if is_valid_string(char):
                    char_events[char].append(e.id)
        
        # 调试日志
        if self.enable_debug:
            invalid_chars = all_chars_found - set(char_events.keys())
            print(f"  [CO Debug] Total chars found: {len(all_chars_found)}")
            print(f"  [CO Debug] Valid chars: {len(char_events)}")
            print(f"  [CO Debug] Invalid/filtered: {len(invalid_chars)}")
            if invalid_chars and len(invalid_chars) <= 10:
                print(f"  [CO Debug] Filtered chars: {invalid_chars}")
        
        E_CO = []
        for char, node_ids in char_events.items():
            if len(node_ids) < 2:
                continue
            
            node_ids_sorted = sorted(node_ids)
            
            if self.co_window_size is not None:
                window_size = self.co_window_size
            else:
                window_size = self.co_max_size
            
            if len(node_ids_sorted) > window_size:
                step = max(1, window_size // 2)
                for i in range(0, len(node_ids_sorted) - 1, step):
                    window = node_ids_sorted[i:i + window_size]
                    if len(window) >= 2:
                        E_CO.append(Hyperedge(
                            type='CO',
                            nodes=window,
                            character=char,
                            weight=1.0
                        ))
            else:
                E_CO.append(Hyperedge(
                    type='CO',
                    nodes=node_ids_sorted,
                    character=char,
                    weight=1.0
                ))
        
        if self.enable_debug:
            print(f"  [CO Debug] Built {len(E_CO)} CO hyperedges")
        
        return E_CO

    def _build_causal_hyperedges(
        self, events: List[Event]
    ) -> Tuple[List[Hyperedge], List[Hyperedge]]:
        """构建因果超边（MO/OM）"""
        E_MO, E_OM = [], []
        
        if len(events) < self.min_events_for_causal:
            return E_MO, E_OM
        
        E_MO = self._build_mo_edges(events, 'global', check_char_overlap=False)
        E_OM = self._build_om_edges(events, 'global', check_char_overlap=False)
        
        return E_MO, E_OM

    def _build_mo_edges(
        self, 
        events: List[Event], 
        character: str,
        check_char_overlap: bool = False
    ) -> List[Hyperedge]:
        """构建多因超边（MO），带密度控制"""
        E_MO = []
        n = len(events)
        candidate_edges = []
        
        for i in range(n):
            effect_event = events[i]
            candidate_causes = []
            
            # 向前扫描，收集时间窗口内的前置事件
            for j in range(i - 1, -1, -1):
                cause_event = events[j]
                
                if check_char_overlap:
                    cause_chars = {c for c in cause_event.characters if is_valid_string(c)}
                    effect_chars = {c for c in effect_event.characters if is_valid_string(c)}
                    if cause_chars and effect_chars and not cause_chars & effect_chars:
                        continue
                
                if not self._within_time_window([cause_event], effect_event):
                    break
                
                candidate_causes.append(cause_event)
            
            candidate_causes.reverse()
            
            if len(candidate_causes) < 2:
                continue
            
            # 阶段2：因果必要性验证
            independent_passed = []
            for cause in candidate_causes:
                is_necessary, confidence = self.verifier.verify_causality(
                    [cause], effect_event,
                    strict_mode=self.strict_causality
                )
                if is_necessary:
                    independent_passed.append((cause, confidence))
            
            # 子集协同验证
            verified_causes = []
            
            if len(independent_passed) >= 2:
                independent_passed.sort(key=lambda x: x[1], reverse=True)
                subset_causes = [ip[0] for ip in independent_passed]
                
                synergy_verified = []
                for ii in range(len(subset_causes)):
                    for jj in range(ii + 1, len(subset_causes)):
                        subset = [subset_causes[ii], subset_causes[jj]]
                        synergy_score = self.verifier._check_synergy(subset)
                        if synergy_score >= 0.3:
                            base_score = (independent_passed[ii][1] + independent_passed[jj][1]) / 2
                            enhanced_score = base_score * (1 + synergy_score * 0.5)
                            synergy_verified.append((subset, enhanced_score))
                
                if synergy_verified:
                    synergy_verified.sort(key=lambda x: x[1], reverse=True)
                    best_subset, best_score = synergy_verified[0]
                    for cause in best_subset:
                        verified_causes.append((cause, best_score / len(best_subset)))
                else:
                    # 无协同子集，回退到独立验证结果
                    verified_causes = independent_passed[:self.mo_size]
            else:
                verified_causes = independent_passed
            
            cause_events_final = [vc[0] for vc in verified_causes]
            cause_ids = [e.id for e in cause_events_final]
            
            if len(cause_ids) < 2:
                continue
            
            # 空列表保护
            avg_confidence = float(np.mean([vc[1] for vc in verified_causes])) if verified_causes else 0.0
            
            candidate_edges.append((cause_ids, effect_event.id, avg_confidence))
        
        # 密度控制
        max_mo_edges = max(1, int(n * self.max_mo_ratio))
        if len(candidate_edges) > max_mo_edges:
            candidate_edges.sort(key=lambda x: x[2], reverse=True)
            dropped = len(candidate_edges) - max_mo_edges
            candidate_edges = candidate_edges[:max_mo_edges]
            if self.enable_debug:
                print(f"  [MO Density Control] Kept {max_mo_edges}/{len(candidate_edges)+dropped} "
                      f"(ratio={max_mo_edges/n:.2f})")
        
        for cause_ids, effect_id, conf in candidate_edges:
            E_MO.append(Hyperedge(
                type='MO',
                nodes=cause_ids + [effect_id],
                source_nodes=cause_ids,
                target_nodes=[effect_id],
                weight=conf,  # 动态置信度权重
                character=character,
                causality_score=conf
            ))
        
        return E_MO

    def _build_om_edges(
        self, 
        events: List[Event], 
        character: str,
        check_char_overlap: bool = False
    ) -> List[Hyperedge]:
        """构建多果超边（OM），带密度控制"""
        E_OM = []
        n = len(events)
        candidate_edges = []
        
        for i in range(n):
            cause_event = events[i]
            candidate_effects = []
            
            for j in range(i + 1, n):
                effect_event = events[j]
                
                if check_char_overlap:
                    cause_chars = {c for c in cause_event.characters if is_valid_string(c)}
                    effect_chars = {c for c in effect_event.characters if is_valid_string(c)}
                    if cause_chars and effect_chars and not cause_chars & effect_chars:
                        continue
                
                if not self._within_time_window([cause_event], effect_event):
                    break
                
                candidate_effects.append(effect_event)
            
            if len(candidate_effects) < 2:
                continue
            
            verified_effects = []
            for effect in candidate_effects:
                is_consequence, confidence = self.verifier.verify_causality(
                    [cause_event], effect,
                    strict_mode=self.strict_causality
                )
                if is_consequence:
                    verified_effects.append((effect, confidence))
            
            if len(verified_effects) < 2:
                continue
            
            verified_effects.sort(key=lambda x: x[1], reverse=True)
            if len(verified_effects) > self.om_size:
                verified_effects = verified_effects[:self.om_size]
            
            effect_events_final = [ve[0] for ve in verified_effects]
            effect_ids = [e.id for e in effect_events_final]
            
            if len(effect_ids) < 2:
                continue
            
            # 空列表保护
            avg_confidence = float(np.mean([ve[1] for ve in verified_effects])) if verified_effects else 0.0
            
            candidate_edges.append((cause_event.id, effect_ids, avg_confidence))
        
        # 密度控制
        max_om_edges = max(1, int(n * self.max_om_ratio))
        if len(candidate_edges) > max_om_edges:
            candidate_edges.sort(key=lambda x: x[2], reverse=True)
            dropped = len(candidate_edges) - max_om_edges
            candidate_edges = candidate_edges[:max_om_edges]
            if self.enable_debug:
                print(f"  [OM Density Control] Kept {max_om_edges}/{len(candidate_edges)+dropped} "
                      f"(ratio={max_om_edges/n:.2f})")
        
        for cause_id, effect_ids, conf in candidate_edges:
            E_OM.append(Hyperedge(
                type='OM',
                nodes=[cause_id] + effect_ids,
                source_nodes=[cause_id],
                target_nodes=effect_ids,
                weight=conf,  # 动态置信度权重
                character=character,
                causality_score=conf
            ))
        
        return E_OM

    def _within_time_window(
        self, 
        source_events: List[Event], 
        target_event: Event
    ) -> bool:
        """检查源事件是否在目标事件的时间窗口内"""
        target_time = target_event.time_seconds
        
        if target_time is None:
            return all(e.id < target_event.id for e in source_events)
        
        for src in source_events:
            src_time = src.time_seconds
            if src_time is None:
                continue
            if src_time > target_time:
                return False
            if target_time - src_time > self.time_window:
                return False
        
        if all(e.time_seconds is None for e in source_events):
            return all(e.id < target_event.id for e in source_events)
        
        return True
        
    def _deduplicate_edges(self, edges: List[Hyperedge]) -> List[Hyperedge]:
        """超边去重（保留方向信息）"""
        seen = set()
        unique = []
        for edge in edges:
            if edge.type in ('MO', 'OM'):
                key = (edge.type, tuple(sorted(edge.source_nodes)), tuple(sorted(edge.target_nodes)))
            else:
                key = (edge.type, tuple(sorted(edge.nodes)))
            
            if key not in seen:
                seen.add(key)
                unique.append(edge)
        return unique


# ===== 序列化 =====

def hyperedge_to_dict(edge: Hyperedge) -> dict:
    """将 Hyperedge 对象转换为可序列化的字典"""
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
    time_window: int = 600,
    mo_size: int = 3,
    om_size: int = 3,
    co_window_size: Optional[int] = None,
    co_max_size: int = 50,
    similarity_threshold: float = 0.5,
    strict_causality: bool = True,  # 默认严格
    min_events_for_causal: int = 4,
    video_id: Optional[str] = None,
    video_lengths: Optional[Dict[str, float]] = None,
    use_adaptive: bool = True,
    max_mo_ratio: float = 0.3,     # 新增
    max_om_ratio: float = 0.3,     # 新增
    enable_debug: bool = False,     # 新增
) -> Tuple[List[dict], List[dict], List[dict], dict]:
    """
    构建超图的主函数（外部调用接口）
    """
    event_objects = []
    for e in events:
        event_objects.append(Event(
            id=e.get('id', ''),
            time_seconds=e.get('time_seconds'),
            start_time=e.get('start_time', ''),
            end_time=e.get('end_time', ''),
            characters=e.get('P_i', []),
            action=e.get('A_i', ''),
            location=e.get('L_i', ''),
            summary=e.get('S_i', ''),
            text=e.get('text', ''),
            embedding=e.get('embedding')
        ))
    
    builder = HypergraphBuilder(
        time_window=time_window,
        mo_size=mo_size,
        om_size=om_size,
        co_window_size=co_window_size,
        co_max_size=co_max_size,
        similarity_threshold=similarity_threshold,
        strict_causality=strict_causality,
        min_events_for_causal=min_events_for_causal,
        video_id=video_id if use_adaptive else None,
        video_lengths=video_lengths if use_adaptive else None,
        max_mo_ratio=max_mo_ratio,
        max_om_ratio=max_om_ratio,
        enable_debug=enable_debug,
    )
    
    E_MO, E_OM, E_CO = builder.build(event_objects)
    
    E_MO_dicts = [hyperedge_to_dict(e) for e in E_MO]
    E_OM_dicts = [hyperedge_to_dict(e) for e in E_OM]
    E_CO_dicts = [hyperedge_to_dict(e) for e in E_CO]
    
    mo_scores = [e.causality_score for e in E_MO]
    om_scores = [e.causality_score for e in E_OM]
    mo_sizes = [len(e.source_nodes) for e in E_MO]
    om_sizes = [len(e.target_nodes) for e in E_OM]
    
    stats = {
        'num_events': len(events),
        'num_mo_edges': len(E_MO_dicts),
        'num_om_edges': len(E_OM_dicts),
        'num_co_edges': len(E_CO_dicts),
        'total_hyperedges': len(E_MO_dicts) + len(E_OM_dicts) + len(E_CO_dicts),
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
    
    return E_MO_dicts, E_OM_dicts, E_CO_dicts, stats


# ===== 命令行接口 =====

def main():
    parser = argparse.ArgumentParser(
        description='CausalHyperGraph - 超图构建模块（论文 3.3 节实现）'
    )
    
    parser.add_argument('--input', required=True, help='Input events JSON file')
    parser.add_argument('--output', required=True, help='Output hypergraph JSON file')
    
    parser.add_argument('--time_window', type=int, default=600)
    parser.add_argument('--mo_size', type=int, default=3)
    parser.add_argument('--om_size', type=int, default=3)
    parser.add_argument('--co_window_size', type=int, default=None)
    parser.add_argument('--co_max_size', type=int, default=50)
    
    parser.add_argument('--similarity_threshold', type=float, default=0.5)
    parser.add_argument('--strict', action='store_true', default=True,  # 默认严格
                        help='Enable strict causality verification (default: True)')
    parser.add_argument('--no_strict', action='store_true',            # 反向开关
                        help='Disable strict causality verification')
    parser.add_argument('--min_events', type=int, default=4)
    
    # 新增密度控制参数
    parser.add_argument('--max_mo_ratio', type=float, default=0.3,
                        help='Max MO edges as ratio of events (default: 0.3)')
    parser.add_argument('--max_om_ratio', type=float, default=0.3,
                        help='Max OM edges as ratio of events (default: 0.3)')
    parser.add_argument('--enable_debug', action='store_true',
                        help='Enable debug logging')
    
    parser.add_argument('--video_length', type=str, default=None)
    parser.add_argument('--video_id', type=str, default=None)
    parser.add_argument('--no_adaptive_window', action='store_true')
    
    args = parser.parse_args()
    
    # 处理 --no_strict 反向开关
    strict_causality = not args.no_strict if args.no_strict else args.strict
    
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    events = data.get('events', data.get('nodes', []))
    if not events:
        print("❌ No events found in input file")
        return
    
    video_lengths = None
    if not args.no_adaptive_window:
        if args.video_length and os.path.exists(args.video_length):
            video_lengths = load_video_lengths(args.video_length)
    
    video_id = args.video_id
    if not video_id and not args.no_adaptive_window:
        video_id = Path(args.input).stem
    
    print(f"\n{'='*60}")
    print(f"Hypergraph Construction Configuration")
    print(f"{'='*60}")
    print(f"  Events: {len(events)}")
    print(f"  Time window: {args.time_window}s")
    print(f"  Adaptive window: {'disabled' if args.no_adaptive_window else 'enabled'}")
    print(f"  Strict causality: {strict_causality}")
    print(f"  MO/OM max size: {args.mo_size}/{args.om_size}")
    print(f"  MO/OM max ratio: {args.max_mo_ratio}/{args.max_om_ratio}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    E_MO, E_OM, E_CO, stats = build_hypergraph(
        events=events,
        time_window=args.time_window,
        mo_size=args.mo_size,
        om_size=args.om_size,
        co_window_size=args.co_window_size,
        co_max_size=args.co_max_size,
        similarity_threshold=args.similarity_threshold,
        strict_causality=strict_causality,
        min_events_for_causal=args.min_events,
        video_id=video_id if not args.no_adaptive_window else None,
        video_lengths=video_lengths if not args.no_adaptive_window else None,
        max_mo_ratio=args.max_mo_ratio,
        max_om_ratio=args.max_om_ratio,
        enable_debug=args.enable_debug,
    )
    
    elapsed = time.time() - start_time
    
    hypergraph = {
        'nodes': events,
        'hyperedges': E_MO + E_OM + E_CO,
        'stats': stats
    }
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(hypergraph, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Hypergraph Construction Results")
    print(f"{'='*60}")
    print(f"  Build time: {elapsed:.1f}s")
    print(f"  MO edges: {stats['num_mo_edges']} (ratio={stats['num_mo_edges']/len(events):.2f})")
    print(f"    - Avg weight: {stats['mo_avg_weight']:.3f}")
    print(f"    - Avg size: {stats['mo_avg_size']:.1f}")
    print(f"  OM edges: {stats['num_om_edges']} (ratio={stats['num_om_edges']/len(events):.2f})")
    print(f"    - Avg weight: {stats['om_avg_weight']:.3f}")
    print(f"    - Avg size: {stats['om_avg_size']:.1f}")
    print(f"  CO edges: {stats['num_co_edges']}")
    print(f"  Total: {stats['total_hyperedges']}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()