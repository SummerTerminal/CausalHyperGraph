"""
build_special_testset.py - 构建多元因果推理专项测试集
用于论文 4.1.1 节的 MTO-Test 和 OMT-Test
"""

import os
import json
import re
import argparse
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional
import random


class SpecialTestSetBuilder:
    """
    专项测试集构建器
    自动筛选 MTO-Test (多因一果) 和 OMT-Test (一因多果)
    """
    
    def __init__(self, hypergraph_dir: str):
        self.hypergraph_dir = hypergraph_dir
        self.hypergraphs = {}
        self._load_hypergraphs()
    
    def _load_hypergraphs(self):
        """加载所有超图"""
        for root, _, files in os.walk(self.hypergraph_dir):
            for f in files:
                if f.endswith('.json') and not f.startswith('_'):
                    vid = os.path.splitext(f)[0]
                    try:
                        with open(os.path.join(root, f), 'r', encoding='utf-8') as file:
                            self.hypergraphs[vid] = json.load(file)
                    except Exception as e:
                        print(f"Warning: Failed to load {f}: {e}")
        print(f"Loaded {len(self.hypergraphs)} hypergraphs")
    
    def _has_mo_edges(self, hg: Dict) -> bool:
        """检查是否有MO超边"""
        edges = hg.get('hyperedges', {})
        if isinstance(edges, dict):
            return len(edges.get('MO', [])) > 0
        for e in edges:
            if e.get('type') == 'MO':
                return True
        return False
    
    def _has_om_edges(self, hg: Dict) -> bool:
        """检查是否有OM超边"""
        edges = hg.get('hyperedges', {})
        if isinstance(edges, dict):
            return len(edges.get('OM', [])) > 0
        for e in edges:
            if e.get('type') == 'OM':
                return True
        return False
    
    def _get_mo_edge_count(self, hg: Dict) -> int:
        """获取MO超边数量"""
        edges = hg.get('hyperedges', {})
        if isinstance(edges, dict):
            return len(edges.get('MO', []))
        return sum(1 for e in edges if e.get('type') == 'MO')
    
    def _get_om_edge_count(self, hg: Dict) -> int:
        """获取OM超边数量"""
        edges = hg.get('hyperedges', {})
        if isinstance(edges, dict):
            return len(edges.get('OM', []))
        return sum(1 for e in edges if e.get('type') == 'OM')
    
    def _get_edge_size(self, edge: Dict) -> int:
        """获取超边大小"""
        return len(edge.get('nodes', []))
    
    def _get_large_edges(self, hg: Dict, edge_type: str, min_size: int = 3) -> List[Dict]:
        """获取大于指定大小的超边"""
        edges = hg.get('hyperedges', {})
        result = []
        
        if isinstance(edges, dict):
            for e in edges.get(edge_type, []):
                if self._get_edge_size(e) >= min_size:
                    result.append(e)
        else:
            for e in edges:
                if e.get('type') == edge_type and self._get_edge_size(e) >= min_size:
                    result.append(e)
        return result
    
    def _question_needs_mto(self, question: str) -> bool:
        """判断问题是否需要多因一果推理"""
        patterns = [
            r'why did', r'why does', r'what caused', r'what led to',
            r'what made', r'what resulted in', r'what contributed to',
            r'what were the reasons', r'what factors', r'due to what',
            r'why was', r'why were', r'what brought about',
            r'what combination', r'what together', r'what jointly'
        ]
        return any(re.search(p, question.lower()) for p in patterns)
    
    def _question_needs_omt(self, question: str) -> bool:
        """判断问题是否需要一因多果推理"""
        patterns = [
            r'what happened after', r'what resulted from', r'consequences of',
            r'effects of', r'impact of', r'what followed', r'what came after',
            r'what did.*cause', r'what chain of events', r'what repercussions',
            r'what was the aftermath', r'what were the results',
            r'what changes occurred', r'what subsequent events'
        ]
        return any(re.search(p, question.lower()) for p in patterns)
    
    def filter_questions_by_hypergraph(
        self,
        questions: List[Dict],
        test_type: str = 'mto'
    ) -> List[Dict]:
        """
        根据超图结构筛选问题
        
        Args:
            questions: 问题列表
            test_type: 'mto' 或 'omt'
        """
        filtered = []
        
        for q in questions:
            vid = q.get('vid', q.get('video_id', ''))
            if vid not in self.hypergraphs:
                continue
            
            hg = self.hypergraphs[vid]
            
            if test_type == 'mto':
                # 需要：有MO超边 + 问题含有多因关键词
                if not self._has_mo_edges(hg):
                    continue
                if not self._question_needs_mto(q.get('question', '')):
                    continue
                
                # 优先选择MO超边较大的问题
                mo_count = self._get_mo_edge_count(hg)
                if mo_count < 2:
                    continue
                
                # 检查是否有大尺寸MO超边
                large_mo = self._get_large_edges(hg, 'MO', min_size=3)
                if not large_mo:
                    continue
                
                filtered.append({
                    **q,
                    '_mo_edge_count': mo_count,
                    '_max_mo_size': max(self._get_edge_size(e) for e in large_mo),
                    '_test_type': 'mto'
                })
                
            elif test_type == 'omt':
                # 需要：有OM超边 + 问题含有多果关键词
                if not self._has_om_edges(hg):
                    continue
                if not self._question_needs_omt(q.get('question', '')):
                    continue
                
                om_count = self._get_om_edge_count(hg)
                if om_count < 2:
                    continue
                
                large_om = self._get_large_edges(hg, 'OM', min_size=3)
                if not large_om:
                    continue
                
                filtered.append({
                    **q,
                    '_om_edge_count': om_count,
                    '_max_om_size': max(self._get_edge_size(e) for e in large_om),
                    '_test_type': 'omt'
                })
        
        return filtered
    
    def build_testset(
        self,
        questions: List[Dict],
        max_samples: int = 2400,
        seed: int = 42
    ) -> Dict[str, List[Dict]]:
        """
        构建专项测试集
        
        Returns:
            {
                'mto': [...],
                'omt': [...]
            }
        """
        random.seed(seed)
        
        mto_candidates = self.filter_questions_by_hypergraph(questions, 'mto')
        omt_candidates = self.filter_questions_by_hypergraph(questions, 'omt')
        
        print(f"MTO candidates: {len(mto_candidates)}")
        print(f"OMT candidates: {len(omt_candidates)}")
        
        # 按MO/OM大小排序，优先选择更复杂的
        mto_candidates.sort(key=lambda x: x.get('_max_mo_size', 0), reverse=True)
        omt_candidates.sort(key=lambda x: x.get('_max_om_size', 0), reverse=True)
        
        # 抽样
        mto_samples = mto_candidates[:min(max_samples // 2, len(mto_candidates))]
        omt_samples = omt_candidates[:min(max_samples // 2, len(omt_candidates))]
        
        # 打乱
        random.shuffle(mto_samples)
        random.shuffle(omt_samples)
        
        return {
            'mto': mto_samples,
            'omt': omt_samples
        }


def build_from_file(
    questions_path: str,
    hypergraph_dir: str,
    output_dir: str,
    max_samples: int = 2400,
    seed: int = 42
):
    """从文件构建专项测试集"""
    
    # 加载问题
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions")
    
    builder = SpecialTestSetBuilder(hypergraph_dir)
    testset = builder.build_testset(questions, max_samples, seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存
    mto_path = os.path.join(output_dir, 'mto_testset.json')
    omt_path = os.path.join(output_dir, 'omt_testset.json')
    
    with open(mto_path, 'w', encoding='utf-8') as f:
        json.dump(testset['mto'], f, ensure_ascii=False, indent=2)
    
    with open(omt_path, 'w', encoding='utf-8') as f:
        json.dump(testset['omt'], f, ensure_ascii=False, indent=2)
    
    # 生成统计报告
    stats = {
        'total_mto': len(testset['mto']),
        'total_omt': len(testset['omt']),
        'avg_mo_size': sum(q.get('_max_mo_size', 0) for q in testset['mto']) / max(len(testset['mto']), 1),
        'avg_om_size': sum(q.get('_max_om_size', 0) for q in testset['omt']) / max(len(testset['omt']), 1)
    }
    
    with open(os.path.join(output_dir, 'testset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"专项测试集构建完成")
    print(f"{'='*60}")
    print(f"  MTO-Test: {stats['total_mto']} samples, avg MO size: {stats['avg_mo_size']:.1f}")
    print(f"  OMT-Test: {stats['total_omt']} samples, avg OM size: {stats['avg_om_size']:.1f}")
    print(f"  保存至: {output_dir}")
    print(f"{'='*60}")
    
    return testset


def main():
    parser = argparse.ArgumentParser(description='构建多元因果推理专项测试集')
    
    parser.add_argument('--questions', required=True, help='问题JSON文件')
    parser.add_argument('--hypergraph_dir', required=True, help='超图目录')
    parser.add_argument('--output_dir', default='experiments/special_testsets')
    parser.add_argument('--max_samples', type=int, default=2400)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    build_from_file(
        args.questions,
        args.hypergraph_dir,
        args.output_dir,
        args.max_samples,
        args.seed
    )


if __name__ == '__main__':
    main()