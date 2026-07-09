"""
question_type_infer.py - 问题类型推断
用于论文 3.6.2 节的辅助损失 L_aux = -Σ_{(q, τ)} log p(τ | q)
"""

import re
from typing import Dict, List, Optional, Tuple
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuestionTypeInferer:
    """
    问题类型推断器
    判断问题期望的超边类型：MO（多因）、OM（多果）、CO（角色共现）
    """
    
    # 因果推理关键词（对应 MO/OM）
    CAUSAL_KEYWORDS = {
        'strong': [
            'because', 'therefore', 'thus', 'hence', 'consequently',
            'as a result', 'due to', 'owing to', 'so that', 'in order to',
            'lead to', 'result in', 'cause', 'trigger', 'bring about',
            'why', 'reason', 'explain', 'what caused', 'what led to',
            'contribute', 'influence', 'affect', 'impact', 'determine'
        ],
        'weak': [
            'then', 'after', 'before', 'when', 'while', 'during',
            'since', 'as', 'because of', 'thanks to', 'owing to',
            'consequence', 'outcome', 'effect', 'reaction', 'response'
        ]
    }
    
    # 角色相关关键词（对应 CO）
    CHARACTER_KEYWORDS = {
        'explicit': [
            'who', 'character', 'person', 'he', 'she', 'they',
            'him', 'her', 'them', 'his', 'her', 'their',
            'role', 'relationship', 'friendship', 'alliance',
            'betray', 'trust', 'loyalty', 'enemy', 'friend',
            'partner', 'companion', 'ally', 'rival'
        ],
        'names': []  # 动态填充
    }
    
    # 动作关键词（辅助判断）
    ACTION_KEYWORDS = [
        'what', 'how', 'when', 'where', 'did', 'does', 'do',
        'happen', 'occur', 'take place', 'perform', 'act',
        'say', 'tell', 'speak', 'talk', 'mention'
    ]
    
    def __init__(self, character_names: Optional[List[str]] = None):
        """
        Args:
            character_names: 视频中的角色名称列表
        """
        self.character_names = set([n.lower() for n in character_names]) if character_names else set()
        
        # 构建完整关键词集
        self.causal_patterns = {
            'strong': [re.compile(rf'\b{kw}\b', re.IGNORECASE) for kw in self.CAUSAL_KEYWORDS['strong']],
            'weak': [re.compile(rf'\b{kw}\b', re.IGNORECASE) for kw in self.CAUSAL_KEYWORDS['weak']]
        }
        
        self.character_patterns = [
            re.compile(rf'\b{kw}\b', re.IGNORECASE) for kw in self.CHARACTER_KEYWORDS['explicit']
        ]
        
        # 添加角色名模式
        for name in self.character_names:
            if len(name) > 2:
                self.character_patterns.append(re.compile(rf'\b{name}\b', re.IGNORECASE))
        
        self.action_patterns = [
            re.compile(rf'\b{kw}\b', re.IGNORECASE) for kw in self.ACTION_KEYWORDS
        ]
    
    def infer_type(self, question: str) -> Tuple[str, Dict[str, float]]:
        """
        推断问题类型
        
        Returns:
            type: 'MO' | 'OM' | 'CO' | 'mixed' | 'unknown'
            scores: 各类别的得分
        """
        question_lower = question.lower()
        
        # 计算各类别得分
        scores = {
            'MO': 0.0,
            'OM': 0.0,
            'CO': 0.0,
            'action': 0.0
        }
        
        # 1. 因果关键词匹配
        causal_strong_count = 0
        causal_weak_count = 0
        
        for pattern in self.causal_patterns['strong']:
            if pattern.search(question_lower):
                causal_strong_count += 1
        
        for pattern in self.causal_patterns['weak']:
            if pattern.search(question_lower):
                causal_weak_count += 1
        
        # 2. 角色关键词匹配
        char_count = 0
        for pattern in self.character_patterns:
            if pattern.search(question_lower):
                char_count += 1
        
        # 3. 动作关键词匹配
        action_count = 0
        for pattern in self.action_patterns:
            if pattern.search(question_lower):
                action_count += 1
        
        # 计算得分
        # MO: 强因果关键词 + 多因暗示 (如 "what caused", "why did")
        scores['MO'] += causal_strong_count * 0.4
        scores['MO'] += causal_weak_count * 0.15
        
        # 检查多因暗示
        multi_cause_patterns = [
            r'what caused', r'why did', r'what led to', r'what made',
            r'what resulted in', r'due to what', r'owing to what'
        ]
        for pattern in multi_cause_patterns:
            if re.search(pattern, question_lower):
                scores['MO'] += 0.3
                break
        
        # OM: 强因果关键词 + 多果暗示 (如 "what happened after", "what resulted from")
        scores['OM'] += causal_strong_count * 0.3
        scores['OM'] += causal_weak_count * 0.2
        
        multi_effect_patterns = [
            r'what happened after', r'what resulted from', r'consequences of',
            r'effects of', r'impact of', r'what followed', r'what came after'
        ]
        for pattern in multi_effect_patterns:
            if re.search(pattern, question_lower):
                scores['OM'] += 0.3
                break
        
        # CO: 角色关键词 + 关系暗示
        scores['CO'] += char_count * 0.3
        
        relationship_patterns = [
            r'relationship between', r'how.*relate', r'who.*with',
            r'friendship', r'alliance', r'betrayal', r'conflict between'
        ]
        for pattern in relationship_patterns:
            if re.search(pattern, question_lower):
                scores['CO'] += 0.3
                break
        
        # 动作类型（辅助判断）
        scores['action'] += action_count * 0.1
        
        # 归一化
        total = sum(scores.values()) or 1.0
        normalized = {k: v / total for k, v in scores.items()}
        
        # 确定主要类型
        max_score = max(normalized.values())
        
        if max_score < 0.25:
            return 'unknown', normalized
        
        # 取最高分类型
        if normalized['CO'] > normalized['MO'] and normalized['CO'] > normalized['OM']:
            return 'CO', normalized
        elif normalized['MO'] > normalized['OM']:
            return 'MO', normalized
        else:
            return 'OM', normalized
    
    def infer_batch(self, questions: List[str]) -> List[Tuple[str, Dict[str, float]]]:
        """批量推断"""
        return [self.infer_type(q) for q in questions]
    
    def get_type_label(self, question: str) -> int:
        """
        返回类型标签用于训练
        0: MO, 1: OM, 2: CO, 3: unknown
        """
        q_type, _ = self.infer_type(question)
        mapping = {'MO': 0, 'OM': 1, 'CO': 2}
        return mapping.get(q_type, 3)


class TypeAwareAuxiliaryLoss(nn.Module):
    """
    类型感知辅助损失
    论文公式: L_aux = -Σ_{(q, τ)} log p(τ | q)
    """
    
    def __init__(self, hidden_dim: int = 512, num_types: int = 3):
        super().__init__()
        self.num_types = num_types
        
        # 问题类型分类器
        self.type_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),  # context + q_emb
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_types)
        )
        
        self.type_names = ['MO', 'OM', 'CO']
    
    def forward(
        self,
        context: torch.Tensor,      # [hidden_dim]
        q_emb: torch.Tensor,        # [hidden_dim]
        target_type: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            context: 超图读出上下文
            q_emb: 问题编码
            target_type: 目标类型标签 (0: MO, 1: OM, 2: CO)
            
        Returns:
            loss: 交叉熵损失（如果 target_type 不为 None）
            logits: 类型预测 logits
        """
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)
        if context.dim() == 1:
            context = context.unsqueeze(0)
        
        combined = torch.cat([context, q_emb], dim=-1)  # [1, 2*hidden_dim]
        logits = self.type_classifier(combined)  # [1, num_types]
        
        if target_type is not None:
            target_tensor = torch.tensor([target_type], device=logits.device)
            loss = F.cross_entropy(logits, target_tensor)
            return loss, logits
        
        return torch.tensor(0.0, device=logits.device), logits
    
    def predict_type(self, context: torch.Tensor, q_emb: torch.Tensor) -> int:
        """预测问题类型"""
        _, logits = self.forward(context, q_emb, target_type=None)
        return logits.argmax(dim=-1).item()


# ===== 工厂函数 =====

def create_type_inferer(character_names: Optional[List[str]] = None) -> QuestionTypeInferer:
    """创建问题类型推断器"""
    return QuestionTypeInferer(character_names)