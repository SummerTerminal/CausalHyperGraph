"""
question_type_infer.py - question type inference for auxiliary loss L_aux (paper 3.6.2)
"""

import re
from typing import Dict, List, Optional, Tuple
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuestionTypeInferer:
    """
    Infer expected hyperedge type for a question: MO / OM / CO.
    Uses keyword matching + heuristics (not learned).
    """

    # cause/effect keywords
    CAUSAL_KW = {
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

    # 角色 / 关系 keywords
    CHAR_KW = {
        'explicit': [
            'who', 'character', 'person', 'he', 'she', 'they',
            'him', 'her', 'them', 'his', 'her', 'their',
            'role', 'relationship', 'friendship', 'alliance',
            'betray', 'trust', 'loyalty', 'enemy', 'friend',
            'partner', 'companion', 'ally', 'rival'
        ],
        'names': []   # filled dynamically from video characters
    }

    ACTION_KW = [
        'what', 'how', 'when', 'where', 'did', 'does', 'do',
        'happen', 'occur', 'take place', 'perform', 'act',
        'say', 'tell', 'speak', 'talk', 'mention'
    ]

    def __init__(self, character_names: Optional[List[str]] = None):
        if character_names:
            self.char_set = set(n.lower() for n in character_names)
        else:
            self.char_set = set()

        # precompile patterns
        self.causal_pat = {
            'strong': [re.compile(rf'\b{kw}\b', re.I) for kw in self.CAUSAL_KW['strong']],
            'weak':   [re.compile(rf'\b{kw}\b', re.I) for kw in self.CAUSAL_KW['weak']]
        }
        self.char_pat = [re.compile(rf'\b{kw}\b', re.I) for kw in self.CHAR_KW['explicit']]
        # 加上角色名字
        for nm in self.char_set:
            if len(nm) > 2:
                self.char_pat.append(re.compile(rf'\b{nm}\b', re.I))

        self.act_pat = [re.compile(rf'\b{kw}\b', re.I) for kw in self.ACTION_KW]

    def infer_type(self, question: str) -> Tuple[str, Dict[str, float]]:
        q = question.lower()
        scores = {'MO': 0.0, 'OM': 0.0, 'CO': 0.0, 'action': 0.0}

        # count strong / weak causal hits
        cs = sum(1 for p in self.causal_pat['strong'] if p.search(q))
        cw = sum(1 for p in self.causal_pat['weak'] if p.search(q))
        ch = sum(1 for p in self.char_pat if p.search(q))
        ac = sum(1 for p in self.act_pat if p.search(q))

        # 多因暗示
        scores['MO'] += cs * 0.4 + cw * 0.15
        if any(re.search(pat, q) for pat in [
            r'what caused', r'why did', r'what led to', r'what made',
            r'what resulted in', r'due to what', r'owing to what']):
            scores['MO'] += 0.3

        # 多果暗示
        scores['OM'] += cs * 0.3 + cw * 0.2
        if any(re.search(pat, q) for pat in [
            r'what happened after', r'what resulted from', r'consequences of',
            r'effects of', r'impact of', r'what followed', r'what came after']):
            scores['OM'] += 0.3

        # 角色相关
        scores['CO'] += ch * 0.3
        if any(re.search(pat, q) for pat in [
            r'relationship between', r'how.*relate', r'who.*with',
            r'friendship', r'alliance', r'betrayal', r'conflict between']):
            scores['CO'] += 0.3

        scores['action'] += ac * 0.1

        # normalize
        total = sum(scores.values()) or 1.0
        norm = {k: v / total for k, v in scores.items()}

        # 判定类型
        maxv = max(norm.values())
        if maxv < 0.25:
            return 'unknown', norm

        if norm['CO'] > norm['MO'] and norm['CO'] > norm['OM']:
            return 'CO', norm
        elif norm['MO'] > norm['OM']:
            return 'MO', norm
        else:
            return 'OM', norm

    def batch_infer(self, questions: List[str]) -> List[Tuple[str, Dict[str, float]]]:
        return [self.infer_type(q) for q in questions]

    def get_label(self, q: str) -> int:
        """0:MO, 1:OM, 2:CO, 3:unknown"""
        t, _ = self.infer_type(q)
        mapping = {'MO': 0, 'OM': 1, 'CO': 2}
        return mapping.get(t, 3)


class TypeAwareAuxiliaryLoss(nn.Module):
    """
    Auxiliary loss to predict question type from context + question.
    L_aux = -Σ_{(q, τ)} log p(τ | q)
    """

    def __init__(self, hidden_dim=512, num_types=3):
        super().__init__()
        self.num_types = num_types
        self.type_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_types)
        )
        self.type_names = ['MO', 'OM', 'CO']   # for debug

    def forward(self, context, q_emb, target_type=None):
        # make sure dims are correct
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)
        if context.dim() == 1:
            context = context.unsqueeze(0)

        comb = torch.cat([context, q_emb], dim=-1)
        logits = self.type_classifier(comb)    # [B, num_types]

        if target_type is not None:
            tgt = torch.tensor([target_type], device=logits.device)
            loss = F.cross_entropy(logits, tgt)
            return loss, logits
        return torch.tensor(0.0, device=logits.device), logits

    def predict(self, context, q_emb):
        _, logits = self.forward(context, q_emb, target_type=None)
        return logits.argmax(dim=-1).item()


def create_type_inferer(character_names=None):
    return QuestionTypeInferer(character_names)