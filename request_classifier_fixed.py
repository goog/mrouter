"""请求意图分类器 — 修复版"""
import asyncio
import time
import random
import string
import tracemalloc
import re
import math
from typing import List, Dict
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum

from langdetect import detect
import tiktoken


class TaskType(str, Enum):
    CODING = "coding"
    REASONING = "reasoning"
    CREATIVE = "creative"
    QUICK_QA = "quick_qa"
    LONG_CONTEXT = "long_context"
    CHINESE = "chinese"
    GENERAL = "general"


@dataclass
class ClassificationResult:
    task_type: TaskType
    confidence: float
    detected_lang: str
    estimated_tokens: int
    is_long_context: bool
    metadata: dict


class RequestClassifier:
    """多维度请求分类器 - 修复版"""
    
    TASK_KEYWORDS = {
        TaskType.CODING: {
            'high': ['代码', 'code', 'function', 'def ', 'class ', 'debug', 
                    'implement', 'api', 'script', '程序开发', '编写', '实现', 
                    '开发', '编程', '写代码', '编码', '程序'],
            'medium': ['python', 'javascript', 'typescript', 'rust', 'golang',
                      'sql', 'bash', 'refactor', '优化代码', '算法', '函数',
                      '模块', '接口', '配置', 'dockerfile', '正则', 'docker',
                      '测试', '单元测试', '部署', '服务器', '数据库'],
            'low': ['变量', '函数', '循环', 'variable', 'loop', 'syntax',
                   '参数', '调用', '返回', '对象', '实例', '缓存', '线程']
        },
        TaskType.REASONING: {
            'high': ['分析', '推理', '证明', 'analyze', 'reasoning', 'prove',
                    'step by step', '逻辑推导', '数学证明', '推导', '论证',
                    '演算', '演绎'],
            'medium': ['计算', 'calculate', 'math', '思考', 'think', '推断',
                      '论证', '逻辑', '数学', '概率', '统计', '数据', '复杂度',
                      '比较', '评估', '诊断', '根本原因', '因果关系'],
            'low': ['比较', 'compare', '评估', 'evaluate', '因素', '规律',
                   '趋势', '结论', '线索']
        },
        TaskType.CREATIVE: {
            'high': ['写作', '创作', '故事', '小说', 'creative', '写作任务',
                    'brainstorm', '创作内容', '写一篇', '创作一个', '撰写',
                    '编故事', '构思'],
            'medium': ['诗', 'poem', 'fiction', 'story', 'narrative', '诗歌',
                      '剧本', '散文', '歌词', '文案', '演讲稿', '文章', '小说',
                      '童话', '寓言', '角色', '世界构建', '开头', '结局'],
            'low': ['想象', 'imagine', '创意', '构思', '有趣', '好玩',
                   '吸引人', '激励', '讽刺']
        },
        TaskType.QUICK_QA: {
            'high': ['什么是', '解释', '简介', 'explain', 'what is', 'define',
                    '概述', '简介', '介绍一下', '解释一下', '说明', '讲一下',
                    '告诉我', '什么是', '定义'],
            'medium': ['summarize', 'briefly', 'overview', 'introduction',
                      '总结', '概括', '概述', '简单', '简明', '入门', '基础概念',
                      '了解', '学习', '区别'],
            'low': ['介绍', 'describe', '说明', 'tell me about', '理解',
                   '澄清', '意思', '含义']
        }
    }
    
    WEIGHT_CONFIG = {
        'keyword_high': 2.0,       # 进一步提高高频词权重
        'keyword_medium': 1.2,     # 提升中频词权重
        'keyword_low': 0.6,        # 提升低频词权重
        'position_early': 1.5,     # 保持位置权重
        'position_middle': 1.0,
        'position_late': 0.8,
        'diversity_bonus': 0.3,    # 提高多样性加成
        'length_penalty': 0.0003,  # 进一步降低惩罚
        'lang_match_bonus': 0.1    # 语言加成
    }
    
    LONG_CONTEXT_THRESHOLD = 16000
    
    def __init__(self):
        self._encoder = tiktoken.get_encoding("cl100k_base")
        self._compile_patterns()
        # 中文字符正则
        self._chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    
    def _compile_patterns(self):
        self._patterns = {}
        for task_type, weight_groups in self.TASK_KEYWORDS.items():
            self._patterns[task_type] = {}
            for weight_level, keywords in weight_groups.items():
                pattern = '|'.join(re.escape(kw) for kw in keywords)
                self._patterns[task_type][weight_level] = re.compile(
                    pattern, re.IGNORECASE
                )
    
    def _count_tokens(self, messages: list) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(self._encoder.encode(content))
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        total += len(self._encoder.encode(block["text"]))
                    elif block.get("type") == "image_url":
                        total += 1024
        return total
    
    def _detect_language(self, text: str) -> str:
        """改进的语言检测 - 双重策略"""
        if not text or len(text.strip()) == 0:
            return "en"
        
        # 策略1: 中文字符比例检测（快速可靠）
        chinese_chars = self._chinese_pattern.findall(text)
        chinese_char_count = sum(len(match) for match in chinese_chars)
        total_chars = len(text)
        
        # 如果中文字符占比超过 30%，判定为中文
        if total_chars > 0 and chinese_char_count / total_chars > 0.3:
            return "zh"
        
        # 策略2: langdetect 检测（适合其他语言）
        try:
            # 只取前500字符提速
            sample = text[:500] if len(text) > 500 else text
            detected = detect(sample)
            # 统一中文编码
            if detected in ['zh-cn', 'zh-tw', 'zh']:
                return "zh"
            return detected
        except Exception:
            return "en"
    
    def _extract_text(self, messages: list) -> str:
        texts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                texts.append(content)
        return " ".join(texts)
    
    def _get_position_weight(self, position: int, text_length: int) -> float:
        if text_length == 0:
            return 1.0
        relative_pos = position / text_length
        if relative_pos <= 0.2:
            return self.WEIGHT_CONFIG['position_early']
        elif relative_pos <= 0.6:
            return self.WEIGHT_CONFIG['position_middle']
        else:
            return self.WEIGHT_CONFIG['position_late']
    
    def _calculate_diversity(self, matches: set, total_keywords: int) -> float:
        if total_keywords == 0:
            return 0.0
        return len(matches) / total_keywords
    
    def _rule_based_classify(self, text: str):
        text_lower = text.lower()
        text_length = len(text)
        
        scores = {}
        match_details = {}
        
        for task_type in self.TASK_KEYWORDS.keys():
            score = 0.0
            matched_keywords = set()
            
            for weight_level, pattern in self._patterns[task_type].items():
                for match in pattern.finditer(text_lower):
                    keyword = match.group()
                    matched_keywords.add(keyword)
                    
                    base_weight = self.WEIGHT_CONFIG[f'keyword_{weight_level}']
                    pos_weight = self._get_position_weight(match.start(), text_length)
                    score += base_weight * pos_weight
            
            diversity = 0.0
            if matched_keywords:
                total_keywords = sum(
                    len(self.TASK_KEYWORDS[task_type][level])
                    for level in ['high', 'medium', 'low']
                )
                diversity = self._calculate_diversity(matched_keywords, total_keywords)
                score += diversity * self.WEIGHT_CONFIG['diversity_bonus']
            
            scores[task_type] = score
            match_details[task_type] = {
                'score': score,
                'matched_keywords': list(matched_keywords),
                'diversity': diversity
            }
        
        if not any(scores.values()):
            return TaskType.GENERAL, 0.3, {'reason': 'no_matches'}
        
        best_task = max(scores, key=scores.get)
        best_score = scores[best_task]
        
        # 调整 sigmoid 参数，提高置信度
        # 使用更激进的参数，让置信度更快提升
        confidence = 1 / (1 + math.exp(-0.5 * (best_score - 1.5)))
        
        # 大幅降低长文本惩罚（仅对超长文本轻微惩罚）
        if text_length > 2000:  # 提高阈值
            length_penalty = self.WEIGHT_CONFIG['length_penalty'] * ((text_length - 2000) / 100)
            confidence = max(0.5, confidence * (1 - min(length_penalty, 0.2)))  # 最多惩罚20%
        
        confidence = max(0.35, min(1.0, confidence))
        
        return best_task, confidence, match_details[best_task]
    
    async def classify(self, messages: list, **kwargs) -> ClassificationResult:
        text = self._extract_text(messages)
        estimated_tokens = self._count_tokens(messages)
        detected_lang = self._detect_language(text)
        
        is_long_context = estimated_tokens > self.LONG_CONTEXT_THRESHOLD
        if is_long_context:
            return ClassificationResult(
                task_type=TaskType.LONG_CONTEXT,
                confidence=1.0,
                detected_lang=detected_lang,
                estimated_tokens=estimated_tokens,
                is_long_context=True,
                metadata={
                    "trigger": "token_threshold",
                    "token_count": estimated_tokens
                }
            )
        
        task_type, confidence, match_info = self._rule_based_classify(text)
        
        # 中文语言处理逻辑
        # 只有在无法识别出具体任务类型时，才判定为CHINESE类型
        if detected_lang == "zh":
            if task_type == TaskType.GENERAL:
                # 中文通用请求，没有匹配到任何任务关键词
                task_type = TaskType.CHINESE
                confidence = 0.8
            else:
                # 中文特定任务（coding/reasoning等），给予语言加成
                # 但不要改变任务类型
                confidence = min(1.0, confidence + self.WEIGHT_CONFIG['lang_match_bonus'])
        
        return ClassificationResult(
            task_type=task_type,
            confidence=confidence,
            detected_lang=detected_lang,
            estimated_tokens=estimated_tokens,
            is_long_context=False,
            metadata={
                "method": "rule_based_enhanced",
                "match_info": match_info,
                "text_length": len(text)
            }
        )
