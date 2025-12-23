"""
Rerank 模块：对检索结果进行重排序

提供两种重排序方法：
1. 基于 LLM 的重排序：使用 LLM 评估每个文档与查询的相关性
2. 基于文本相似度的重排序：使用简单的文本匹配作为备选方案
"""

from typing import List, Dict, Any, Optional
from llm.llm_client import client
from config.configs import Config
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback


def rerank_with_llm(query: str, 
                    candidates: List[Dict[str, Any]], 
                    top_k: Optional[int] = None,
                    batch_size: int = 5) -> List[Dict[str, Any]]:
    """
    使用 LLM 对检索结果进行重排序
    
    Args:
        query: 用户查询
        candidates: 候选文档列表，每个元素应包含 'chunk' 字段（文本内容）
        top_k: 返回前 k 个结果，如果为 None 则返回所有结果
        batch_size: 批量处理的大小，避免一次性处理太多文档
        
    Returns:
        重排序后的文档列表，按相关性从高到低排序
    """
    if not candidates:
        return []
    
    # 如果候选数量较少，直接处理
    if len(candidates) <= batch_size:
        scored_candidates = _score_candidates_batch(query, candidates)
    else:
        # 批量处理
        scored_candidates = []
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            batch_scores = _score_candidates_batch(query, batch)
            scored_candidates.extend(batch_scores)
    
    # 按分数从高到低排序
    scored_candidates.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
    
    # 移除临时添加的分数字段，保持原始数据结构
    result = []
    for item in scored_candidates:
        # 创建一个新字典，包含原始数据但不包含 rerank_score
        ranked_item = {k: v for k, v in item.items() if k != 'rerank_score'}
        result.append(ranked_item)
    
    # 返回前 top_k 个结果
    if top_k is not None and top_k > 0:
        return result[:top_k]
    
    return result


def _score_candidates_batch(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    批量评估候选文档与查询的相关性分数
    
    Args:
        query: 用户查询
        candidates: 候选文档列表
        
    Returns:
        包含 rerank_score 字段的候选文档列表
    """
    if not candidates:
        return []
    
    # 准备评估提示
    candidates_text = ""
    for i, candidate in enumerate(candidates):
        chunk_text = candidate.get('chunk', '')
        # 限制文本长度，避免超出模型限制
        chunk_preview = chunk_text[:500] if len(chunk_text) > 500 else chunk_text
        candidates_text += f"\n[文档 {i + 1}]:\n{chunk_preview}\n"
    
    system_prompt = """你是一个信息检索评估专家。你的任务是根据用户查询，评估每个文档的相关性。

请为每个文档给出一个相关性分数（0-100），其中：
- 90-100: 高度相关，直接回答查询
- 70-89: 相关，包含有用信息
- 50-69: 部分相关，有一些相关信息
- 30-49: 低相关性，只有少量相关信息
- 0-29: 不相关

请以 JSON 格式返回，格式如下：
{
  "scores": [分数1, 分数2, ...]
}

分数数组的顺序应该与提供的文档顺序一致。"""
    
    user_prompt = f"""用户查询：{query}

需要评估的文档：
{candidates_text}

请为每个文档给出相关性分数（0-100），并以 JSON 格式返回分数数组。"""
    
    try:
        response = client.chat.completions.create(
            model=Config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1  # 降低温度以获得更稳定的评分
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # 解析 JSON 响应
        import json
        try:
            result = json.loads(result_text)
            scores = result.get('scores', [])
            
            # 确保分数数量与候选文档数量一致
            if len(scores) != len(candidates):
                print(f"警告: LLM 返回的分数数量 ({len(scores)}) 与候选文档数量 ({len(candidates)}) 不一致")
                # 如果分数不足，用 0 填充；如果过多，截断
                if len(scores) < len(candidates):
                    scores.extend([0] * (len(candidates) - len(scores)))
                else:
                    scores = scores[:len(candidates)]
            
            # 为每个候选文档添加分数
            scored_candidates = []
            for i, candidate in enumerate(candidates):
                score = scores[i] if i < len(scores) else 0
                candidate_copy = candidate.copy()
                candidate_copy['rerank_score'] = float(score)
                scored_candidates.append(candidate_copy)
            
            return scored_candidates
            
        except json.JSONDecodeError:
            print(f"警告: 无法解析 LLM 返回的 JSON: {result_text[:200]}...")
            # 回退：尝试从文本中提取数字
            return _fallback_scoring(query, candidates)
            
    except Exception as e:
        print(f"LLM 重排序出错: {e}")
        traceback.print_exc()
        # 回退到简单的文本相似度评分
        return _fallback_scoring(query, candidates)


def _fallback_scoring(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    回退方案：基于简单的文本相似度评分
    
    Args:
        query: 用户查询
        candidates: 候选文档列表
        
    Returns:
        包含 rerank_score 字段的候选文档列表
    """
    query_words = set(re.findall(r'\w+', query.lower()))
    
    scored_candidates = []
    for candidate in candidates:
        chunk_text = candidate.get('chunk', '').lower()
        chunk_words = set(re.findall(r'\w+', chunk_text))
        
        # 计算词重叠率
        if len(query_words) == 0:
            score = 0
        else:
            overlap = len(query_words & chunk_words)
            score = (overlap / len(query_words)) * 100
        
        candidate_copy = candidate.copy()
        candidate_copy['rerank_score'] = float(score)
        scored_candidates.append(candidate_copy)
    
    return scored_candidates


def rerank_with_text_similarity(query: str, 
                                candidates: List[Dict[str, Any]], 
                                top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    基于文本相似度的重排序（简单实现）
    
    Args:
        query: 用户查询
        candidates: 候选文档列表
        top_k: 返回前 k 个结果
        
    Returns:
        重排序后的文档列表
    """
    scored_candidates = _fallback_scoring(query, candidates)
    scored_candidates.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
    
    # 移除分数字段
    result = []
    for item in scored_candidates:
        ranked_item = {k: v for k, v in item.items() if k != 'rerank_score'}
        result.append(ranked_item)
    
    if top_k is not None and top_k > 0:
        return result[:top_k]
    
    return result


def rerank(query: str, 
           candidates: List[Dict[str, Any]], 
           method: str = "llm",
           top_k: Optional[int] = None,
           batch_size: int = 5) -> List[Dict[str, Any]]:
    """
    统一的 rerank 接口
    
    Args:
        query: 用户查询
        candidates: 候选文档列表
        method: 重排序方法，"llm" 或 "text_similarity"
        top_k: 返回前 k 个结果
        batch_size: LLM 方法使用的批量处理大小
        
    Returns:
        重排序后的文档列表
    """
    if method == "llm":
        return rerank_with_llm(query, candidates, top_k=top_k, batch_size=batch_size)
    elif method == "text_similarity":
        return rerank_with_text_similarity(query, candidates, top_k=top_k)
    else:
        print(f"警告: 未知的重排序方法 '{method}'，使用默认的 'llm' 方法")
        return rerank_with_llm(query, candidates, top_k=top_k, batch_size=batch_size)

