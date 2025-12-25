import json
import numpy as np
import faiss
from llm.embedding_client import vectorize_query
from search.rerank2 import rerank

# 简单的向量搜索，用于基本对比
def vector_search(query, index_path, metadata_path, limit, use_rerank=False, rerank_method="llm", rerank_top_k=None):
    """
    向量搜索函数，支持可选的 rerank 重排序
    
    Args:
        query: 查询文本
        index_path: FAISS 索引文件路径
        metadata_path: 元数据文件路径
        limit: 初始检索返回的候选数量
        use_rerank: 是否使用 rerank 重排序
        rerank_method: rerank 方法，"llm" 或 "text_similarity"
        rerank_top_k: rerank 后返回的结果数量，如果为 None 则返回所有重排序后的结果
        
    Returns:
        检索结果列表，如果启用 rerank 则按相关性重排序
    """
    query_vector = vectorize_query(query)
    if query_vector.size == 0:
        return []

    query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)

    index = faiss.read_index(index_path)
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except UnicodeDecodeError:
        print(f"警告：{metadata_path} 包含非法字符，使用 UTF-8 忽略错误重新加载")
        with open(metadata_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
            metadata = json.loads(content)

    # 初始向量检索：如果启用 rerank，检索更多候选以便重排序
    initial_limit = limit * 2 if use_rerank else limit
    D, I = index.search(query_vector, initial_limit)
    results = [metadata[i] for i in I[0] if i < len(metadata)]
    
    # 如果启用 rerank，对结果进行重排序
    if use_rerank and results:
        print(f"对 {len(results)} 个候选结果进行 rerank 重排序...")
        results = rerank(query, results, method=rerank_method, top_k=rerank_top_k or limit)
        print(f"rerank 完成，返回 {len(results)} 个结果")
    
    # 如果未启用 rerank，直接返回前 limit 个结果
    if not use_rerank:
        results = results[:limit]
    
    return results

