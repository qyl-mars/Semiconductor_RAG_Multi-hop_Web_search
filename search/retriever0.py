import json
import numpy as np
import faiss
from llm.embedding_client import vectorize_query

# 简单的向量搜索，用于基本对比
def vector_search(query, index_path, metadata_path, limit):
    """基本向量搜索函数"""
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

    D, I = index.search(query_vector, limit)
    results = [metadata[i] for i in I[0] if i < len(metadata)]
    return results
