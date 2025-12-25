import json
import numpy as np
import faiss
import os
from llm.embedding_client import vectorize_query


def vector_search(query, index_path, metadata_path, limit=5):
    """
    基本向量搜索函数
    Args:
        query: 用户问题
        index_path: FAISS 索引路径
        metadata_path: 元数据路径
        limit: 返回数量 (由调用方控制，例如传入 50)
    """
    # 1. 向量化 Query
    query_vector = vectorize_query(query)

    # 判空处理
    if query_vector is None or (isinstance(query_vector, np.ndarray) and query_vector.size == 0):
        print("Warning: Query vectorization failed.")
        return []

    # FAISS 需要 float32 类型的二维数组
    query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)

    # 2. 加载索引
    if not os.path.exists(index_path):
        print(f"Error: Index file not found at {index_path}")
        return []

    try:
        index = faiss.read_index(index_path)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return []

    # 3. 加载元数据
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return []

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except UnicodeDecodeError:
        print(f"警告：{metadata_path} 编码异常，尝试忽略错误读取...")
        with open(metadata_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
            metadata = json.loads(content)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return []

    # 4. 执行搜索
    # 这里的 limit 是关键，streaming_handler 会传入 50
    try:
        D, I = index.search(query_vector, limit)
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return []

    # 5. 提取结果
    results = []
    # I[0] 是索引 ID 列表，D[0] 是距离/分数列表
    for i in I[0]:
        if i < len(metadata):
            # 确保返回的是完整的元数据对象
            item = metadata[i]
            # 可以在这里把 FAISS 的距离分数也带上，方便调试，但不是必须的
            # item['vector_score'] = float(D[0][idx])
            results.append(item)

    return results