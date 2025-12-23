from config.configs import Config
from openai import OpenAI
import numpy as np
from ingest.text_cleaner import clean_text
import traceback


# 向量化查询 - 通用函数，被多处使用
def vectorize_query(query, model_name=Config.model_name, batch_size=Config.batch_size) -> np.ndarray:
    """向量化文本查询，返回嵌入向量，改进错误处理和批处理"""
    embedding_client = OpenAI(
        api_key=Config.api_key,
        base_url=Config.base_url
    )

    if not query:
        print("警告: 传入向量化的查询为空")
        return np.array([])

    # 将单个查询字符串转换为列表
    if isinstance(query, str):
        query = [query]

    # 验证所有查询文本，确保它们符合API要求
    valid_queries = []
    for q in query:
        if not q or not isinstance(q, str):
            print(f"警告: 跳过无效查询: {type(q)}")
            continue

        # 清理文本并检查长度
        clean_q = clean_text(q)
        if not clean_q:
            print("警告: 清理后的查询文本为空")
            continue

        # 检查长度是否在API限制范围内
        if len(clean_q) > 8000:
            print(f"警告: 查询文本过长 ({len(clean_q)} 字符)，截断至 8000 字符")
            clean_q = clean_q[:8000]

        valid_queries.append(clean_q)

    if not valid_queries:
        print("错误: 所有查询都无效，无法进行向量化")
        return np.array([])

    # 分批处理有效查询
    all_vectors = []
    for i in range(0, len(valid_queries), batch_size):
        batch = valid_queries[i:i + batch_size]
        try:
            # 记录批次信息便于调试
            print(f"正在向量化批次 {i // batch_size + 1}/{(len(valid_queries) - 1) // batch_size + 1}, "
                  f"包含 {len(batch)} 个文本，第一个文本长度: {len(batch[0][:50])}...")

            completion = embedding_client.embeddings.create(
                model=model_name,
                input=batch,
                dimensions=Config.dimensions,
                encoding_format="float"
            )
            vectors = [embedding.embedding for embedding in completion.data]
            all_vectors.extend(vectors)
            print(f"批次 {i // batch_size + 1} 向量化成功，获得 {len(vectors)} 个向量")
        except Exception as e:
            print(f"向量化批次 {i // batch_size + 1} 失败：{str(e)}")
            print(f"问题批次中的第一个文本: {batch[0][:100]}...")
            traceback.print_exc()
            # 如果是第一批就失败，直接返回空数组
            if i == 0:
                return np.array([])
            # 否则返回已处理的向量
            break

    # 检查是否获得了任何向量
    if not all_vectors:
        print("错误: 向量化过程没有产生任何向量")
        return np.array([])

    return np.array(all_vectors)
