"""
测试文件：演示如何使用 rerank 功能

测试内容：
1. 测试 rerank 模块的基本功能
2. 对比使用 rerank 前后的检索结果
3. 测试不同的 rerank 方法（LLM 和文本相似度）
"""

import os
import sys

# 添加项目根目录到路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.rerank2 import rerank, rerank_with_llm, rerank_with_text_similarity
from search.retriever import vector_search
from kb.kb_paths import get_kb_paths
from kb.kb_config import DEFAULT_KB


def test_rerank_basic():
    """测试1：基本的 rerank 功能"""
    print("\n" + "="*60)
    print("测试1：基本的 rerank 功能")
    print("="*60)
    
    # 模拟一些候选文档
    query = "幽门螺杆菌感染的治疗方法"
    candidates = [
        {"id": "chunk1", "chunk": "幽门螺杆菌感染是常见的胃部疾病。治疗方法包括三联疗法和四联疗法。"},
        {"id": "chunk2", "chunk": "人工智能是计算机科学的一个分支。它试图理解智能的实质。"},
        {"id": "chunk3", "chunk": "三联疗法包括质子泵抑制剂、克拉霉素和阿莫西林。治疗周期一般为10-14天。"},
        {"id": "chunk4", "chunk": "深度学习是机器学习的一个子集。它使用神经网络来模拟人脑。"},
        {"id": "chunk5", "chunk": "幽门螺杆菌可以通过多种方式传播，包括口-口传播和粪-口传播。"},
    ]
    
    print(f"查询: {query}")
    print(f"原始候选文档数量: {len(candidates)}")
    print("\n原始顺序:")
    for i, cand in enumerate(candidates, 1):
        print(f"  {i}. {cand['chunk'][:60]}...")
    
    # 使用文本相似度方法进行 rerank（更快，不需要调用 LLM）
    print("\n使用文本相似度方法进行 rerank...")
    reranked = rerank_with_text_similarity(query, candidates, top_k=3)
    
    print(f"\n重排序后（返回前3个）:")
    for i, cand in enumerate(reranked, 1):
        print(f"  {i}. {cand['chunk'][:60]}...")
    
    print("\n✓ 基本 rerank 功能测试完成")


def test_rerank_with_llm():
    """测试2：使用 LLM 方法进行 rerank"""
    print("\n" + "="*60)
    print("测试2：使用 LLM 方法进行 rerank")
    print("="*60)
    
    query = "幽门螺杆菌感染的治疗方案"
    candidates = [
        {"id": "chunk1", "chunk": "幽门螺杆菌感染是常见的胃部疾病。治疗方法包括三联疗法和四联疗法。"},
        {"id": "chunk2", "chunk": "人工智能是计算机科学的一个分支。它试图理解智能的实质。"},
        {"id": "chunk3", "chunk": "三联疗法包括质子泵抑制剂、克拉霉素和阿莫西林。治疗周期一般为10-14天。"},
        {"id": "chunk4", "chunk": "深度学习是机器学习的一个子集。它使用神经网络来模拟人脑。"},
        {"id": "chunk5", "chunk": "幽门螺杆菌可以通过多种方式传播，包括口-口传播和粪-口传播。"},
    ]
    
    print(f"查询: {query}")
    print(f"候选文档数量: {len(candidates)}")
    print("\n注意: LLM rerank 需要调用 API，可能需要一些时间...")
    
    try:
        reranked = rerank_with_llm(query, candidates, top_k=3, batch_size=3)
        
        print(f"\nLLM 重排序后（返回前3个）:")
        for i, cand in enumerate(reranked, 1):
            print(f"  {i}. {cand['chunk'][:60]}...")
        
        print("\n✓ LLM rerank 测试完成")
    except Exception as e:
        print(f"\n❌ LLM rerank 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_vector_search_with_rerank():
    """测试3：在 vector_search 中集成 rerank"""
    print("\n" + "="*60)
    print("测试3：在 vector_search 中集成 rerank")
    print("="*60)
    
    # 使用默认知识库
    kb_name = DEFAULT_KB
    kb_paths = get_kb_paths(kb_name)
    index_path = kb_paths["index_path"]
    metadata_path = kb_paths["metadata_path"]
    
    # 检查索引是否存在
    if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
        print(f"警告: 知识库 '{kb_name}' 的索引文件不存在")
        print(f"  索引路径: {index_path}")
        print(f"  元数据路径: {metadata_path}")
        print("  请先运行文件上传和向量化测试来创建索引")
        return
    
    query = "幽门螺杆菌感染的治疗"
    limit = 5
    
    print(f"查询: {query}")
    print(f"知识库: {kb_name}")
    print(f"检索数量: {limit}")
    
    # 不使用 rerank 的检索
    print("\n--- 不使用 rerank 的检索结果 ---")
    results_without_rerank = vector_search(
        query=query,
        index_path=index_path,
        metadata_path=metadata_path,
        limit=limit,
        use_rerank=False
    )
    
    print(f"检索到 {len(results_without_rerank)} 个结果:")
    for i, result in enumerate(results_without_rerank[:3], 1):
        chunk_text = result.get('chunk', '')[:80]
        print(f"  {i}. {chunk_text}...")
    
    # 使用 rerank 的检索（文本相似度方法，更快）
    print("\n--- 使用 rerank（文本相似度方法）的检索结果 ---")
    results_with_rerank = vector_search(
        query=query,
        index_path=index_path,
        metadata_path=metadata_path,
        limit=limit,
        use_rerank=True,
        rerank_method="text_similarity",
        rerank_top_k=limit
    )
    
    print(f"检索到 {len(results_with_rerank)} 个结果:")
    for i, result in enumerate(results_with_rerank[:3], 1):
        chunk_text = result.get('chunk', '')[:80]
        print(f"  {i}. {chunk_text}...")
    
    # 对比结果
    print("\n--- 结果对比 ---")
    print(f"不使用 rerank: 返回 {len(results_without_rerank)} 个结果")
    print(f"使用 rerank: 返回 {len(results_with_rerank)} 个结果")
    
    # 检查结果是否不同
    if results_without_rerank and results_with_rerank:
        first_without = results_without_rerank[0].get('id', '')
        first_with = results_with_rerank[0].get('id', '')
        if first_without != first_with:
            print("✓ rerank 改变了结果的排序顺序")
        else:
            print("注意: rerank 后的第一个结果与未使用 rerank 时相同")
    
    print("\n✓ vector_search 集成 rerank 测试完成")


def test_rerank_methods_comparison():
    """测试4：对比不同 rerank 方法"""
    print("\n" + "="*60)
    print("测试4：对比不同 rerank 方法")
    print("="*60)
    
    query = "治疗幽门螺杆菌感染"
    candidates = [
        {"id": "chunk1", "chunk": "幽门螺杆菌感染是常见的胃部疾病。治疗方法包括三联疗法和四联疗法。"},
        {"id": "chunk2", "chunk": "人工智能是计算机科学的一个分支。"},
        {"id": "chunk3", "chunk": "三联疗法包括质子泵抑制剂、克拉霉素和阿莫西林。治疗周期一般为10-14天。"},
        {"id": "chunk4", "chunk": "深度学习是机器学习的一个子集。"},
        {"id": "chunk5", "chunk": "幽门螺杆菌可以通过多种方式传播。"},
    ]
    
    print(f"查询: {query}")
    print(f"候选文档数量: {len(candidates)}")
    
    # 方法1：文本相似度
    print("\n--- 方法1：文本相似度 rerank ---")
    reranked_text = rerank_with_text_similarity(query, candidates, top_k=3)
    for i, cand in enumerate(reranked_text, 1):
        print(f"  {i}. {cand['chunk'][:60]}...")
    
    # 方法2：LLM（如果可用）
    print("\n--- 方法2：LLM rerank ---")
    print("注意: 这需要调用 LLM API，可能需要一些时间...")
    try:
        reranked_llm = rerank_with_llm(query, candidates, top_k=3, batch_size=3)
        for i, cand in enumerate(reranked_llm, 1):
            print(f"  {i}. {cand['chunk'][:60]}...")
        
        # 对比两种方法的结果
        print("\n--- 方法对比 ---")
        text_ids = [c['id'] for c in reranked_text]
        llm_ids = [c['id'] for c in reranked_llm]
        print(f"文本相似度方法排序: {text_ids}")
        print(f"LLM 方法排序: {llm_ids}")
        
        if text_ids == llm_ids:
            print("两种方法的结果排序相同")
        else:
            print("两种方法的结果排序不同")
            
    except Exception as e:
        print(f"LLM rerank 失败: {e}")
        print("（这是正常的，如果 API 不可用或网络问题）")
    
    print("\n✓ rerank 方法对比测试完成")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("Rerank 功能测试")
    print("="*60)
    
    try:
        # 测试1：基本功能
        test_rerank_basic()
        
        # 测试2：LLM 方法（可选，需要 API）
        print("\n是否测试 LLM rerank？（需要调用 API，可能需要一些时间）")
        print("如果不想测试，可以直接跳过...")
        # test_rerank_with_llm()  # 取消注释以测试
        
        # 测试3：集成到 vector_search
        test_vector_search_with_rerank()
        
        # 测试4：方法对比
        test_rerank_methods_comparison()
        
        print("\n" + "="*60)
        print("所有测试完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

