import sys
import os
# 这个脚本用于验证：你的 API Key 是否有效，以及 Rerank 是否真的把相关文档排到了前面。

# 确保能找到项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from search.reranker import Reranker
from config.configs import Config


def test_siliconflow_rerank():
    print("========================================")
    print("🧪 正在测试 SiliconFlow BGE Rerank API...")
    print("========================================")

    # 1. 初始化配置
    cfg = Config()
    if not getattr(cfg, 'rerank_api_key', None):
        print("❌ 错误: Config 中未找到 rerank_api_key，请先在 config/configs.py 中配置！")
        return

    print(f"🔑 使用 API Key: {cfg.rerank_api_key[:6]}******")
    print(f"🤖 使用模型: {getattr(cfg, 'rerank_model', 'Default')}")

    # 2. 构造模拟数据
    # 假设用户问“光刻机”，我们故意把不相关的放前面，相关的放后面
    query = "光刻机的作用是什么？"

    candidates = [
        {"chunk": "今天天气真不错，适合出去野餐。", "id": 1, "source": "noise_doc"},
        {"chunk": "苹果公司发布了最新的 iPhone 16。", "id": 2, "source": "noise_doc"},
        {"chunk": "ASML是全球最大的半导体设备制造商。", "id": 3, "source": "related_doc"},
        {"chunk": "光刻机是制造芯片的核心设备，利用光线将电路图投射到硅片上。", "id": 4, "source": "target_doc"},
        {"chunk": "Python是一种非常流行的编程语言。", "id": 5, "source": "noise_doc"}
    ]
    print(f"\nquery: {query}")
    print("\n📋 原始顺序 (Top 5):")
    for doc in candidates:
        print(f"   - [ID:{doc['id']}] {doc['chunk'][:30]}...")

    # 3. 执行 Rerank
    try:
        ranker = Reranker(cfg)
        reranked_results = ranker.rerank(query, candidates, top_k=5)

        print("\n✅ Rerank 成功！排序后结果 (按相关性降序):")
        for i, doc in enumerate(reranked_results):
            score = doc.get('rerank_score', 0)
            print(f"   {i + 1}. [分数: {score:.4f}] [ID:{doc['id']}] {doc['chunk']}")

        # 4. 简单断言
        top_doc_id = reranked_results[0]['id']
        if top_doc_id == 4:
            print("\n🎉 测试通过！最相关的文档 (ID:4) 排到了第一位。")
        else:
            print(f"\n⚠️ 测试存疑：最相关的文档没有排第一，当前第一是 ID:{top_doc_id}。")

    except Exception as e:
        print(f"\n❌ 测试失败，发生异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_siliconflow_rerank()