import sys
import os
import time
# 这个脚本模拟用户提问，调用修改后的 process_question_with_reasoning，你需要观察控制台输出的状态流。

# 确保能找到项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.streaming_handler import process_question_with_reasoning


def test_full_pipeline():
    print("========================================")
    print("🚀 正在测试完整 RAG 流水线 (简单检索 + Rerank)")
    print("========================================")

    # 模拟用户问题
    question = "什么是EUV光刻机？"  # 请确保你的本地知识库里有相关内容，否则会fallback到联网
    kb_name = "aaa"  # ⚠️ 请修改为你本地真实存在的知识库名字

    print(f"❓ 问题: {question}")
    print(f"📚 知识库: {kb_name}")
    print("----------------------------------------")

    try:
        # 调用生成器
        # multi_hop=False 以触发简单检索分支 (我们修改的那个分支)
        generator = process_question_with_reasoning(
            question=question,
            kb_name=kb_name,
            use_search=True,  # 开启联网以测试并行
            use_table_format=False,
            multi_hop=False
        )

        start_time = time.time()

        # 逐步消费生成器
        final_answer = ""
        print("\n📡 --- 实时状态流 ---")
        for update, answer in generator:
            # 解析 update 文本，寻找关键特征
            status_line = "未知状态"
            if "检索状态" in update:
                # 简单提取状态部分用于显示
                parts = update.split("检索状态")
                if len(parts) > 1:
                    status_line = parts[1].split("\n")[1].strip()

            print(f"⏱️ [{time.time() - start_time:.1f}s] 状态: {status_line}")

            # 检查关键日志特征
            if "精排" in status_line or "模型精准筛选" in status_line:
                print("   🌟 【验证成功】检测到 Rerank 正在运行！")

            if "扩大召回" in status_line or "广度召回" in status_line:
                print("   🔍 【验证成功】检测到 扩大召回 正在运行！")

            if answer:
                final_answer = answer

        print("\n----------------------------------------")
        print("💡 最终回答片段:")
        print(final_answer[:200] + "..." if len(final_answer) > 200 else final_answer)
        print("\n✅ 测试结束。")

    except Exception as e:
        print(f"\n❌ 管道运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # ⚠️ 运行前请确保：
    # 1. config/configs.py 里填了 rerank_api_key
    # 2. kb_name 变量改成你电脑上真实存在的知识库名
    test_full_pipeline()