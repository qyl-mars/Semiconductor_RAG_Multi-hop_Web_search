import sys
import os
import json

# --- 1. 路径配置 (防止 ModuleNotFoundError) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


def test_ddg_search_only():
    print("=" * 50)
    print(" 正在测试  纯联网功能 (不含 Rerank)")
    print("=" * 50)

    try:
        # 尝试导入
        from search.web_search import search_bing

        query = "2024年最先进的半导体光刻机型号"
        print(f"🔍 搜索关键词: [{query}]")
        print("⏳ 正在请求 Tavily API (可能需要几秒)...")

        # 执行搜索
        results = search_bing(query)

        # 验证结果
        if not results:
            print("❌ 搜索结果为空！")
            print("   可能原因：网络不通（需代理）或 DDG 服务暂时不可用。")
            return

        print(f"✅ 搜索成功！获取到 {len(results)} 条结果。\n")

        # 打印前2条数据的结构，供你检查是否符合 Rerank 的输入要求
        print("--- 数据结构预览 (前2条) ---")
        for i, item in enumerate(results[:2]):
            print(f"[{i + 1}] 标题: {item.get('title')}")
            print(f"    链接: {item.get('url')}")
            # 只截取前 50 个字展示
            text_preview = item.get('text', '')[:50].replace('\n', ' ')
            print(f"    正文: {text_preview}...")
            print("-" * 30)

    except ImportError:
        print("❌ 导入错误：找不到 search.web_search 模块。请确保你在 Rag 根目录下运行。")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")


if __name__ == "__main__":
    test_ddg_search_only()
