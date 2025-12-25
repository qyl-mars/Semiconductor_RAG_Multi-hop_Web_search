import sys
import os
import time

# --- 1. 环境配置 (确保 Python 能找到你的模块) ---
# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将项目根目录添加到 sys.path 中 (假设你在根目录运行，这一步是双保险)
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from search.web_search import get_web_search_content, search_bing, web_search_and_rerank
except ImportError as e:
    print("❌ 导入模块失败！请确保你在项目根目录 'Rag/' 下运行此脚本。")
    print(f"错误详情: {e}")
    sys.exit(1)


def test_connectivity():
    """测试基础网络连接"""
    print("\n[1/3] 正在测试基础网络连接 (Ping cn.bing.com)...")
    try:
        import requests
        resp = requests.get("https://cn.bing.com", timeout=5)
        if resp.status_code == 200:
            print(f"✅ 网络通畅，状态码: {resp.status_code}")
            return True
        else:
            print(f"⚠️ 无法访问 Bing，状态码: {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ 网络连接失败: {e}")
        return False


def test_raw_crawler(query):
    """单独测试爬虫函数 (search_bing)，排除排序算法的干扰"""
    print(f"\n[2/3] 正在测试底层爬虫 search_bing('{query}')...")
    print("      (如果这一步失败，说明是 Cookie 过期了)")

    start_time = time.time()
    results = search_bing(query)
    end_time = time.time()

    if results and len(results) > 0:
        print(f"✅ 爬虫成功！抓取到 {len(results)} 条结果。")
        print(f"      耗时: {end_time - start_time:.2f} 秒")
        print(f"      第一条标题: {results[0].get('title', '无标题')}")
        print(f"      第一条链接: {results[0].get('url', '无链接')}")
        return True
    else:
        print("❌ 爬虫返回结果为空。")
        print("💡 建议：请更新 web_search.py 中的 Cookie。")
        return False


def test_full_pipeline(query):
    """测试完整流程 (爬虫 + 排序 + 清洗)"""
    print(f"\n[3/3] 正在测试完整流程 get_web_search_content('{query}')...")

    start_time = time.time()
    final_content = get_web_search_content(query, max_length=500)
    end_time = time.time()

    if final_content:
        print(f"✅ 完整流程成功！")
        print(f"      耗时: {end_time - start_time:.2f} 秒")
        print(f"      返回内容长度: {len(final_content)} 字符")
        print("-" * 30)
        print("预览内容 (前 200 字):")
        print(final_content[:200] + "...")
        print("-" * 30)
    else:
        print("❌ 完整流程失败，返回内容为空。")


if __name__ == "__main__":
    test_query = "目前最先进的半导体光刻机是哪个公司的"

    # 按顺序执行测试
    if test_connectivity():
        if test_raw_crawler(test_query):
            test_full_pipeline(test_query)
        else:
            print("\n⛔ 测试终止：底层爬虫失败，无需测试后续流程。")
    else:
        print("\n⛔ 测试终止：网络不通。")