import re

# 核心联网搜索功能
def get_search_background(query: str, max_length: int = 1500) -> str:
    try:
        from gupao.retrievor import q_searching
        search_results = q_searching(query)
        cleaned_results = re.sub(r'\s+', ' ', search_results).strip()
        return cleaned_results[:max_length]
    except Exception as e:
        print(f"联网搜索失败：{str(e)}")
        return ""
