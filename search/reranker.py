import requests
import json
import traceback
from typing import List, Dict
from config.configs import Config


class Reranker:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # =======================================================
        # ✅ 从 Config 中动态读取参数
        # 对应 config.configs.Config 中的 rerank_api_key 和 rerank_model
        # =======================================================
        self.api_key = getattr(cfg, 'rerank_api_key', None)
        self.model_name = getattr(cfg, 'rerank_model', "BAAI/bge-reranker-v2-m3")  # 默认 fallback

        # SiliconFlow 的 Rerank 端点
        self.api_url = "https://api.siliconflow.cn/v1/rerank"

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        使用配置的 Rerank 模型 (SiliconFlow API) 对文档进行重排序
        Args:
            query: 用户问题
            candidates: 候选文档列表 [{'chunk': '...', ...}]
            top_k: 返回前 K 个
        """
        if not candidates:
            return []

        # 检查 API Key 是否配置
        if not self.api_key:
            print("⚠️ [Reranker] 警告: Config 中未找到 'rerank_api_key'，跳过重排序，直接返回原顺序！")
            return candidates[:top_k]

        try:
            # 1. 准备 API 请求数据
            documents_text = [doc.get('chunk', '') for doc in candidates]

            payload = {
                "model": self.model_name,
                "query": query,
                "documents": documents_text,
                "return_documents": False,  # 只返回分数和索引
                "top_n": top_k
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # 2. 发送请求
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=5)

            # 检查 HTTP 状态码
            if response.status_code != 200:
                print(f"❌ [Reranker] API 错误: {response.status_code} - {response.text}")
                return candidates[:top_k]  # 出错降级

            # 3. 解析结果
            api_results = response.json().get('results', [])

            reranked_candidates = []
            for item in api_results:
                original_index = item['index']
                score = item['relevance_score']

                # 找到原始文档对象
                doc = candidates[original_index].copy()
                # 注入分数
                doc['rerank_score'] = score
                reranked_candidates.append(doc)

            return reranked_candidates

        except Exception as e:
            print(f"❌ [Reranker] 调用异常: {e}")
            traceback.print_exc()
            # 发生异常回退
            return candidates[:top_k]