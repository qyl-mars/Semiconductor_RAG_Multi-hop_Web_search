from typing import Tuple, Dict
from kb.kb_paths import get_kb_paths
from config.configs import Config
from rag.multi_hop_rag import ReasoningRAG
from search.retriever import vector_search
from llm.llm_client import client

def multi_hop_generate_answer(query: str, kb_name: str, use_table_format: bool = False,
                              system_prompt: str = "你是一名半导体专家。") -> Tuple[str, Dict]:
    """使用多跳推理RAG生成答案，基于指定知识库"""
    kb_paths = get_kb_paths(kb_name)

    reasoning_rag = ReasoningRAG(
        index_path=kb_paths["index_path"],
        metadata_path=kb_paths["metadata_path"],
        max_hops=3,
        initial_candidates=5,
        refined_candidates=3,
        reasoning_model=Config.llm_model,
        verbose=True
    )

    answer, debug_info = reasoning_rag.retrieve_and_answer(query, use_table_format)
    return answer, debug_info


# 使用简单向量检索生成答案，基于指定知识库
def simple_generate_answer(query: str, kb_name: str, use_table_format: bool = False) -> str:
    """使用简单的向量检索生成答案，不使用多跳推理"""
    try:
        kb_paths = get_kb_paths(kb_name)

        # 使用基本向量搜索
        search_results = vector_search(query, kb_paths["index_path"], kb_paths["metadata_path"], limit=5)

        if not search_results:
            return "未找到相关信息。"

        # 准备背景信息
        background_chunks = "\n\n".join([f"[相关信息 {i + 1}]: {result['chunk']}"
                                         for i, result in enumerate(search_results)])

        # 生成答案
        system_prompt = "你是一名半导体专家。基于提供的背景信息回答用户的问题。"

        if use_table_format:
            system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"

        user_prompt = f"""
        问题：{query}

        背景信息：
        {background_chunks}

        请基于以上背景信息回答用户的问题。
        """

        response = client.chat.completions.create(
            model=Config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"生成答案时出错：{str(e)}"
