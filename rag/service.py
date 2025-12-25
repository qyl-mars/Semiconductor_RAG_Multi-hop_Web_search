from llm.llm_client import client
from kb.kb_config import  DEFAULT_KB
from kb.kb_paths import  get_kb_paths
import os
from llm.answer_generator import generate_answer_from_deepseek
from concurrent.futures import ThreadPoolExecutor, as_completed
from search.web_search import get_web_search_content
from rag.pipeline import simple_generate_answer


# 修改主要的问题处理函数以支持指定知识库
def ask_question_parallel(question: str, kb_name: str = DEFAULT_KB, use_search: bool = True, use_table_format: bool = False, multi_hop: bool = False) -> str:
    """基于指定知识库回答问题"""
    try:
        kb_paths = get_kb_paths(kb_name)
        index_path = kb_paths["index_path"]
        metadata_path = kb_paths["metadata_path"]

        search_background = ""
        local_answer = ""
        debug_info = {}

        # 并行处理
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}

            if use_search:
                futures[executor.submit(get_web_search_content, question)] = "search"

            if os.path.exists(index_path):
                if multi_hop:
                    # 使用多跳推理
                    futures[executor.submit(get_web_search_content, question, kb_name, use_table_format)] = "rag"
                else:
                    # 使用简单向量检索
                    futures[executor.submit(simple_generate_answer, question, kb_name, use_table_format)] = "simple"

            for future in as_completed(futures):
                result = future.result()
                if futures[future] == "search":
                    search_background = result or ""
                elif futures[future] == "rag":
                    local_answer, debug_info = result
                elif futures[future] == "simple":
                    local_answer = result

        # 如果同时有搜索和本地结果，合并它们
        if search_background and local_answer:
            system_prompt = "你是一名半导体专家，请整合网络搜索和本地知识库提供全面的解答。"

            table_instruction = ""
            if use_table_format:
                table_instruction = """
                请尽可能以Markdown表格的形式呈现你的回答，特别是对于症状、治疗方法、药物等结构化信息。

                请确保你的表格遵循正确的Markdown语法：
                | 列标题1 | 列标题2 | 列标题3 |
                | ------- | ------- | ------- |
                | 数据1   | 数据2   | 数据3   |
                """

            user_prompt = f"""
            问题：{question}

            网络搜索结果：{search_background}

            本地知识库分析：{local_answer}

            {table_instruction}

            请根据以上信息，提供一个综合的回答。
            """

            try:
                response = client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                combined_answer = response.choices[0].message.content.strip()
                return combined_answer
            except Exception as e:
                # 如果合并失败，回退到本地答案
                return local_answer
        elif local_answer:
            return local_answer
        elif search_background:
            # 仅从搜索结果生成答案
            system_prompt = "你是一名半导体专家。"
            if use_table_format:
                system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"
            return generate_answer_from_deepseek(question, system_prompt=system_prompt, background_info=f"[联网搜索结果]：{search_background}")
        else:
            return "未找到相关信息。"

    except Exception as e:
        return f"查询失败：{str(e)}"
