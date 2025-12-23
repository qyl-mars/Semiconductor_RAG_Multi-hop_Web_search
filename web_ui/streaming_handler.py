from kb.kb_config import  DEFAULT_KB
from kb.kb_paths import  get_kb_paths
from concurrent.futures import ThreadPoolExecutor
from search.web_search import get_search_background
import os
from llm.answer_generator import generate_answer_from_deepseek
from search.retriever import vector_search
from llm.llm_client import client
from config.configs import Config
from rag.multi_hop_rag import ReasoningRAG
import traceback
from typing import List

# 修改以支持多知识库的流式响应函数
def process_question_with_reasoning(question: str, kb_name: str = DEFAULT_KB, use_search: bool = True, use_table_format: bool = False, multi_hop: bool = False, chat_history: List = None):
    """增强版process_question，支持流式响应，实时显示检索和推理过程，支持多知识库和对话历史"""
    try:
        kb_paths = get_kb_paths(kb_name)
        index_path = kb_paths["index_path"]
        metadata_path = kb_paths["metadata_path"]

        # 构建带对话历史的问题
        if chat_history and len(chat_history) > 0:
            # 构建对话上下文
            context = "之前的对话内容：\n"
            for user_msg, assistant_msg in chat_history[-3:]:  # 只取最近3轮对话
                context += f"用户：{user_msg}\n"
                context += f"助手：{assistant_msg}\n"
            context += f"\n当前问题：{question}"
            enhanced_question = f"基于以下对话历史，回答用户的当前问题。\n{context}"
        else:
            enhanced_question = question

        # 初始状态
        search_result = "联网搜索进行中..." if use_search else "未启用联网搜索"

        if multi_hop:
            reasoning_status = f"正在准备对知识库 '{kb_name}' 进行多跳推理检索..."
            search_display = f"### 联网搜索结果\n{search_result}\n\n### 推理状态\n{reasoning_status}"
            yield search_display, "正在启动多跳推理流程..."
        else:
            reasoning_status = f"正在准备对知识库 '{kb_name}' 进行向量检索..."
            search_display = f"### 联网搜索结果\n{search_result}\n\n### 检索状态\n{reasoning_status}"
            yield search_display, "正在启动简单检索流程..."

        # 如果启用，并行运行搜索
        search_future = None
        with ThreadPoolExecutor(max_workers=1) as executor:
            if use_search:
                search_future = executor.submit(get_search_background, question)

        # 检查索引是否存在
        if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
            # 如果索引不存在，提前返回
            if search_future:
                # 等待搜索结果
                search_result = "等待联网搜索结果..."
                search_display = f"### 联网搜索结果\n{search_result}\n\n### 检索状态\n知识库 '{kb_name}' 中未找到索引"
                yield search_display, "等待联网搜索结果..."

                search_result = search_future.result() or "未找到相关网络信息"
                system_prompt = "你是一名医疗专家。请考虑对话历史并回答用户的问题。"
                if use_table_format:
                    system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"
                answer = generate_answer_from_deepseek(enhanced_question, system_prompt=system_prompt, background_info=f"[联网搜索结果]：{search_result}")

                search_display = f"### 联网搜索结果\n{search_result}\n\n### 检索状态\n无法在知识库 '{kb_name}' 中进行本地检索（未找到索引）"
                yield search_display, answer
            else:
                yield f"知识库 '{kb_name}' 中未找到索引，且未启用联网搜索", "无法回答您的问题。请先上传文件到该知识库或启用联网搜索。"
            return

        # 开始流式处理
        current_answer = "正在分析您的问题..."

        if multi_hop:
            # 使用多跳推理的流式接口
            reasoning_rag = ReasoningRAG(
                index_path=index_path,
                metadata_path=metadata_path,
                max_hops=3,
                initial_candidates=5,
                refined_candidates=3,
                verbose=True
            )

            # 使用enhanced_question进行检索
            for step_result in reasoning_rag.stream_retrieve_and_answer(enhanced_question, use_table_format):
                # 更新当前状态
                status = step_result["status"]
                reasoning_display = step_result["reasoning_display"]

                # 如果有新的答案，更新
                if step_result["answer"]:
                    current_answer = step_result["answer"]

                # 如果搜索结果已返回，更新搜索结果
                if search_future and search_future.done():
                    search_result = search_future.result() or "未找到相关网络信息"

                # 构建并返回当前状态
                current_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 推理状态\n{status}\n\n{reasoning_display}"
                yield current_display, current_answer
        else:
            # 简单向量检索的流式处理
            yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n正在执行向量相似度搜索...", "正在检索相关信息..."

            # 执行简单向量搜索，使用enhanced_question
            try:
                search_results = vector_search(enhanced_question, index_path, metadata_path, limit=5)

                if not search_results:
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n未找到相关信息", f"知识库 '{kb_name}' 中未找到相关信息。"
                    current_answer = f"知识库 '{kb_name}' 中未找到相关信息。"
                else:
                    # 显示检索到的信息
                    chunks_detail = "\n\n".join \
                        ([f"**相关信息 { i +1}**:\n{result['chunk']}" for i, result in enumerate(search_results[:5])])
                    chunks_preview = "\n".join \
                        ([f"- {result['chunk'][:100]}..." for i, result in enumerate(search_results[:3])])
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n找到 {len(search_results)} 个相关信息块\n\n### 检索到的信息预览\n{chunks_preview}", "正在生成答案..."

                    # 生成答案
                    background_chunks = "\n\n".join([f"[相关信息 { i +1}]: {result['chunk']}"
                                                     for i, result in enumerate(search_results)])

                    system_prompt = "你是一名医疗专家。基于提供的背景信息和对话历史回答用户的问题。"
                    if use_table_format:
                        system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"

                    user_prompt = f"""
                    {enhanced_question}

                    背景信息：
                    {background_chunks}

                    请基于以上背景信息和对话历史回答用户的问题。
                    """

                    response = client.chat.completions.create(
                        model=Config.llm_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )

                    current_answer = response.choices[0].message.content.strip()
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n检索完成，已生成答案\n\n### 检索到的内容\n{chunks_detail}", current_answer

            except Exception as e:
                error_msg = f"检索过程中出错: {str(e)}"
                yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n{error_msg}", f"检索过程中出错: {str(e)}"
                current_answer = f"检索过程中出错: {str(e)}"

        # 检索完成后，如果有搜索结果，可以考虑合并知识
        if search_future and search_future.done():
            search_result = search_future.result() or "未找到相关网络信息"

            # 如果同时有搜索结果和本地检索结果，可以考虑合并
            if search_result and current_answer and current_answer not in ["正在分析您的问题...", "本地知识库中未找到相关信息。"]:
                status_text = "正在合并联网搜索和知识库结果..."
                if multi_hop:
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 推理状态\n{status_text}", current_answer
                else:
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n{status_text}", current_answer

                # 合并结果
                system_prompt = "你是一名医疗专家，请整合网络搜索和本地知识库提供全面的解答。请考虑对话历史。"

                if use_table_format:
                    system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"

                user_prompt = f"""
                {enhanced_question}

                网络搜索结果：{search_result}

                本地知识库分析：{current_answer}

                请根据以上信息和对话历史，提供一个综合的回答。确保使用Markdown表格来呈现适合表格形式的信息。
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

                    final_status = "已整合联网和知识库结果"
                    if multi_hop:
                        final_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 本地知识库分析\n已完成多跳推理分析，检索到的内容已在上方显示\n\n### 综合分析\n{final_status}"
                    else:
                        # 获取之前检索到的内容
                        chunks_info = "".join \
                            ([part.split("### 检索到的内容\n")[-1] if "### 检索到的内容\n" in part else "" for part in search_display.split("### 联网搜索结果")])
                        if not chunks_info.strip():
                            chunks_info = "检索内容已在上方显示"
                        final_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 本地知识库分析\n已完成向量检索分析\n\n### 检索到的内容\n{chunks_info}\n\n### 综合分析\n{final_status}"

                    yield final_display, combined_answer
                except Exception as e:
                    # 如果合并失败，使用现有答案
                    error_status = f"合并结果失败: {str(e)}"
                    if multi_hop:
                        final_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 本地知识库分析\n已完成多跳推理分析，检索到的内容已在上方显示\n\n### 综合分析\n{error_status}"
                    else:
                        # 获取之前检索到的内容
                        chunks_info = "".join \
                            ([part.split("### 检索到的内容\n")[-1] if "### 检索到的内容\n" in part else "" for part in search_display.split("### 联网搜索结果")])
                        if not chunks_info.strip():
                            chunks_info = "检索内容已在上方显示"
                        final_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 本地知识库分析\n已完成向量检索分析\n\n### 检索到的内容\n{chunks_info}\n\n### 综合分析\n{error_status}"

                    yield final_display, current_answer

    except Exception as e:
        error_msg = f"处理失败：{str(e)}\n{traceback.format_exc()}"
        yield f"### 错误信息\n{error_msg}", f"处理您的问题时遇到错误：{str(e)}"

