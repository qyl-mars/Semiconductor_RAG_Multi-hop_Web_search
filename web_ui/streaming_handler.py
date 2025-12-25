from kb.kb_config import DEFAULT_KB
from kb.kb_paths import get_kb_paths
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from search.web_search import get_web_search_content
import os
from llm.answer_generator import generate_answer_from_deepseek
from search.retriever import vector_search
from llm.llm_client import client
from config.configs import Config
from rag.multi_hop_rag import ReasoningRAG
import traceback
from typing import List
from utils.logger_config import setup_logger


# 设置日志
logger = setup_logger("streaming_handler.log")

# ==========================================
# 优化说明：
# 1. ThreadPoolExecutor 范围扩大，涵盖整个检索流程，实现真正的并行。
# 2. 移除 .done() 检查，改为 result(timeout=x) 智能等待，防止联网结果丢失。
# 3. 优化了数据获取：直接复用内存变量，不再通过 split 暴力解析字符串。
# 4. 统一使用 Config.llm_model，避免硬编码模型名称。
# ==========================================

def process_question_with_reasoning(question: str, kb_name: str = DEFAULT_KB, use_search: bool = True,
                                    use_table_format: bool = False, multi_hop: bool = False, chat_history: List = None):
    """增强版process_question，支持流式响应，并行处理联网与本地检索，支持多知识库和对话历史"""
    try:
        kb_paths = get_kb_paths(kb_name)
        index_path = kb_paths["index_path"]
        metadata_path = kb_paths["metadata_path"]

        # 1. 构建带对话历史的问题
        if chat_history and len(chat_history) > 0:
            context = "之前的对话内容：\n"
            for user_msg, assistant_msg in chat_history[-3:]:  # 只取最近3轮
                context += f"用户：{user_msg}\n"
                context += f"助手：{assistant_msg}\n"
            context += f"\n当前问题：{question}"
            enhanced_question = f"基于以下对话历史，回答用户的当前问题。\n{context}"
        else:
            enhanced_question = question

        # 2. 发送初始状态反馈
        search_result_placeholder = "联网搜索进行中..." if use_search else "未启用联网搜索"

        if multi_hop:
            reasoning_status = f"正在准备对知识库 '{kb_name}' 进行多跳推理检索..."
            search_display = f"### 联网搜索结果\n{search_result_placeholder}\n\n### 推理状态\n{reasoning_status}"
            yield search_display, "正在启动多跳推理流程..."
        else:
            reasoning_status = f"正在准备对知识库 '{kb_name}' 进行向量检索..."
            search_display = f"### 联网搜索结果\n{search_result_placeholder}\n\n### 检索状态\n{reasoning_status}"
            yield search_display, "正在启动简单检索流程..."

        # =========================================================================
        # 核心并行区：开启线程池，并保持直到所有任务结束
        # =========================================================================
        with ThreadPoolExecutor(max_workers=1) as executor:
            # 3.1 立即启动后台联网搜索 (非阻塞)
            search_future = None
            if use_search:
                logger.info("正在启动联网搜索...")
                search_future = executor.submit(get_web_search_content, question)

            # 3.2 并行执行：主线程继续处理本地逻辑

            # --- 分支 A: 索引不存在 (纯联网兜底) ---
            print(f"当前工作目录: {os.getcwd()}")
            logger.info("index_path:{index_path}")
            if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
                if search_future:
                    # 状态反馈
                    print("索引不存在，启动联网搜索")
                    yield f"### 联网搜索结果\n等待联网搜索结果...\n\n### 检索状态\n知识库 '{kb_name}' 中未找到索引", "等待联网搜索结果..."

                    try:
                        # 智能等待：最多等 60 秒
                        final_search_res = search_future.result(timeout=60)
                    except TimeoutError:
                        final_search_res = "网络搜索超时，未能获取相关信息。"
                    except Exception as e:
                        final_search_res = f"网络搜索出错: {str(e)}"

                    # 生成纯联网回答
                    system_prompt = "你是一名半导体专家。请考虑对话历史并回答用户的问题。"
                    if use_table_format:
                        system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"

                    answer = generate_answer_from_deepseek(
                        enhanced_question,
                        system_prompt=system_prompt,
                        background_info=f"[联网搜索结果]：{final_search_res}"
                    )

                    yield f"### 联网搜索结果\n{final_search_res}\n\n### 检索状态\n无法在知识库 '{kb_name}' 中进行本地检索（未找到索引）", answer
                else:
                    yield f"知识库 '{kb_name}' 中未找到索引，且未启用联网搜索", "无法回答您的问题。请先上传文件到该知识库或启用联网搜索。"

                # 结束函数 (with 会自动清理线程资源)
                print("退出函数，联网搜索结束")
                return

            # --- 分支 B: 索引存在 (本地检索流程) ---
            # 注意：此时 search_future 依然在后台运行中，这里是真正的并行

            current_answer = "正在分析您的问题..."
            local_chunks_info = ""  # 用于存储检索到的原始文本，供后续合并使用

            if multi_hop:
                # B.1 多跳推理模式
                reasoning_rag = ReasoningRAG(
                    index_path=index_path,
                    metadata_path=metadata_path,
                    max_hops=3,
                    initial_candidates=5,
                    refined_candidates=3,
                    verbose=True
                )

                for step_result in reasoning_rag.stream_retrieve_and_answer(enhanced_question, use_table_format):
                    status = step_result["status"]
                    reasoning_display = step_result["reasoning_display"]
                    if step_result["answer"]:
                        current_answer = step_result["answer"]

                    # 实时更新状态 (此时联网结果可能还没出来，显示占位符)
                    current_display = f"### 联网搜索结果\n{search_result_placeholder}\n\n### 知识库: {kb_name}\n### 推理状态\n{status}\n\n{reasoning_display}"
                    yield current_display, current_answer

                # 多跳结束后，将推理过程作为 context
                local_chunks_info = "（内容已包含在上方多跳推理过程中）"

            else:
                # B.2 简单向量检索模式
                yield f"### 联网搜索结果\n{search_result_placeholder}\n\n### 知识库: {kb_name}\n### 检索状态\n正在执行向量相似度搜索...", "正在检索相关信息..."

                try:
                    search_results = vector_search(enhanced_question, index_path, metadata_path, limit=5)

                    if not search_results:
                        current_answer = f"知识库 '{kb_name}' 中未找到相关信息。"
                        local_chunks_info = "无相关信息"
                        yield f"### 联网搜索结果\n{search_result_placeholder}\n\n### 知识库: {kb_name}\n### 检索状态\n未找到相关信息", current_answer
                    else:
                        # 格式化检索结果
                        local_chunks_info = "\n\n".join([f"**相关信息 {i + 1}**:\n{result['chunk']}" for i, result in
                                                         enumerate(search_results[:5])])
                        chunks_preview = "\n".join(
                            [f"- {result['chunk'][:100]}..." for i, result in enumerate(search_results[:3])])

                        yield f"### 联网搜索结果\n{search_result_placeholder}\n\n### 知识库: {kb_name}\n### 检索状态\n找到 {len(search_results)} 个相关信息块\n\n### 检索到的信息预览\n{chunks_preview}", "正在生成答案..."

                        # 生成本地答案
                        background_chunks = "\n\n".join(
                            [f"[相关信息 {i + 1}]: {result['chunk']}" for i, result in enumerate(search_results)])

                        system_prompt = "你是一名半导体专家。基于提供的背景信息和对话历史回答用户的问题。"
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

                        yield f"### 联网搜索结果\n{search_result_placeholder}\n\n### 知识库: {kb_name}\n### 检索状态\n检索完成，已生成答案\n\n### 检索到的内容\n{local_chunks_info}", current_answer

                except Exception as e:
                    error_msg = f"本地检索出错: {str(e)}"
                    current_answer = error_msg
                    yield f"### 联网搜索结果\n{search_result_placeholder}\n\n### 知识库: {kb_name}\n### 检索状态\n{error_msg}", error_msg

            # =====================================================================
            # 4. 最终合并阶段 (此时本地检索已完成，开始等待联网结果)
            # =====================================================================
            if search_future:
                final_search_res = "未找到相关网络信息"
                try:
                    # 【关键修复】智能等待联网结果，最多再等 15 秒
                    # 如果本地跑得快，这里会等待；如果联网已经跑完，这里立即返回
                    final_search_res = search_future.result(timeout=15) or "未找到相关网络信息"
                except TimeoutError:
                    final_search_res = "联网搜索超时 (超过15秒未能返回)"
                except Exception as e:
                    final_search_res = f"联网搜索发生错误: {str(e)}"

                # 判断是否需要合并 (必须两边都有有效内容)
                need_merge = (
                        final_search_res not in ["未找到相关网络信息", "联网搜索超时 (超过15秒未能返回)"]
                        and current_answer
                        and "未找到相关信息" not in current_answer
                        and "出错" not in current_answer
                )

                # 构造最终的 Display 文本 (复用 local_chunks_info，不再解析字符串)
                if multi_hop:
                    final_display_base = f"### 联网搜索结果\n{final_search_res}\n\n### 知识库: {kb_name}\n### 推理状态\n{{status}}\n\n(多跳推理详情见上方)"
                else:
                    final_display_base = f"### 联网搜索结果\n{final_search_res}\n\n### 知识库: {kb_name}\n### 检索状态\n{{status}}\n\n### 检索到的内容\n{local_chunks_info}"

                if need_merge:
                    # 4.1 执行合并
                    status_text = "已获取联网结果，正在与本地知识库进行智能融合..."
                    yield final_display_base.format(status=status_text), current_answer

                    merge_system_prompt = "你是一名半导体专家，请整合网络搜索和本地知识库提供全面的解答。请考虑对话历史。"
                    if use_table_format:
                        merge_system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"

                    merge_user_prompt = f"""
                    {enhanced_question}

                    【网络搜索结果】：{final_search_res}

                    【本地知识库分析】：{current_answer}

                    请根据以上信息和对话历史，提供一个综合的回答。
                    """

                    try:
                        # 使用配置的模型进行合并，不再硬编码 qwen-plus
                        response = client.chat.completions.create(
                            model=Config.llm_model,
                            messages=[
                                {"role": "system", "content": merge_system_prompt},
                                {"role": "user", "content": merge_user_prompt}
                            ]
                        )
                        combined_answer = response.choices[0].message.content.strip()

                        yield final_display_base.format(status="已整合联网和知识库结果"), combined_answer

                    except Exception as e:
                        # 合并失败，回退到显示本地结果
                        yield final_display_base.format(status=f"合并失败，显示本地结果: {e}"), current_answer
                else:
                    # 4.2 不需要合并 (直接更新显示联网结果)
                    status_text = "检索完成" if not multi_hop else "多跳推理完成"
                    yield final_display_base.format(status=status_text), current_answer

    except Exception as e:
        error_msg = f"全局处理失败：{str(e)}\n{traceback.format_exc()}"
        yield f"### 错误信息\n{error_msg}", f"处理您的问题时遇到严重错误：{str(e)}"