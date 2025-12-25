from config.configs import Config
import json
import faiss
import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple
from llm.embedding_client import vectorize_query
from llm.llm_client import client
import traceback


# 多跳推理RAG系统 - 核心创新点
class ReasoningRAG:
    """
    多跳推理RAG系统，通过迭代式的检索和推理过程回答问题，支持流式响应
    """

    def __init__(self,
                 index_path: str,
                 metadata_path: str,
                 max_hops: int = 3,
                 initial_candidates: int = 5,
                 refined_candidates: int = 3,
                 reasoning_model: str = Config.llm_model,
                 verbose: bool = False):
        """
        初始化推理RAG系统

        参数:
            index_path: FAISS索引的路径
            metadata_path: 元数据JSON文件的路径
            max_hops: 最大推理-检索跳数
            initial_candidates: 初始检索候选数量
            refined_candidates: 精炼检索候选数量
            reasoning_model: 用于推理步骤的LLM模型
            verbose: 是否打印详细日志
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.max_hops = max_hops
        self.initial_candidates = initial_candidates
        self.refined_candidates = refined_candidates
        self.reasoning_model = reasoning_model
        self.verbose = verbose

        # 加载索引和元数据
        self._load_resources()

    def _load_resources(self):
        """加载FAISS索引和元数据"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except UnicodeDecodeError:
                with open(self.metadata_path, 'rb') as f:
                    content = f.read().decode('utf-8', errors='ignore')
                    self.metadata = json.loads(content)
        else:
            raise FileNotFoundError(f"Index or metadata not found at {self.index_path} or {self.metadata_path}")

    def _vectorize_query(self, query: str) -> np.ndarray:
        """将查询转换为向量"""
        return vectorize_query(query).reshape(1, -1)

    def _retrieve(self, query_vector: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        """使用向量相似性检索块"""
        if query_vector.size == 0:
            return []

        D, I = self.index.search(query_vector, limit)
        results = [self.metadata[i] for i in I[0] if i < len(self.metadata)]
        return results

    def _generate_reasoning(self,
                            query: str,
                            retrieved_chunks: List[Dict[str, Any]],
                            previous_queries: List[str] = None,
                            hop_number: int = 0) -> Dict[str, Any]:
        """
        为检索到的信息生成推理分析并识别信息缺口

        返回包含以下字段的字典:
            - analysis: 对当前信息的推理分析
            - missing_info: 已识别的缺失信息
            - follow_up_queries: 填补信息缺口的后续查询列表
            - is_sufficient: 表示信息是否足够的布尔值
        """
        if previous_queries is None:
            previous_queries = []

        # 为模型准备上下文
        chunks_text = "\n\n".join([f"[Chunk {i + 1}]: {chunk['chunk']}"
                                   for i, chunk in enumerate(retrieved_chunks)])

        previous_queries_text = "\n".join([f"Q{i + 1}: {q}" for i, q in enumerate(previous_queries)])

        system_prompt = """
        你是半导体信息检索的专家分析系统。
        你的任务是分析检索到的信息块，识别缺失的内容，并提出有针对性的后续查询来填补信息缺口。

        重点关注半导体领域知识，如:
        - 芯片设计和制造
        - 半导体材料和工艺
        - 集成电路和系统
        - 电子器件和电路
        - 行业标准和技术规范
        """

        user_prompt = f"""
        ## 原始查询
        {query}

        ## 先前查询（如果有）
        {previous_queries_text if previous_queries else "无"}

        ## 检索到的信息（跳数 {hop_number}）
        {chunks_text if chunks_text else "未检索到信息。"}

        ## 你的任务
        1. 分析已检索到的信息与原始查询的关系
        2. 确定能够更完整回答查询的特定缺失信息
        3. 提出1-3个针对性的后续查询，以检索缺失信息
        4. 确定当前信息是否足够回答原始查询

        以JSON格式回答，包含以下字段:
        - analysis: 对当前信息的详细分析
        - missing_info: 特定缺失信息的列表
        - follow_up_queries: 1-3个具体的后续查询
        - is_sufficient: 表示信息是否足够的布尔值
        """

        try:
            response = client.chat.completions.create(
                model=Config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            reasoning_text = response.choices[0].message.content.strip()

            # 解析JSON响应
            try:
                reasoning = json.loads(reasoning_text)
                # 确保预期的键存在
                required_keys = ["analysis", "missing_info", "follow_up_queries", "is_sufficient"]
                for key in required_keys:
                    if key not in reasoning:
                        reasoning[key] = [] if key != "is_sufficient" else False
                return reasoning
            except json.JSONDecodeError:
                # 如果JSON解析失败，则回退
                if self.verbose:
                    print(f"无法从模型输出解析JSON: {reasoning_text[:100]}...")
                return {
                    "analysis": "无法分析检索到的信息。",
                    "missing_info": ["无法识别缺失信息"],
                    "follow_up_queries": [],
                    "is_sufficient": False
                }

        except Exception as e:
            if self.verbose:
                print(f"推理生成错误: {e}")
                print(traceback.format_exc())
            return {
                "analysis": "分析过程出错。",
                "missing_info": [],
                "follow_up_queries": [],
                "is_sufficient": False
            }

    def _synthesize_answer(self,
                           query: str,
                           all_chunks: List[Dict[str, Any]],
                           reasoning_steps: List[Dict[str, Any]],
                           use_table_format: bool = False) -> str:
        """从所有检索到的块和推理步骤中合成最终答案"""
        # 合并所有块，去除重复
        unique_chunks = []
        chunk_ids = set()
        for chunk in all_chunks:
            if chunk["id"] not in chunk_ids:
                unique_chunks.append(chunk)
                chunk_ids.add(chunk["id"])

        # 准备上下文
        chunks_text = "\n\n".join([f"[Chunk {i + 1}]: {chunk['chunk']}"
                                   for i, chunk in enumerate(unique_chunks)])

        # 准备推理跟踪
        reasoning_trace = ""
        for i, step in enumerate(reasoning_steps):
            reasoning_trace += f"\n\n推理步骤 {i + 1}:\n"
            reasoning_trace += f"分析: {step['analysis']}\n"
            reasoning_trace += f"缺失信息: {', '.join(step['missing_info'])}\n"
            reasoning_trace += f"后续查询: {', '.join(step['follow_up_queries'])}"

        system_prompt = """
        你是半导体领域的专家。基于检索到的信息块，为用户的查询合成一个全面的答案。

        重点提供有关半导体技术的准确信息，包括芯片设计、制造工艺、材料科学等方面

        逻辑地组织你的答案，并在适当时引用块中的具体信息。如果信息不完整，请承认限制。
        """

        output_format_instruction = ""
        if use_table_format:
            output_format_instruction = """
            请尽可能以Markdown表格格式组织你的回答。如果信息适合表格形式展示，请使用表格；
            如果不适合表格形式，可以先用文本介绍，然后再使用表格总结关键信息。

            表格语法示例：
            | 标题1 | 标题2 | 标题3 |
            | ----- | ----- | ----- |
            | 内容1 | 内容2 | 内容3 |

            确保表格格式符合Markdown标准，以便正确渲染。
            """

        user_prompt = f"""
        ## 原始查询
        {query}

        ## 检索到的信息块
        {chunks_text}

        ## 推理过程
        {reasoning_trace}

        ## 你的任务
        使用提供的信息块为原始查询合成一个全面的答案。你的答案应该:

        1. 直接回应查询
        2. 结构清晰，易于理解
        3. 基于检索到的信息
        4. 承认可用信息中的任何重大缺口

        {output_format_instruction}

        以直接回应提出原始查询的用户的方式呈现你的答案。
        """

        try:
            response = client.chat.completions.create(
                model=Config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if self.verbose:
                print(f"答案合成错误: {e}")
                print(traceback.format_exc())
            return "由于出错，无法生成答案。"

    def stream_retrieve_and_answer(self, query: str, use_table_format: bool = False):
        """
        执行多跳检索和回答生成的流式方法，逐步返回结果

        这是一个生成器函数，会在处理的每个阶段产生中间结果
        """
        all_chunks = []
        all_queries = [query]
        reasoning_steps = []

        # 生成状态更新
        yield {
            "status": "正在将查询向量化...",
            "reasoning_display": "",
            "answer": None,
            "all_chunks": [],
            "reasoning_steps": []
        }

        # 初始检索
        try:
            query_vector = self._vectorize_query(query)
            if query_vector.size == 0:
                yield {
                    "status": "向量化失败",
                    "reasoning_display": "由于嵌入错误，无法处理查询。",
                    "answer": "由于嵌入错误，无法处理查询。",
                    "all_chunks": [],
                    "reasoning_steps": []
                }
                return

            yield {
                "status": "正在执行初始检索...",
                "reasoning_display": "",
                "answer": None,
                "all_chunks": [],
                "reasoning_steps": []
            }

            initial_chunks = self._retrieve(query_vector, self.initial_candidates)
            all_chunks.extend(initial_chunks)

            if not initial_chunks:
                yield {
                    "status": "未找到相关信息",
                    "reasoning_display": "未找到与您的查询相关的信息。",
                    "answer": "未找到与您的查询相关的信息。",
                    "all_chunks": [],
                    "reasoning_steps": []
                }
                return

            # 更新状态，展示找到的初始块
            chunks_preview = "\n".join([f"- {chunk['chunk'][:100]}..." for chunk in initial_chunks[:2]])
            yield {
                "status": f"找到 {len(initial_chunks)} 个相关信息块，正在生成初步分析...",
                "reasoning_display": f"### 检索到的初始信息\n{chunks_preview}\n\n### 正在分析...",
                "answer": None,
                "all_chunks": all_chunks,
                "reasoning_steps": []
            }

            # 初始推理
            reasoning = self._generate_reasoning(query, initial_chunks, hop_number=0)
            reasoning_steps.append(reasoning)

            # 生成当前的推理显示
            reasoning_display = "### 多跳推理过程\n"
            reasoning_display += f"**推理步骤 1**\n"
            reasoning_display += f"- 分析: {reasoning['analysis'][:200]}...\n"
            reasoning_display += f"- 缺失信息: {', '.join(reasoning['missing_info'])}\n"
            if reasoning['follow_up_queries']:
                reasoning_display += f"- 后续查询: {', '.join(reasoning['follow_up_queries'])}\n"
            reasoning_display += f"- 信息是否足够: {'是' if reasoning['is_sufficient'] else '否'}\n\n"

            yield {
                "status": "初步分析完成",
                "reasoning_display": reasoning_display,
                "answer": None,
                "all_chunks": all_chunks,
                "reasoning_steps": reasoning_steps
            }

            # 检查是否需要额外的跳数
            hop = 1
            while (hop < self.max_hops and
                   not reasoning["is_sufficient"] and
                   reasoning["follow_up_queries"]):

                follow_up_status = f"执行跳数 {hop}，正在处理 {len(reasoning['follow_up_queries'])} 个后续查询..."
                yield {
                    "status": follow_up_status,
                    "reasoning_display": reasoning_display + f"\n\n### {follow_up_status}",
                    "answer": None,
                    "all_chunks": all_chunks,
                    "reasoning_steps": reasoning_steps
                }

                hop_chunks = []

                # 处理每个后续查询
                for i, follow_up_query in enumerate(reasoning["follow_up_queries"]):
                    all_queries.append(follow_up_query)

                    query_status = f"处理后续查询 {i + 1}/{len(reasoning['follow_up_queries'])}: {follow_up_query}"
                    yield {
                        "status": query_status,
                        "reasoning_display": reasoning_display + f"\n\n### {query_status}",
                        "answer": None,
                        "all_chunks": all_chunks,
                        "reasoning_steps": reasoning_steps
                    }

                    # 为后续查询检索
                    follow_up_vector = self._vectorize_query(follow_up_query)
                    if follow_up_vector.size > 0:
                        follow_up_chunks = self._retrieve(follow_up_vector, self.refined_candidates)
                        hop_chunks.extend(follow_up_chunks)
                        all_chunks.extend(follow_up_chunks)

                        # 更新状态，显示新找到的块数量
                        yield {
                            "status": f"查询 '{follow_up_query}' 找到了 {len(follow_up_chunks)} 个相关块",
                            "reasoning_display": reasoning_display + f"\n\n为查询 '{follow_up_query}' 找到了 {len(follow_up_chunks)} 个相关块",
                            "answer": None,
                            "all_chunks": all_chunks,
                            "reasoning_steps": reasoning_steps
                        }

                # 为此跳数生成推理
                yield {
                    "status": f"正在为跳数 {hop} 生成推理分析...",
                    "reasoning_display": reasoning_display + f"\n\n### 正在为跳数 {hop} 生成推理分析...",
                    "answer": None,
                    "all_chunks": all_chunks,
                    "reasoning_steps": reasoning_steps
                }

                reasoning = self._generate_reasoning(
                    query,
                    hop_chunks,
                    previous_queries=all_queries[:-1],
                    hop_number=hop
                )
                reasoning_steps.append(reasoning)

                # 更新推理显示
                reasoning_display += f"\n**推理步骤 {hop + 1}**\n"
                reasoning_display += f"- 分析: {reasoning['analysis'][:200]}...\n"
                reasoning_display += f"- 缺失信息: {', '.join(reasoning['missing_info'])}\n"
                if reasoning['follow_up_queries']:
                    reasoning_display += f"- 后续查询: {', '.join(reasoning['follow_up_queries'])}\n"
                reasoning_display += f"- 信息是否足够: {'是' if reasoning['is_sufficient'] else '否'}\n"

                yield {
                    "status": f"跳数 {hop} 完成",
                    "reasoning_display": reasoning_display,
                    "answer": None,
                    "all_chunks": all_chunks,
                    "reasoning_steps": reasoning_steps
                }

                hop += 1

            # 合成最终答案
            yield {
                "status": "正在合成最终答案...",
                "reasoning_display": reasoning_display + "\n\n### 正在合成最终答案...",
                "answer": "正在处理您的问题，请稍候...",
                "all_chunks": all_chunks,
                "reasoning_steps": reasoning_steps
            }

            answer = self._synthesize_answer(query, all_chunks, reasoning_steps, use_table_format)

            # 为最终显示准备检索内容汇总
            all_chunks_summary = "\n\n".join([f"**检索块 {i + 1}**:\n{chunk['chunk']}"
                                              for i, chunk in enumerate(all_chunks[:10])])  # 限制显示前10个块

            if len(all_chunks) > 10:
                all_chunks_summary += f"\n\n...以及另外 {len(all_chunks) - 10} 个块（总计 {len(all_chunks)} 个）"

            enhanced_display = reasoning_display + "\n\n### 检索到的内容\n" + all_chunks_summary + "\n\n### 回答已生成"

            yield {
                "status": "回答已生成",
                "reasoning_display": enhanced_display,
                "answer": answer,
                "all_chunks": all_chunks,
                "reasoning_steps": reasoning_steps
            }

        except Exception as e:
            error_msg = f"处理过程中出错: {str(e)}"
            if self.verbose:
                print(error_msg)
                print(traceback.format_exc())

            yield {
                "status": "处理出错",
                "reasoning_display": error_msg,
                "answer": f"处理您的问题时遇到错误: {str(e)}",
                "all_chunks": all_chunks,
                "reasoning_steps": reasoning_steps
            }

    def retrieve_and_answer(self, query: str, use_table_format: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        执行多跳检索和回答生成的主要方法

        返回:
            包含以下内容的元组:
            - 最终答案
            - 包含推理步骤和所有检索到的块的调试字典
        """
        all_chunks = []
        all_queries = [query]
        reasoning_steps = []
        debug_info = {"reasoning_steps": [], "all_chunks": [], "all_queries": all_queries}

        # 初始检索
        query_vector = self._vectorize_query(query)
        if query_vector.size == 0:
            return "由于嵌入错误，无法处理查询。", debug_info

        initial_chunks = self._retrieve(query_vector, self.initial_candidates)
        all_chunks.extend(initial_chunks)
        debug_info["all_chunks"].extend(initial_chunks)

        if not initial_chunks:
            return "未找到与您的查询相关的信息。", debug_info

        # 初始推理
        reasoning = self._generate_reasoning(query, initial_chunks, hop_number=0)
        reasoning_steps.append(reasoning)
        debug_info["reasoning_steps"].append(reasoning)

        # 检查是否需要额外的跳数
        hop = 1
        while (hop < self.max_hops and
               not reasoning["is_sufficient"] and
               reasoning["follow_up_queries"]):

            if self.verbose:
                print(f"开始跳数 {hop}，有 {len(reasoning['follow_up_queries'])} 个后续查询")

            hop_chunks = []

            # 处理每个后续查询
            for follow_up_query in reasoning["follow_up_queries"]:
                all_queries.append(follow_up_query)
                debug_info["all_queries"].append(follow_up_query)

                # 为后续查询检索
                follow_up_vector = self._vectorize_query(follow_up_query)
                if follow_up_vector.size > 0:
                    follow_up_chunks = self._retrieve(follow_up_vector, self.refined_candidates)
                    hop_chunks.extend(follow_up_chunks)
                    all_chunks.extend(follow_up_chunks)
                    debug_info["all_chunks"].extend(follow_up_chunks)

            # 为此跳数生成推理
            reasoning = self._generate_reasoning(
                query,
                hop_chunks,
                previous_queries=all_queries[:-1],
                hop_number=hop
            )
            reasoning_steps.append(reasoning)
            debug_info["reasoning_steps"].append(reasoning)

            hop += 1

        # 合成最终答案
        answer = self._synthesize_answer(query, all_chunks, reasoning_steps, use_table_format)

        return answer, debug_info
