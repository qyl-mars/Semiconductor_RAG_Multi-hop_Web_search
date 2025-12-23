from typing import List
import os
import shutil
import json
import traceback
import chardet
from concurrent.futures import ThreadPoolExecutor, as_completed

# 仅引入 LlamaIndex 用于替换底层的分块和向量化计算
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config.configs import Config
from ingest.pdf_loader import extract_text_from_pdf
from ingest.text_cleaner import clean_text

# 路径保持不变
KB_BASE_DIR = Config.kb_base_dir
DEFAULT_KB = Config.default_kb
OUTPUT_DIR = Config.output_dir

# 初始化 Embedding 模型 (用于语义分块和向量化)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")


def process_single_file(file_path: str) -> str:
    """完全保持你原有的解码逻辑，仅确保护回干净文本"""
    try:
        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
            if not text: return f"PDF文件 {file_path} 内容为空"
        else:
            with open(file_path, "rb") as f:
                content = f.read()
            result = chardet.detect(content)
            detected_encoding = result['encoding']
            confidence = result['confidence']

            if detected_encoding and confidence > 0.7:
                try:
                    text = content.decode(detected_encoding)
                except UnicodeDecodeError:
                    text = content.decode('utf-8', errors='ignore')
            else:
                encodings = ['utf-8', 'gbk', 'gb18030', 'gb2312', 'latin-1', 'utf-16', 'cp936', 'big5']
                text = None
                for encoding in encodings:
                    try:
                        text = content.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                if text is None:
                    text = content.decode('utf-8', errors='ignore')

        # 依然调用你的文本清洗函数
        text = clean_text(text)
        return text
    except Exception as e:
        traceback.print_exc()
        return f"处理文件 {file_path} 失败：{str(e)}"


def process_and_index_files(file_objs: List, kb_name: str = DEFAULT_KB) -> str:
    """保持原有结构，仅优化分块和索引保存方式"""
    kb_dir = os.path.join(KB_BASE_DIR, kb_name)
    os.makedirs(kb_dir, exist_ok=True)

    # 路径定义与原版一致
    semantic_chunk_output = os.path.join(OUTPUT_DIR, "semantic_chunk_output.json")

    all_docs = []  # 存储 LlamaIndex Document 对象
    error_messages = []

    try:
        if not file_objs: return "错误：没有选择任何文件"

        print(f"开始处理 {len(file_objs)} 个文件...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(process_single_file, f.name): f for f in file_objs}
            for future in as_completed(future_to_file):
                result = future.result()
                file_obj = future_to_file[future]

                if result.startswith("处理文件") or result.startswith("处理失败"):
                    error_messages.append(result)
                    continue

                if not result.strip(): continue

                # 复制文件到知识库 (保留原有逻辑)
                dest_file_path = os.path.join(kb_dir, os.path.basename(file_obj.name))
                shutil.copy2(file_obj.name, dest_file_path)

                # 将文本包装为 Document 供 LlamaIndex 使用
                all_docs.append(Document(text=result, metadata={"file_name": file_obj.name}))

        if not all_docs:
            return "处理失败\n" + "\n".join(error_messages)

        # --- 优化点：使用语义分块器 ---
        splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95,
                                              embed_model=embed_model)

        # --- 优化点：统一索引管理，替代原来的三个 JSON 维护 ---
        index_path = os.path.join(kb_dir, "llama_index_storage")
        if os.path.exists(index_path):
            storage_context = StorageContext.from_defaults(persist_dir=index_path)
            index = load_index_from_storage(storage_context, embed_model=embed_model)
            for doc in all_docs:
                index.insert(doc)
        else:
            index = VectorStoreIndex.from_documents(all_docs, transformations=[splitter], embed_model=embed_model)

        # 持久化索引
        index.storage_context.persist(persist_dir=index_path)

        status = f"知识库 {kb_name} 更新成功！共处理 {len(all_docs)} 个文件。\n"
        if error_messages: status += "异常：\n" + "\n".join(error_messages)
        return status

    except Exception as e:
        traceback.print_exc()
        return f"构建索引出错：{str(e)}"


def batch_upload_to_kb(file_objs: List, kb_name: str) -> str:
    """完全保留接口函数"""
    return process_and_index_files(file_objs, kb_name)