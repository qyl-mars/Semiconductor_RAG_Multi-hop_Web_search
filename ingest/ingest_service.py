from typing import List
import os
from config.configs import Config
from ingest.pdf_loader import extract_text_from_pdf
import chardet  # 用于自动检测编码
from ingest.text_cleaner import clean_text
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from ingest.chunker import semantic_chunk
from ingest.vectorizer import vectorize_file
from rag.indexer import build_faiss_index
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil


# 创建知识库根目录和临时文件目录
KB_BASE_DIR = Config.kb_base_dir
os.makedirs(KB_BASE_DIR, exist_ok=True)

# 创建默认知识库目录
DEFAULT_KB = Config.default_kb
DEFAULT_KB_DIR = os.path.join(KB_BASE_DIR, DEFAULT_KB)
os.makedirs(DEFAULT_KB_DIR, exist_ok=True)

# 创建临时输出目录
OUTPUT_DIR = Config.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 处理单个文件
def process_single_file(file_path: str) -> str:
    try:
        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
            if not text:
                return f"PDF文件 {file_path} 内容为空或无法提取"
        else:
            # "rb"：代表以二进制（Binary）模式读取。
            with open(file_path, "rb") as f:
                content = f.read()
            # chardet 是一个著名的字符集检测库。
            #  核心动作：它会对 content 里的字节进行统计分析（比如根据某些字节出现的频率特征），并给出一个最可能的编码方案。
            result = chardet.detect(content)
            detected_encoding = result['encoding']
            confidence = result['confidence']

            # 尝试多种编码方式
            if detected_encoding and confidence > 0.7:
                try:
                    text = content.decode(detected_encoding)
                    print(f"文件 {file_path} 使用检测到的编码 {detected_encoding} 解码成功")
                except UnicodeDecodeError:
                    text = content.decode('utf-8', errors='ignore')
                    print(f"文件 {file_path} 使用 {detected_encoding} 解码失败，强制使用 UTF-8 忽略非法字符")
            else:
                # 尝试多种常见编码
                encodings = ['utf-8', 'gbk', 'gb18030', 'gb2312', 'latin-1', 'utf-16', 'cp936', 'big5']
                text = None
                for encoding in encodings:
                    try:
                        text = content.decode(encoding)
                        print(f"文件 {file_path} 使用 {encoding} 解码成功")
                        break
                    except UnicodeDecodeError:
                        continue

                # 如果所有编码都失败，使用忽略错误的方式解码
                if text is None:
                    text = content.decode('utf-8', errors='ignore')
                    print(f"警告：文件 {file_path} 强制使用 UTF-8，已经忽略非法字符")

        # 确保文本是干净的，移除非法字符
        text = clean_text(text)
        return text
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        traceback.print_exc()
        return f"处理文件 {file_path} 失败：{str(e)}"


# 批量处理并索引文件 - 修改为支持指定知识库
def process_and_index_files(file_objs: List, kb_name: str = DEFAULT_KB) -> str:
    """处理并索引文件到指定的知识库"""
    # 确保知识库目录存在
    kb_dir = os.path.join(KB_BASE_DIR, kb_name)
    os.makedirs(kb_dir, exist_ok=True)

    # 设置临时处理文件路径
    semantic_chunk_output = os.path.join(OUTPUT_DIR, "semantic_chunk_output.json")
    semantic_chunk_vector = os.path.join(OUTPUT_DIR, "semantic_chunk_vector.json")

    # 设置知识库索引文件路径
    semantic_chunk_index = os.path.join(kb_dir, "semantic_chunk.index")
    semantic_chunk_metadata = os.path.join(kb_dir, "semantic_chunk_metadata.json")

    all_chunks = []
    error_messages = []
    try:
        if not file_objs or len(file_objs) == 0:
            return "错误：没有选择任何文件"

        print(f"开始处理 {len(file_objs)} 个文件，目标知识库: {kb_name}...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(process_single_file, file_obj.name): file_obj for file_obj in file_objs}
            for future in as_completed(future_to_file):
                result = future.result()
                file_obj = future_to_file[future]
                file_name = file_obj.name

                if isinstance(result, str) and result.startswith("处理文件"):
                    error_messages.append(result)
                    print(result)
                    continue

                # 检查结果是否为有效文本
                if not result or not isinstance(result, str) or len(result.strip()) == 0:
                    error_messages.append(f"文件 {file_name} 处理后内容为空")
                    print(f"警告: 文件 {file_name} 处理后内容为空")
                    continue

                print(f"对文件 {file_name} 进行语义分块...")
                chunks = semantic_chunk(result)

                if not chunks or len(chunks) == 0:
                    error_messages.append(f"文件 {file_name} 无法生成任何分块")
                    print(f"警告: 文件 {file_name} 无法生成任何分块")
                    continue

                # 将处理后的文件保存到知识库目录
                file_basename = os.path.basename(file_name)
                dest_file_path = os.path.join(kb_dir, file_basename)
                try:
                    shutil.copy2(file_name, dest_file_path)
                    print(f"已将文件 {file_basename} 复制到知识库 {kb_name}")
                except Exception as e:
                    print(f"复制文件到知识库失败: {str(e)}")

                all_chunks.extend(chunks)
                print(f"文件 {file_name} 处理完成，生成 {len(chunks)} 个分块")

        if not all_chunks:
            return "所有文件处理失败或内容为空\n" + "\n".join(error_messages)

        # 确保分块内容干净且长度合适
        valid_chunks = []
        for chunk in all_chunks:
            # 深度清理文本
            clean_chunk_text = clean_text(chunk["chunk"])

            # 检查清理后的文本是否有效
            if clean_chunk_text and 1 <= len(clean_chunk_text) <= 8000:
                chunk["chunk"] = clean_chunk_text
                valid_chunks.append(chunk)
            elif len(clean_chunk_text) > 8000:
                # 如果文本太长，截断它
                chunk["chunk"] = clean_chunk_text[:8000]
                valid_chunks.append(chunk)
                print(f"警告: 分块 {chunk['id']} 过长已被截断")
            else:
                print(f"警告: 跳过无效分块 {chunk['id']}")

        if not valid_chunks:
            return "所有生成的分块内容无效或为空\n" + "\n".join(error_messages)

        print(f"处理了 {len(all_chunks)} 个分块，有效分块数: {len(valid_chunks)}")

        # 保存语义分块
        with open(semantic_chunk_output, 'w', encoding='utf-8') as json_file:
            json.dump(valid_chunks, json_file, ensure_ascii=False, indent=4)
        print(f"语义分块完成: {semantic_chunk_output}")

        # 向量化语义分块
        print(f"开始向量化 {len(valid_chunks)} 个分块...")
        vectorize_file(valid_chunks, semantic_chunk_vector)
        print(f"语义分块向量化完成: {semantic_chunk_vector}")

        # 验证向量文件是否有效
        try:
            with open(semantic_chunk_vector, 'r', encoding='utf-8') as f:
                vector_data = json.load(f)

            if not vector_data or len(vector_data) == 0:
                return f"向量化失败: 生成的向量文件为空\n" + "\n".join(error_messages)

            # 检查向量数据结构
            if 'vector' not in vector_data[0]:
                return f"向量化失败: 数据中缺少向量字段\n" + "\n".join(error_messages)

            print(f"成功生成 {len(vector_data)} 个向量")
        except Exception as e:
            return f"读取向量文件失败: {str(e)}\n" + "\n".join(error_messages)

        # 构建索引
        print(f"开始为知识库 {kb_name} 构建索引...")
        build_faiss_index(semantic_chunk_vector, semantic_chunk_index, semantic_chunk_metadata)
        print(f"知识库 {kb_name} 索引构建完成: {semantic_chunk_index}")

        status = f"知识库 {kb_name} 更新成功！共处理 {len(valid_chunks)} 个有效分块。\n"
        if error_messages:
            status += "以下文件处理过程中出现问题：\n" + "\n".join(error_messages)
        return status
    except Exception as e:
        error = f"知识库 {kb_name} 索引构建过程中出错：{str(e)}"
        print(error)
        traceback.print_exc()
        return error + "\n" + "\n".join(error_messages)


# 添加处理函数，批量上传文件到指定知识库
def batch_upload_to_kb(file_objs: List, kb_name: str) -> str:
    """批量上传文件到指定知识库并进行处理"""
    try:
        if not kb_name or not kb_name.strip():
            return "错误：未指定知识库"

        # 确保知识库目录存在
        kb_dir = os.path.join(KB_BASE_DIR, kb_name)
        if not os.path.exists(kb_dir):
            os.makedirs(kb_dir, exist_ok=True)

        if not file_objs or len(file_objs) == 0:
            return "错误：未选择任何文件"

        return process_and_index_files(file_objs, kb_name)
    except Exception as e:
        return f"上传文件到知识库失败: {str(e)}"

