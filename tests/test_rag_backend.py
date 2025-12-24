"""
独立测试脚本：用于绕过前端 web_ui 直接测试后端 RAG 全流程

功能：
1. 自动获取当前已有的知识库列表，并默认使用 DEFAULT_KB
2. 模拟上传 test_docs/ 文件夹下的所有文件并完成向量化
3. 模拟一次完整的问答流

使用方法:
    python tests/test_rag_backend.py [--kb KB_NAME] [--skip-upload] [--skip-parse] [--skip-vectorize] [--skip-qa]
"""

import os
import sys
import argparse
import shutil
from typing import List

# 确保可以从项目根目录导入模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kb.kb_manager import get_knowledge_bases
from kb.kb_config import KB_BASE_DIR, DEFAULT_KB, OUTPUT_DIR
from ingest.ingest_service import process_and_index_files, process_single_file
from ingest.chunker import semantic_chunk
from ingest.vectorizer import vectorize_file
from rag.indexer import build_faiss_index
from web_ui.streaming_handler import process_question_with_reasoning

# 测试开关（通过修改这里的 True/False 来控制是否执行各个测试）
RUN_TEST_1_UPLOAD = False            # 测试1：仅上传文件到知识库目录
RUN_TEST_2_PARSE = False             # 测试2：仅解析文件内容
RUN_TEST_3_VECTORIZE = False         # 测试3：分块 + 向量化 + 建索引
RUN_TEST_4_QA = True               # 测试4：问答流程（基于已建索引）


class FileObj:
    """简单的文件对象包装类，用于模拟上传的文件"""
    def __init__(self, file_path: str):
        self.name = file_path
        self.path = file_path


def get_test_files(test_docs_dir: str = "test_docs") -> List[FileObj]:
    """
    获取测试文件夹下的所有文件
    
    Args:
        test_docs_dir: 测试文档文件夹路径
        
    Returns:
        文件对象列表
    """
    if not os.path.exists(test_docs_dir):
        print(f"警告: 测试文件夹 '{test_docs_dir}' 不存在，将创建该文件夹")
        os.makedirs(test_docs_dir, exist_ok=True)
        return []
    
    files = []
    for file_name in os.listdir(test_docs_dir):
        file_path = os.path.join(test_docs_dir, file_name)
        if os.path.isfile(file_path):
            files.append(FileObj(file_path))
            print(f"找到测试文件: {file_name}")
    
    return files


def test_knowledge_base_setup(kb_name: str) -> str:
    """辅助函数：确保指定知识库存在，并打印当前知识库列表

    如果指定的 kb_name 不存在，则创建该目录；
    同时始终确保默认知识库存在。
    """
    print("\n" + "="*60)
    print("知识库环境检查")
    print("="*60)

    kb_list = get_knowledge_bases()
    print(f"当前已有知识库: {kb_list}")

    if kb_name not in kb_list:
        target_dir = os.path.join(KB_BASE_DIR, kb_name)
        print(f"指定的知识库 '{kb_name}' 不在现有列表中，将在 {target_dir} 下创建该知识库目录。")
        os.makedirs(target_dir, exist_ok=True)
        kb_list = get_knowledge_bases()
        print(f"创建后知识库列表: {kb_list}")

    print(f"本次测试将使用知识库: {kb_name}")
    return kb_name


def test_upload_files(kb_name: str = DEFAULT_KB):
    """测试1：仅上传文件到指定知识库目录，不做解析和向量化"""
    print("\n" + "="*60)
    print("测试1：上传文件到知识库目录（只复制文件）")
    print("="*60)

    test_files = get_test_files("test_docs")

    if not test_files:
        print("错误: test_docs/ 文件夹中没有找到任何文件")
        print("请将测试文件放入 test_docs/ 文件夹中")
        return False

    kb_dir = os.path.join(KB_BASE_DIR, kb_name)
    os.makedirs(kb_dir, exist_ok=True)
    print(f"目标知识库目录: {kb_dir}")

    for f in test_files:
        src = f.name
        dst = os.path.join(kb_dir, os.path.basename(f.name))
        try:
            shutil.copy2(src, dst)
            print(f"已复制: {src} -> {dst}")
        except Exception as e:
            print(f"复制文件失败: {src} -> {dst}, 错误: {e}")

    print("文件上传（复制）测试完成。")
    return True


def test_parse_files():
    """测试2：仅解析文件内容，查看处理后的纯文本"""
    print("\n" + "="*60)
    print("测试2：解析文件内容（不分块、不向量化）")
    print("="*60)

    test_files = get_test_files("test_docs")

    if not test_files:
        print("提示: test_docs/ 中没有找到文件，无法演示解析。")
        print("请在 test_docs/ 目录下放入一个 PDF 或文本文件后再试。")
        return False

    for idx, file_obj in enumerate(test_files, start=1):
        file_path = file_obj.name
        print(f"\n--- 文件 {idx}: {file_path} ---")
        try:
            parsed_text = process_single_file(file_path)
        except Exception as e:
            print(f"解析文件时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

        if isinstance(parsed_text, str) and parsed_text.startswith("处理文件"):
            print(f"解析失败，返回信息: {parsed_text}")
            continue

        if not parsed_text or not isinstance(parsed_text, str):
            print("解析结果为空或类型异常")
            continue

        print(f"解析成功，文本总长度: {len(parsed_text)} 字符")
        preview_len = min(300, len(parsed_text))
        print(f"\n【解析文本前 {preview_len} 个字符预览】\n")
        print(parsed_text[:preview_len])

    print("\n解析文件测试完成。")
    return True


def test_vectorize_and_index(kb_name: str = DEFAULT_KB):
    """测试3：对解析后的文本进行分块、向量化并构建索引"""
    print("\n" + "="*60)
    print("测试3：分块 + 向量化 + 构建索引")
    print("="*60)

    test_files = get_test_files("test_docs")

    if not test_files:
        print("错误: test_docs/ 文件夹中没有找到任何文件")
        print("请将测试文件放入 test_docs/ 文件夹中")
        return False

    all_chunks = []
    for idx, file_obj in enumerate(test_files, start=1):
        file_path = file_obj.name
        print(f"\n--- 处理文件 {idx}: {file_path} ---")
        try:
            text = process_single_file(file_path)
        except Exception as e:
            print(f"解析文件时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

        if not text or not isinstance(text, str):
            print("解析结果为空或类型异常，跳过该文件")
            continue

        print("开始语义分块...")
        chunks = semantic_chunk(text)
        print(f"生成分块数量: {len(chunks)}")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("没有可用的分块，无法继续向量化和建索引")
        return False

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    kb_dir = os.path.join(KB_BASE_DIR, kb_name)
    os.makedirs(kb_dir, exist_ok=True)

    semantic_chunk_output = os.path.join(OUTPUT_DIR, "semantic_chunk_output_test.json")
    semantic_chunk_vector = os.path.join(OUTPUT_DIR, "semantic_chunk_vector_test.json")
    semantic_chunk_index = os.path.join(kb_dir, "semantic_chunk_test.index")
    semantic_chunk_metadata = os.path.join(kb_dir, "semantic_chunk_metadata_test.json")

    # 保存分块结果
    import json
    with open(semantic_chunk_output, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"\n已保存分块结果到: {semantic_chunk_output}")

    # 向量化
    print("\n开始向量化分块...")
    vectorize_file(all_chunks, semantic_chunk_vector)
    print(f"向量化结果已保存到: {semantic_chunk_vector}")

    # 构建索引
    print("\n开始基于向量文件构建索引...")
    build_faiss_index(semantic_chunk_vector, semantic_chunk_index, semantic_chunk_metadata)
    print(f"索引文件: {semantic_chunk_index}")
    print(f"元数据文件: {semantic_chunk_metadata}")

    print("\n向量化和索引构建测试完成。")
    return True


def test_question_answering(kb_name: str = DEFAULT_KB, question: str = None):
    """测试4: 模拟一次完整的问答流（依赖已构建的索引）"""
    print("\n" + "="*60)
    print("测试4: 问答流程测试")
    print("="*60)
    
    # 如果没有提供问题，使用默认问题
    if question is None:
        question = "如何修改密码"
    
    print(f"问题: {question}")
    print(f"使用的知识库: {kb_name}")
    print("\n开始处理问题...\n")
    
    try:
        # 调用 streaming_handler 中的处理函数
        # 注意: process_question_with_reasoning 是一个生成器函数
        for search_display, answer in process_question_with_reasoning(
            question=question,
            kb_name=kb_name,
            use_search=True,  # 测试时关闭联网搜索，专注于本地知识库
            use_table_format=True,
            multi_hop=True,
            chat_history=None
        ):
            # 打印流式输出的状态
            print("=" * 60)
            print("状态更新:")
            print(search_display)
            print("\n当前答案:")
            print(answer)
            print("=" * 60 + "\n")
        
        print("问答流程完成!")
        return True
        
    except Exception as e:
        print(f"问答流程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='RAG 后端全流程测试脚本')
    parser.add_argument('--kb', dest='kb_name', default=DEFAULT_KB,
                       help='要使用的知识库名称（默认: default）')
    parser.add_argument('--skip-upload', action='store_true',
                       help='跳过测试1：上传文件到知识库目录')
    parser.add_argument('--skip-parse', action='store_true',
                       help='跳过测试2：解析文件内容')
    parser.add_argument('--skip-vectorize', action='store_true',
                       help='跳过测试3：分块 + 向量化 + 建索引')
    parser.add_argument('--skip-qa', action='store_true',
                       help='跳过测试4：问答流程（仅在 RUN_TEST_4_QA=True 时生效）')
    parser.add_argument('--question', action='append', dest='questions',
                       help='指定要测试的问题（仅在 RUN_TEST_4_QA=True 时生效，可多次使用）')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("RAG 后端全流程测试")
    print("="*60)
    
    # 确定使用的知识库（来自命令行参数，默认 default）
    kb_name = test_knowledge_base_setup(args.kb_name)

    # 测试1：上传文件
    if RUN_TEST_1_UPLOAD:
        if not args.skip_upload:
            test_upload_files(kb_name)
        else:
            print("\n跳过测试1：上传文件到知识库目录")
    else:
        print("\n当前已关闭测试1（如需开启，请将 RUN_TEST_1_UPLOAD 设为 True）")

    # 测试2：解析文件
    if RUN_TEST_2_PARSE:
        if not args.skip_parse:
            test_parse_files()
        else:
            print("\n跳过测试2：解析文件内容")
    else:
        print("\n当前已关闭测试2（如需开启，请将 RUN_TEST_2_PARSE 设为 True）")

    # 测试3：分块 + 向量化 + 建索引
    if RUN_TEST_3_VECTORIZE:
        if not args.skip_vectorize:
            test_vectorize_and_index(kb_name)
        else:
            print("\n跳过测试3：分块 + 向量化 + 建索引")
    else:
        print("\n当前已关闭测试3（如需开启，请将 RUN_TEST_3_VECTORIZE 设为 True）")

    # 测试4：问答流程
    if RUN_TEST_4_QA and not args.skip_qa:
        # 使用命令行参数中的问题，或使用默认问题
        if args.questions:
            test_questions = args.questions
        else:
            test_questions = [
                "如何修改密码"
            ]
        
        for question in test_questions:
            print(f"kb_name={kb_name}")
            test_question_answering(kb_name, question)
            print("\n" + "-"*60 + "\n")
    elif RUN_TEST_4_QA and args.skip_qa:
        print("\n已开启 RUN_TEST_4_QA，但通过参数 --skip-qa 跳过了问答流程测试")
    else:
        print("\n当前已关闭测试4（如需开启，请将 RUN_TEST_4_QA 设为 True）")
    
    print("\n" + "="*60)
    print("所有已开启的测试执行完毕!")
    print("="*60)


if __name__ == "__main__":
    main()


