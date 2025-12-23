import os
from kb.kb_config import KB_BASE_DIR,DEFAULT_KB,OUTPUT_DIR
from typing import List, Dict, Any, Optional, Tuple
import re
import shutil


# 获取知识库列表
def get_knowledge_bases() -> List[str]:
    """获取所有知识库名称"""
    try:
        if not os.path.exists(KB_BASE_DIR):
            os.makedirs(KB_BASE_DIR, exist_ok=True)

        kb_dirs = [d for d in os.listdir(KB_BASE_DIR)
                   if os.path.isdir(os.path.join(KB_BASE_DIR, d))]

        # 确保默认知识库存在
        if DEFAULT_KB not in kb_dirs:
            os.makedirs(os.path.join(KB_BASE_DIR, DEFAULT_KB), exist_ok=True)
            kb_dirs.append(DEFAULT_KB)

        return sorted(kb_dirs)
    except Exception as e:
        print(f"获取知识库列表失败: {str(e)}")
        return [DEFAULT_KB]


# 创建新知识库
def create_knowledge_base(kb_name: str) -> str:
    """创建新的知识库"""
    try:
        if not kb_name or not kb_name.strip():
            return "错误：知识库名称不能为空"

        # 净化知识库名称，只允许字母、数字、下划线和中文，
        # 把“不是 字母/数字/下划线/中文汉字 的字符”全部替换为 _
        kb_name = re.sub(r'[^\w\u4e00-\u9fff]', '_', kb_name.strip())

        kb_path = os.path.join(KB_BASE_DIR, kb_name)
        if os.path.exists(kb_path):
            return f"知识库 '{kb_name}' 已存在"

        os.makedirs(kb_path, exist_ok=True)
        return f"知识库 '{kb_name}' 创建成功"
    except Exception as e:
        return f"创建知识库失败: {str(e)}"


# 删除知识库
def delete_knowledge_base(kb_name: str) -> str:
    """删除指定的知识库"""
    try:
        if kb_name == DEFAULT_KB:
            return f"无法删除默认知识库 '{DEFAULT_KB}'"

        kb_path = os.path.join(KB_BASE_DIR, kb_name)
        if not os.path.exists(kb_path):
            return f"知识库 '{kb_name}' 不存在"

        # 递归删除 知识库，相当于rm -rf，令人害怕，千万别设成更目录，否则完蛋！！！
        shutil.rmtree(kb_path)
        return f"知识库 '{kb_name}' 已删除"
    except Exception as e:
        return f"删除知识库失败: {str(e)}"


# 获取知识库文件列表
def get_kb_files(kb_name: str) -> List[str]:
    """获取指定知识库中的文件列表"""
    try:
        kb_path = os.path.join(KB_BASE_DIR, kb_name)
        if not os.path.exists(kb_path):
            return []

        # 获取所有文件（排除索引文件和元数据文件）
        files = [f for f in os.listdir(kb_path)
                 if os.path.isfile(os.path.join(kb_path, f)) and
                 not f.endswith(('.index', '.json'))]

        return sorted(files)
    except Exception as e:
        print(f"获取知识库文件列表失败: {str(e)}")
        return []