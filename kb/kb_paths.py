from typing import Dict
import os
from kb.kb_config import KB_BASE_DIR,DEFAULT_KB,OUTPUT_DIR

# 基于选定知识库生成索引路径
def get_kb_paths(kb_name: str) -> Dict[str, str]:
    """获取指定知识库的索引文件路径"""
    kb_dir = os.path.join(KB_BASE_DIR, kb_name)
    return {
        "index_path": os.path.join(kb_dir, "semantic_chunk.index"),
        "metadata_path": os.path.join(kb_dir, "semantic_chunk_metadata.json")
    }
