import re
import logging
from typing import List, Dict, Any
from llama_index.core.node_parser import SentenceSplitter

# 配置日志，工程必备
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedChunker:
    def __init__(self, chunk_size=800, chunk_overlap=50, min_chunk_len=20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_len = min_chunk_len

        # 优化1：中英兼容的分隔符正则表达式
        # 同时涵盖了中文标点 [。！？；] 和英文标点 [.!?]
        self.sentence_seps = r'([。！？；\n]|[.!?](?=\s|$))'

        # 初始化 LlamaIndex 的基础切分器
        self.base_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=" "  # 基础层交给空格处理
        )

    def _recursive_split(self, text: str) -> List[str]:
        """核心改进：递归切分。如果段落太长，按句切；句还长，按字符切。"""
        if len(text) <= self.chunk_size:
            return [text]

        # 尝试按标点符号切分
        parts = re.split(self.sentence_seps, text)
        res = []
        current = ""

        for part in parts:
            # 如果加上这个部分还没超标，就继续累加
            if len(current) + len(part) <= self.chunk_size:
                current += part
            else:
                if current: res.append(current.strip())
                # 如果单个句子就超过了 chunk_size，暴力按长度切分（保底逻辑）
                if len(part) > self.chunk_size:
                    for i in range(0, len(part), self.chunk_size):
                        res.append(part[i:i + self.chunk_size])
                    current = ""
                else:
                    current = part
        if current:
            res.append(current.strip())
        return res

    def split(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict]:
        """
        落地逻辑：
        1. 按段落切分
        2. 段落内递归细切
        3. 短句自动向前合并，不暴力丢弃
        """
        if not text: return []

        # 预处理：统一换行符
        text = text.replace("\r\n", "\n")
        raw_paragraphs = text.split("\n\n")

        final_chunks = []
        temp_buffer = ""  # 用于合并短句的缓冲区

        for para in raw_paragraphs:
            para = para.strip()
            if not para: continue

            # 对每个段落进行递归切分，确保不溢出
            sub_splits = self._recursive_split(para)

            for s in sub_splits:
                # 优化2：短句合并策略。如果不满 min_chunk_len，先攒着
                if len(temp_buffer) + len(s) < self.min_chunk_len:
                    temp_buffer += (" " + s)
                    continue

                # 缓冲区够长了，合并输出
                complete_text = (temp_buffer + " " + s).strip()
                final_chunks.append(complete_text)
                temp_buffer = ""

        # 处理最后剩下的尾巴
        if temp_buffer:
            if final_chunks:
                final_chunks[-1] += (" " + temp_buffer)
            else:
                final_chunks.append(temp_buffer)

        # 封装为带 ID 和 Metadata 的结构
        return self._format_output(final_chunks, metadata)

    def _format_output(self, chunks: List[str], metadata: Dict[str, Any]) -> List[Dict]:
        output = []
        for i, content in enumerate(chunks):
            chunk_struct = {
                "id": f"chunk_{i:04d}",
                "content": content,
                "metadata": metadata or {},
                "char_len": len(content)
            }
            output.append(chunk_struct)
        return output


# --- 使用示例 ---
if __name__ == "__main__":
    test_text = "这是一个超长段落测试。" * 100 + "\n\n短句。" + "另一段话。"
    chunker = EnhancedChunker(chunk_size=200, min_chunk_len=30)

    # 模拟传入元数据
    meta = {"source": "manual_v1.pdf", "page": 12}
    result = chunker.split(test_text, metadata=meta)

    for c in result:
        print(f"[{c['id']}] (len: {c['char_len']}) -> {c['content'][:50]}...")