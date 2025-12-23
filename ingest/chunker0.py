from typing import List
from llama_index.core.node_parser import SentenceSplitter
import re
from utils.logger_config import setup_logger
from config.configs import Config
"""
这是咕泡的源代码
"""
# 设置日志
logger = setup_logger("chunker.log")

# 分块参数
chunk_size = Config.chunk_size
chunk_filter_len = Config.chunk_filter_len
chunk_overlap = Config.chunk_overlap


# 语义分块函数
def semantic_chunk(text: str, chunk_size=800, chunk_overlap=20) -> List[dict]:
    """
    # 继承并增强llama_index的句子文本分割器SentenceSplitter，但是问题是
    # ❌ 默认更偏英文 ❌ 中文断句不理想 ❌ 分隔符不够灵活
    自定义了分隔符和分句逻辑，从而提高RAG的召回粒度和语义完整性。
    自定义分隔符：它在默认分隔符（通常是句号 。）的基础上，额外增加了中文分号 ；、感叹号 !、问号 ? 和换行符 \n。
    """

    class EnhancedSentenceSplitter(SentenceSplitter):
        def __init__(self, *args, **kwargs):
            custom_seps = ["；", "!", "?", "\n"]
            separators = [kwargs.get("separator", "。")] + custom_seps
            kwargs["separator"] = '|'.join(map(re.escape, separators))
            super().__init__(*args, **kwargs)

        def _split_text(self, text: str, **kwargs) -> List[str]:
            # 按照标点符号来切分，但是保留标点符号，因为f-string中有括号
            splits = re.split(f'({self.separator})', text)
            chunks = []
            current_chunk = []
            for part in splits:
                # strip() 会同时去掉字符串首尾的 空格 和 \n\r（换行符）,\t,\v,\f，也就是说如果seprarator如何是空格或者\n,在这里是会呗去掉的
                part = part.strip()
                if not part:
                    continue
                # 匹配是否是标点符号
                if re.fullmatch(self.separator, part):
                    if current_chunk:
                        # 把标点符合符号页加上，原代码没有加标点进去，这样召回显示给用户看时，没有标点的段落读起来会非常累。
                        current_chunk.append(part)  # 保留标点，我加的。
                        # 注意这里用join()方法，把current_chunk中的内容拼接成字符串，并添加到chunks列表中。这是为了语义的完整性
                        chunks.append("".join(current_chunk))
                        current_chunk = []
                else:
                    # 如果separator如何是空格或者\n,各个part会在这里append到current_chunk中
                    current_chunk.append(part)
                    # logger.info("append part")

            if current_chunk:
                chunks.append("".join(current_chunk))
            return [chunk.strip() for chunk in chunks if chunk.strip()]

    text_splitter = EnhancedSentenceSplitter(
        separator="。",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n"
    )

    ##############---按照段落或长度来分段---#################
    #原来的代码
    logger.info("执行分段逻辑")
    # 分段，有问题，没有处理超长段落
    paragraphs = []
    current_para = []
    current_len = 0
    para_id = 0

    # 按照\n\n来分段落，遍历每个段落
    # 合并小段落，直到段落长度小于= chunk_size
    # 问题：如果全文没有\n\n，那么全文就是一个长段落，相当于没有划分，没有处理长段落逻辑。
    i = 0
    logger.info("执行分段逻辑")
    for para in text.split("\n\n"):
        para = para.strip()
        para_len = len(para)
        logger.info(f"段落i={i},长度para_len={para_len}")

        if para_len == 0:
            continue
        if current_len + para_len <= chunk_size:
            current_para.append(para)
            current_len += para_len
        else:
            if current_para:
                paragraphs.append("\n".join(current_para))
                logger.info("*" * 60)
                logger.info(f"分段id:{para_id},存入：{current_para}")
                para_id += 1
            if current_len > chunk_size:
                logger.info("*" * 60)
                logger.info(f"存入超出段落，chunk_size > {chunk_size}, 内容为: {current_para}")


            current_para = [para]
            current_len = para_len
        i += 1

    # 如果最后一个段落，没有加入到段落列表中，则加入
    if current_para:
        paragraphs.append("\n".join(current_para))

    logger.info(f"i={i},最后一个分段id:para_id={para_id},存入：{current_para}")


    #############---按照标点符号来分块--####################
    logger.info("执行分块逻辑")
    chunk_data_list = []
    chunk_id = 0
    for para in paragraphs:
        chunks = text_splitter.split_text(para)
        for chunk in chunks:
            # 这里默认chunk_filter_len=2是为了过滤掉过短的段落，可以自己调整.原先是20，我改成了2，后期可以把这个放到config文件中，方便配置。
            if len(chunk) < Config.chunk_filter_len:
                logger.info("-" * 60)
                logger.info(f"过滤掉过短的段落：{chunk}")
                continue
            chunk_data_list.append({
                "id": f'chunk{chunk_id}',
                "chunk": chunk,
                "method": "semantic_chunk"
            })
            log_data = {
                "id": f'chunk{chunk_id}',
                "chunk": chunk,  # 截断以防刷屏
                "method": "semantic_chunk"
            }
            logger.info("-" * 60)
            logger.info(f"生成分块数据: {log_data}")
            chunk_id += 1

    return chunk_data_list

