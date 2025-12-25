import os

class Config():

    #retrievor参数
    topd = 3    #召回文章的数量
    topt = 6    #召回文本片段的数量
    maxlen = 128  #召回文本片段的长度
    topk = 5    #query召回的关键词数量
    bert_path = '/workspace/model/embedding/tao-8k'
    recall_way = 'embed'  #召回方式 ,keyword,embed

    #chunker 参数
    chunk_size = 800
    chunk_overlap = 20
    chunk_filter_len = 2

    #generator参数
    max_source_length = 767  #输入的最大长度
    max_target_length = 256  #生成的最大长度
    model_max_length = 1024  #序列最大长度
    
    #embedding API 参数 - 用于 text2vec.py
    use_api = True  # 是否使用API而非本地模型
    api_key = "sk-9cd7a8dd61214b52a869bd3af53a0a1d"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name = "text-embedding-v3"
    dimensions = 1024
    batch_size = 10
    
    #LLM API 参数 - 用于 rag.py
    # ali qwen
    llm_api_key = "sk-9cd7a8dd61214b52a869bd3af53a0a1d"  # 与embedding共用同一个key
    llm_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 与embedding共用同一个URL
    llm_model = "qwen-plus"  # 默认使用的LLM模型

    """
    # gpt
    llm_api_key = "sk-VI3k06ySl6WOWU16o1uyb7yykSh3rKWBDXJNIfk6TsouHrP1"  # 与embedding共用同一个key
    llm_base_url = "https://yibuapi.com/v1"  # 与embedding共用同一个URL
    llm_model = "gpt-5.2-pro"  # 默认使用的LLM模型
    #"""

    # tavily ，联网搜索
    tavily_api_key = "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"

    # Rerank模型
    # 硅基流动
    rerank_model = "BAAI/bge-reranker-v2-m3"
    #其他可选：netease-youdao/bce-reranker-base_v1 \ Qwen/Qwen3-Reranker-4B \Qwen/Qwen3-Reranker-8B
    # Qwen/Qwen3-Reranker-0.6B / Pro/BAAI/bge-reranker-v2-m3
    rerank_api_kay= "sk-dentfpwhfygjdkjhwgltshwiltqcfrvmmkajxvskyfcmgzmr"

    # 知识库配置
    kb_base_dir = "knowledge_bases"  # 知识库根目录
    default_kb = "aaa"  # 默认知识库名称
    
    # 输出目录配置 - 现在用作临时文件目录
    output_dir = "output_files"

    # logging 配置
    log_dir = "logs"
    
    # rerank 配置
    use_rerank = False  # 是否启用 rerank 重排序
    rerank_method = "llm"  # rerank 方法: "llm" 或 "text_similarity"
    rerank_batch_size = 5  # LLM rerank 的批量处理大小
    rerank_top_k = None  # rerank 后返回的结果数量，None 表示返回所有重排序后的结果
    
    # 提示词配置
    default_domain = "semiconductor"  # 默认领域: "semiconductor"（半导体）

    #######################
