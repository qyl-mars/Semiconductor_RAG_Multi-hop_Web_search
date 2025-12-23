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
    llm_api_key = "sk-9cd7a8dd61214b52a869bd3af53a0a1d"  # 与embedding共用同一个key
    llm_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 与embedding共用同一个URL
    llm_model = "qwen-plus"  # 默认使用的LLM模型


    # 知识库配置
    kb_base_dir = "knowledge_bases"  # 知识库根目录
    default_kb = "default"  # 默认知识库名称
    
    # 输出目录配置 - 现在用作临时文件目录
    output_dir = "output_files"

    # logging 配置
    log_dir = "logs"
    
    # rerank 配置
    use_rerank = False  # 是否启用 rerank 重排序
    rerank_method = "llm"  # rerank 方法: "llm" 或 "text_similarity"
    rerank_batch_size = 5  # LLM rerank 的批量处理大小
    rerank_top_k = None  # rerank 后返回的结果数量，None 表示返回所有重排序后的结果

    #######################
