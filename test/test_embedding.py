from config.configs import Config
from openai import OpenAI
import numpy as np
from ingest.text_cleaner import clean_text
import traceback
import time

start_time = time.perf_counter()
model_name=Config.model_name
embedding_client = OpenAI(
    api_key=Config.api_key,
    base_url=Config.base_url
)

batch = ["你好。" , "我是测试embedding code"]
"""
Pydantic 对象
completion 不是一个字典 (Dictionary)，而是一个 Pydantic 对象 (Object)。
1. 用点调用，到自动补全
❌ 不能像查字典那样用中括号取值：completion['data'] 会报错。
✅ 必须用“点”来调用属性：completion.data。
"""
completion = embedding_client.embeddings.create(
                model=model_name,
                input=batch,
                dimensions=Config.dimensions,
                encoding_format="float"
            )
print('completion:\n',completion)

# 把它转成真正的字典
completion_dict = completion.model_dump()

# 现在可以打印所有的键了
print(completion_dict.keys())
#输出是：dict_keys(['data', 'model', 'object', 'usage', 'id'])

vectors = [embedding.embedding for embedding in completion.data]

# 看看外层有多少个（就是你的 batch size）
print(f"总共有多少条向量: {len(vectors)}")

# 看看单个向量有多长（就是模型的 dimensions）
print(f"单条向量的维度: {len(vectors[0])}")

end_time = time.perf_counter()
print(f"运行时间: {end_time - start_time:.2f} 秒")