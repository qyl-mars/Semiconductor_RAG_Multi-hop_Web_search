# gradio 快速制作网页

dogradio = 0
if dogradio == 1:
    import gradio as gr

    # 假设的情感分析函数（实际中可能是调用模型）
    def sentiment_analysis(text):
        if "好" in text or "棒" in text:
            return "正面情感"
        else:
            return "负面情感"

    # 创建界面：输入为文本框，输出为文本
    iface = gr.Interface(
        fn=sentiment_analysis,
        inputs=gr.Textbox(label="输入文本"),
        outputs=gr.Textbox(label="情感结果")
    )

    # 启动服务（默认在本地7860端口）
    iface.launch()

# fassi-cup 向量相似性高效查询
import numpy as np
import faiss

# 1. 生成模拟数据：100个向量，每个向量维度为5
np.random.seed(42)
vectors = np.random.rand(100, 5).astype("float32")  # 数据库向量
query = np.random.rand(1, 5).astype("float32")      # 查询向量

# 2. 构建FAISS索引（这里用最简单的暴力搜索索引）
index = faiss.IndexFlatL2(5)  # 维度为5，L2距离（欧氏距离）
index.add(vectors)            # 将数据库向量加入索引

# 3. 搜索：找与query最相似的前3个向量
k = 3
distances, indices = index.search(query, k)  # distances是距离，indices是索引位置

print("query:",query)
print("最相似的3个向量：", vectors[indices[0]])
print("最相似的3个向量索引：", indices[0])  # 输出：[xx, xx, xx]
print("对应的距离：", distances[0])          # 距离越小越相似