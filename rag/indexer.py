import json
import faiss
import numpy as np
import traceback

# 构建Faiss索引
def build_faiss_index(vector_file, index_path, metadata_path):
    try:
        with open(vector_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            raise ValueError("向量数据为空，请检查输入文件。")

        # 确认所有数据项都有向量
        valid_data = []
        for item in data:
            if 'vector' in item and item['vector']:
                valid_data.append(item)
            else:
                print(f"警告: 跳过没有向量的数据项 ID: {item.get('id', '未知')}")

        if not valid_data:
            raise ValueError("没有找到任何有效的向量数据。")

        # 提取向量
        vectors = [item['vector'] for item in valid_data]
        vectors = np.array(vectors, dtype=np.float32)

        if vectors.size == 0:
            raise ValueError("向量数组为空，转换失败。")

        # 检查向量维度
        dim = vectors.shape[1]
        n_vectors = vectors.shape[0]
        print(f"构建索引: {n_vectors} 个向量，每个向量维度: {dim}")

        # 确定索引类型和参数
        max_nlist = n_vectors // 39
        nlist = min(max_nlist, 128) if max_nlist >= 1 else 1

        # 在 Faiss 的 IndexIVFFlat 训练机制中的硬性规定：
        # 要训练出 n 个聚类中心，训练数据最好是 n 的 39 倍以上。
        # n_vectors >= 39， 走 IndexIVFFlat
        # 源代码
        #if nlist >= 1 and n_vectors >= nlist * 39:
        # qyl,增加10000判断
        if n_vectors > 10000 and nlist >= 1 and n_vectors >= nlist * 39:
            print(f"使用 IndexIVFFlat 索引，nlist={nlist}")
            # 创建暴力搜索索引，是创建了一个使用内积作为相似度度量的 Flat 向量索引
            quantizer = faiss.IndexFlatIP(dim)
            # 创建索引，
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            if not index.is_trained:
                # k-均值聚类，将向量分配到不同的簇（cluster）
                index.train(vectors)
            index.add(vectors)
        # 源文件，n_vectors 小于 39，走 IndexFlatIP
        # qyl：，n_vectors 小于10000
        else:
            print(f"使用 IndexFlatIP 索引")
            index = faiss.IndexFlatIP(dim)
            index.add(vectors)

        faiss.write_index(index, index_path)
        print(f"成功写入索引到 {index_path}")

        # 创建元数据
        metadata = [{'id': item['id'], 'chunk': item['chunk'], 'method': item['method']} for item in valid_data]
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        print(f"成功写入元数据到 {metadata_path}")

        return True
    except Exception as e:
        print(f"构建索引失败: {str(e)}")
        traceback.print_exc()
        raise
