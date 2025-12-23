import json
from llm.embedding_client import vectorize_query

# 向量化文件内容
def vectorize_file(data_list, output_file_path, field_name="chunk"):
    """向量化文件内容，处理长度限制并确保输入有效"""
    if not data_list:
        print("警告: 没有数据需要向量化")
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump([], outfile, ensure_ascii=False, indent=4)
        return

    # 准备查询文本，确保每个文本有效且长度适中
    valid_data = []
    valid_texts = []

    for data in data_list:
        text = data.get(field_name, "")
        # 确保文本不为空且长度合适
        if text and 1 <= len(text) <= 8000:  # 略小于API限制的8192，留出一些余量
            valid_data.append(data)
            valid_texts.append(text)
        else:
            # 如果文本太长，截断它
            if len(text) > 8000:
                truncated_text = text[:8000]
                print(f"警告: 文本过长，已截断至8000字符。原始长度: {len(text)}")
                data[field_name] = truncated_text
                valid_data.append(data)
                valid_texts.append(truncated_text)
            else:
                print(f"警告: 跳过空文本或长度为0的文本")

    if not valid_texts:
        print("错误: 所有文本都无效，无法进行向量化")
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump([], outfile, ensure_ascii=False, indent=4)
        return

    # 向量化有效文本
    vectors = vectorize_query(valid_texts)

    # 检查向量化是否成功
    if vectors.size == 0 or len(vectors) != len(valid_data):
        print \
            (f"错误: 向量化失败或向量数量({len(vectors) if vectors.size > 0 else 0})与数据条目({len(valid_data)})不匹配")
        # 保存原始数据，但不含向量
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(valid_data, outfile, ensure_ascii=False, indent=4)
        return

    # 添加向量到数据中
    for data, vector in zip(valid_data, vectors):
        data['vector'] = vector.tolist()

    # 保存结果
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(valid_data, outfile, ensure_ascii=False, indent=4)

    print(f"成功向量化 {len(valid_data)} 条数据并保存到 {output_file_path}")

