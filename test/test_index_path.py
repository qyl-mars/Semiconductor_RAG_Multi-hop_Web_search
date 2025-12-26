# 检查索引文件大小和修改时间
import os

index_path = "knowledge_bases/aaa/semantic_chunk.index"

#index_path = r"D:\qiumuxi\demo\Rag\knowledge_bases\碳化硅MOSFET\semantic_chunk.index"

if 0:
    # 打印出 Python 当前站的位置（参考系原点）
    print(f"当前工作目录: {os.getcwd()}")
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    print(f"项目根目录: {project_root}")
    index_path = os.path.join(project_root, index_path)


print(f"索引文件路径: {index_path}")
if os.path.exists(index_path):
    file_size = os.path.getsize(index_path)
    print(f"Index file size: {file_size} bytes")
else:
    print("Index file does not exist")