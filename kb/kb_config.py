from config.configs import Config
import os

# 创建知识库根目录和临时文件目录
KB_BASE_DIR = Config.kb_base_dir
os.makedirs(KB_BASE_DIR, exist_ok=True)

# 创建默认知识库目录
DEFAULT_KB = Config.default_kb
DEFAULT_KB_DIR = os.path.join(KB_BASE_DIR, DEFAULT_KB)
os.makedirs(DEFAULT_KB_DIR, exist_ok=True)

# 创建临时输出目录
OUTPUT_DIR = Config.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
