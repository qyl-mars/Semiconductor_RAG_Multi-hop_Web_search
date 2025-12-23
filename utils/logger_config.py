import os
from config.configs import Config
import logging
from logging.handlers import TimedRotatingFileHandler

def setup_logger(log_name="rag.log"):
    # 配置日志保存路径
    # --- 日志配置开始 ---
    # 1. 创建日志目录
    LOG_DIR = Config.log_dir
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    LOG_FILE = os.path.join(LOG_DIR,log_name)
    # 2. 定义全局日志格式
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')

    # 3. 创建处理器 (Handler)
    # TimedRotatingFileHandler 参数解释：
    # when="D": 按天滚动
    # interval=1: 间隔为 1 天
    # backupCount=7: 保留最近 7 天的日志，旧的自动删除
    # encoding='utf-8': 防止中文乱码
    file_handler = TimedRotatingFileHandler(
        LOG_FILE, when="D", interval=1, backupCount=60, encoding='utf-8'
    )
    file_handler.setFormatter(log_format)

    # 创建控制台处理器（让你在 Terminal 也能看到）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    # 4. 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # 设置日志级别为 INFO
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # --- 日志配置结束 ---
    return logger