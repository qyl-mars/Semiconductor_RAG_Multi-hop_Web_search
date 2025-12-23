# 解决503报错，关闭代理
import os

def disable_proxy():
    # 1. 清空常见代理环境变量
    for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
        os.environ.pop(k, None)

    # 2. 告诉 Python/HTTP 库：访问这些地址时不要走代理
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"



