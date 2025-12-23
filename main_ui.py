
from config.env import disable_proxy
from web_ui.web_ui import demo

'''
原先的代码rag.py 被拆解
这里作为入口函数
'''
if __name__ == "__main__":
    #demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
    disable_proxy()
    demo.launch(
        server_name="127.0.0.1",  # 只在本机访问就用 127.0.0.1
        server_port=7863  # 端口你可以改成别的，比如 7861
        # 不要写 share=True，先把公网分享关掉
    )