if 0:
    import streamlit as st

    st.title("我的 AI Demo")
    name = st.text_input("输入你的名字：")
    if name:
        st.write("Hello,", name)

## 503报错
if 1:
    import os

    # 彻底关闭本进程的代理设置，避免访问 127.0.0.1 走代理
    for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
              "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"]:
        os.environ.pop(k, None)

    # 明确告诉程序：访问本机不要走代理
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"

    import gradio as gr

    def echo(text):
        return f"你说的是：{text}"


    demo = gr.Interface(fn=echo,
                        inputs="text",
                        outputs="text",
                        title="测试 Gradio")

    if __name__ == "__main__":
        demo.launch(server_name="127.0.0.1", server_port=7860)  # 注意这里不用 share=True