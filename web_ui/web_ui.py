import gradio as gr
import os
from kb.kb_config import KB_BASE_DIR,DEFAULT_KB
from kb.kb_manager import get_knowledge_bases,create_knowledge_base,delete_knowledge_base,\
    get_kb_files
from ingest.ingest_service import batch_upload_to_kb
from rag.streaming_handler import process_question_with_reasoning



# Gradio 界面 - 修改为支持多知识库
custom_css = """
.web-search-toggle .form { display: flex !important; align-items: center !important; }
.web-search-toggle .form > label { order: 2 !important; margin-left: 10px !important; }
.web-search-toggle .checkbox-wrap { order: 1 !important; background: #d4e4d4 !important; border-radius: 15px !important; padding: 2px !important; width: 50px !important; height: 28px !important; }
.web-search-toggle .checkbox-wrap .checkbox-container { width: 24px !important; height: 24px !important; transition: all 0.3s ease !important; }
.web-search-toggle input:checked + .checkbox-wrap { background: #2196F3 !important; }
.web-search-toggle input:checked + .checkbox-wrap .checkbox-container { transform: translateX(22px) !important; }
#search-results { max-height: 400px; overflow-y: auto; border: 1px solid #2196F3; border-radius: 5px; padding: 10px; background-color: #e7f0f9; }
#question-input { border-color: #2196F3 !important; }
#answer-output { background-color: #f0f7f0; border-color: #2196F3 !important; max-height: 400px; overflow-y: auto; }
.submit-btn { background-color: #2196F3 !important; border: none !important; }
.reasoning-steps { background-color: #f0f7f0; border: 1px dashed #2196F3; padding: 10px; margin-top: 10px; border-radius: 5px; }
.loading-spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid rgba(33, 150, 243, 0.3); border-radius: 50%; border-top-color: #2196F3; animation: spin 1s ease-in-out infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.stream-update { animation: fade 0.5s ease-in-out; }
@keyframes fade { from { background-color: rgba(33, 150, 243, 0.1); } to { background-color: transparent; } }
.status-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; font-weight: bold; }
.status-processing { background-color: #e3f2fd; color: #1565c0; border-left: 4px solid #2196F3; }
.status-success { background-color: #e8f5e9; color: #2e7d32; border-left: 4px solid #4CAF50; }
.status-error { background-color: #ffebee; color: #c62828; border-left: 4px solid #f44336; }
.multi-hop-toggle .form { display: flex !important; align-items: center !important; }
.multi-hop-toggle .form > label { order: 2 !important; margin-left: 10px !important; }
.multi-hop-toggle .checkbox-wrap { order: 1 !important; background: #d4e4d4 !important; border-radius: 15px !important; padding: 2px !important; width: 50px !important; height: 28px !important; }
.multi-hop-toggle .checkbox-wrap .checkbox-container { width: 24px !important; height: 24px !important; transition: all 0.3s ease !important; }
.multi-hop-toggle input:checked + .checkbox-wrap { background: #4CAF50 !important; }
.multi-hop-toggle input:checked + .checkbox-wrap .checkbox-container { transform: translateX(22px) !important; }
.kb-management { border: 1px solid #2196F3; border-radius: 5px; padding: 15px; margin-bottom: 15px; background-color: #f0f7ff; }
.kb-selector { margin-bottom: 10px; }
/* 缩小文件上传区域高度 */
.compact-upload {
    margin-bottom: 10px;
}

.file-upload.compact {
    padding: 10px;  /* 减小内边距 */
    min-height: 120px; /* 减小最小高度 */
    margin-bottom: 10px;
}

/* 优化知识库内容显示区域 */
.kb-files-list {
    height: 400px;
    overflow-y: auto;
}

/* 确保右侧列有足够空间 */
#kb-files-group {
    height: 100%;
    display: flex;
    flex-direction: column;
}
.kb-files-list { max-height: 250px; overflow-y: auto; border: 1px solid #ccc; border-radius: 5px; padding: 10px; margin-top: 10px; background-color: #f9f9f9; }
#kb-management-container {
    max-width: 800px !important;
    margin: 0 !important; /* 移除自动边距，靠左对齐 */
    margin-left: 20px !important; /* 添加左边距 */
}
.container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
.file-upload {
    border: 2px dashed #2196F3;
    padding: 15px;
    border-radius: 10px;
    background-color: #f0f7ff;
    margin-bottom: 15px;
}
.tabs.tab-selected {
    background-color: #e3f2fd;
    border-bottom: 3px solid #2196F3;
}
.group {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 15px;
    background-color: #fafafa;
}

/* 添加更多针对知识库管理页面的样式 */
#kb-controls, #kb-file-upload, #kb-files-group {
    width: 100% !important;
    max-width: 800px !important;
    margin-right: auto !important;
}

/* 修改Gradio默认的标签页样式以支持左对齐 */
.tabs > .tab-nav > button {
    flex: 0 1 auto !important; /* 修改为不自动扩展，只占用必要空间 */
}
.tabs > .tabitem {
    padding-left: 0 !important; /* 移除左边距，使内容靠左 */
}
/* 对于首页的顶部标题部分 */
#app-container h1, #app-container h2, #app-container h3, 
#app-container > .prose {
    text-align: left !important;
    padding-left: 20px !important;
}
"""

custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="gray",
    text_size="lg",
    spacing_size="md",
    radius_size="md"
)

# 添加简单的JavaScript，通过html组件实现
js_code = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    // 当页面加载完毕后，找到提交按钮，并为其添加点击事件
    const observer = new MutationObserver(function(mutations) {
        // 找到提交按钮
        const submitButton = document.querySelector('button[data-testid="submit"]');
        if (submitButton) {
            submitButton.addEventListener('click', function() {
                // 找到检索标签页按钮并点击它
                setTimeout(function() {
                    const retrievalTab = document.querySelector('[data-testid="tab-button-retrieval-tab"]');
                    if (retrievalTab) retrievalTab.click();
                }, 100);
            });
            observer.disconnect(); // 一旦找到并设置事件，停止观察
        }
    });

    // 开始观察文档变化
    observer.observe(document.body, { childList: true, subtree: true });
});
</script>
"""


with gr.Blocks(title="半导体知识问答系统", theme=custom_theme, css=custom_css, elem_id="app-container") as demo:
    with gr.Column(elem_id="header-container"):
        gr.Markdown("""
        # 🏥 半导体知识问答系统
        **智能半导体助手，支持多知识库管理、多轮对话、普通语义检索、联网检索和高级多跳推理**  
        本系统支持创建多个知识库，上传TXT或PDF文件，通过语义向量检索或创新的多跳推理机制提供企业半导体信息查询服务。
        """)

    # 添加JavaScript脚本
    '''
    ✔ 页面加载完成后
    ✔ 自动给“提交问题”按钮绑定点击事件
    ✔ 用户一点击提交
    ✔ 自动切换到“检索进展”Tab
    '''
    gr.HTML(js_code, visible=False)

    # 使用State来存储对话历史
    '''
    跨多次按钮点击保存对话历史
    它是 会话级（session-level）状态
    页面刷新 → State 重置
    服务器重启 → State 消失
    '''
    chat_history_state = gr.State([])

    # 创建标签页的总容器
    with gr.Tabs() as tabs:
        # 第一个标签页：知识库管理标签页
        with gr.TabItem("知识库管理"):
            with gr.Row():
                # 第一个标签页左侧列：控制区
                with gr.Column(scale=1, min_width=400):
                    gr.Markdown("### 📚 知识库管理与构建")

                    with gr.Row(elem_id="kb-controls"):
                        with gr.Column(scale=1): # 创建一个列，在同一 Row 里占 1 份宽度
                            # 用户输入框，“新知识库名称”，单行，并把它赋值给变量 new_kb_name，作为create_kb_btn的输入。
                            new_kb_name = gr.Textbox(
                                label="新知识库名称", # 标签
                                placeholder="输入新知识库名称", # 提示
                                lines=1 # 输入框的行数
                            )
                            # “创建知识库”的按钮
                            create_kb_btn = gr.Button("创建知识库", variant="primary", scale=1)

                        # 从下拉框中选择一个知识库，
                        # 下拉框中所有可选项 = 当前已有的知识库列表
                        with gr.Column(scale=1):
                            current_kbs = get_knowledge_bases()
                            kb_dropdown = gr.Dropdown(
                                label="选择知识库",
                                choices=current_kbs,
                                # 默认选中DEFAULT知识库, 如果DEFAULT不存在，则默认选中第一个, 否在不选中任何内容
                                value=DEFAULT_KB if DEFAULT_KB in current_kbs else (
                                    current_kbs[0] if current_kbs else None),
                                # 下拉框样式，给这个组件加一个 CSS class
                                elem_classes="kb-selector"
                            )

                            # 横排2个按钮，刷新知识库列表、删除知识库操作
                            with gr.Row():
                                refresh_kb_btn = gr.Button("刷新列表", size="sm", scale=1)
                                delete_kb_btn = gr.Button("删除知识库", size="sm", variant="stop", scale=1)

                    # 第一个标签页：左侧列：知识库状态
                    # interactive=False，是个接收器，只读”的显示区域，不能输入
                    # kb_status，最输出到显示区的内容，如果没有，则显示"选择或创建知识库"
                    # kb_status的值取决于最新一个触发事件（如click)的outputs,展示数据库是创建、删除、更改的状态（是否成功或者失败）
                    kb_status = gr.Textbox(label="知识库状态", interactive=False, placeholder="选择或创建知识库")

                    # 第一个标签页：左侧列：最后一个组件
                    # 上传文件，并完成解析、向量化，并保存到数据库
                    # gr.File，上传文件（拖拽或点击）
                    with gr.Group(elem_id="kb-file-upload", elem_classes="compact-upload"):
                        gr.Markdown("### 📄 上传文件到知识库")
                        file_upload = gr.File(
                            label="选择文件（支持多选TXT/PDF）",
                            type="filepath",
                            file_types=[".txt", ".pdf"],
                            file_count="multiple",
                            elem_classes="file-upload compact"
                        )
                        upload_status = gr.Textbox(label="上传状态", interactive=False, placeholder="上传后显示状态")


                    kb_select_for_chat = gr.Dropdown(
                        label="为对话选择知识库",
                        choices=current_kbs,
                        value=DEFAULT_KB if DEFAULT_KB in current_kbs else (current_kbs[0] if current_kbs else None),
                        visible=False  # 隐藏，仅用于同步
                    )
                # 第一个标签页：右侧列
                # 展示选中知识库的文件列表
                with gr.Column(scale=1, min_width=400):
                    with gr.Group(elem_id="kb-files-group"):
                        gr.Markdown("### 📋 知识库内容")
                        kb_files_list = gr.Markdown(
                            value="选择知识库查看文件...",
                            elem_classes="kb-files-list"
                        )
                # kb_select_for_chat 是一个隐藏的 UI 状态组件，
                # 用于在多 Tab、多事件链之间同步当前选中的知识库，
                # 确保 RAG 检索和对话始终使用一致的上下文
                # 用于对话界面的知识库选择器
                kb_select_for_chat = gr.Dropdown(
                    label="为对话选择知识库",
                    choices=current_kbs,
                    value=DEFAULT_KB if DEFAULT_KB in current_kbs else (current_kbs[0] if current_kbs else None),
                    visible=False  # 隐藏，仅用于同步
                )
        # 第二个标签页：对话交互标签页
        with gr.TabItem("对话交互"):
            with gr.Row():
                # 第二个标签页：左侧
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 对话设置")


                    kb_dropdown_chat = gr.Dropdown(
                        label="选择知识库进行对话",
                        choices=current_kbs,
                        value=DEFAULT_KB if DEFAULT_KB in current_kbs else (current_kbs[0] if current_kbs else None),
                    )

                    with gr.Row():
                        web_search_toggle = gr.Checkbox(
                            label="🌐 启用联网搜索",
                            value=True,
                            info="获取最新半导体动态",
                            elem_classes="web-search-toggle"
                        )
                        table_format_toggle = gr.Checkbox(
                            label="📊 表格格式输出",
                            value=True,
                            info="使用Markdown表格展示结构化回答",
                            elem_classes="web-search-toggle"
                        )

                    multi_hop_toggle = gr.Checkbox(
                        label="🔄 启用多跳推理",
                        value=False,
                        info="使用高级多跳推理机制（较慢但更全面）",
                        elem_classes="multi-hop-toggle"
                    )

                    # 展示检索进展，可折叠区域
                    with gr.Accordion("显示检索进展", open=False):
                        search_results_output = gr.Markdown(
                            label="检索过程",
                            elem_id="search-results",
                            value="等待提交问题..."
                        )
                # 第二个标签页：右侧
                with gr.Column(scale=3):
                    gr.Markdown("### 💬 对话历史")
                    chatbot = gr.Chatbot(
                        elem_id="chatbot",
                        label="对话历史",
                        height=550
                    )

            with gr.Row():
                question_input = gr.Textbox(
                    label="输入半导体相关问题",
                    placeholder="例如：碳化硅沟槽MOSFET有哪些可靠性问题？",
                    lines=2,
                    elem_id="question-input"
                )

            with gr.Row(elem_classes="submit-row"):
                submit_btn = gr.Button("提交问题", variant="primary", elem_classes="submit-btn")
                clear_btn = gr.Button("清空输入", variant="secondary")
                clear_history_btn = gr.Button("清空对话历史", variant="secondary", elem_classes="clear-history-btn")

            # 状态显示框
            status_box = gr.HTML(
                value='<div class="status-box status-processing">准备就绪，等待您的问题</div>',
                visible=True
            )

            gr.Examples(
                examples=[
                    ["怎样通过显示器上的图标“报警”故障类别？"],
                    ["碳化硅MOSFET的栅氧可靠性受哪些因素影响？"],
                    ["存内计算芯片是什么？"],
                    ["华为有哪些先进的AI芯片？"],
                    ["今天星期几？"],
                    ["光刻机的作用是什么？"]

                ],
                inputs=question_input,
                label="示例问题（点击尝试）"
            )


    ################## 事件回调函数（event handlers / callbacks） ##################
    # 创建知识库函数
    # 创建一个新的知识库，然后刷新所有“知识库选择下拉框”，并把新建的知识库自动选中。
    def create_kb_and_refresh(kb_name):
        result = create_knowledge_base(kb_name)
        kbs = get_knowledge_bases()
        # 更新两个下拉菜单
        return result, gr.update(choices=kbs, value=kb_name if "创建成功" in result else None), gr.update(choices=kbs,
                                                                                                          value=kb_name if "创建成功" in result else None)


    # 刷新知识库列表

    def refresh_kb_list():
        kbs = get_knowledge_bases()
        # 更新两个下拉菜单
        return gr.update(choices=kbs, value=kbs[0] if kbs else None), gr.update(choices=kbs,
                                                                                value=kbs[0] if kbs else None)


    # 删除知识库
    def delete_kb_and_refresh(kb_name):
        result = delete_knowledge_base(kb_name)
        kbs = get_knowledge_bases()
        # 更新两个下拉菜单
        return result, gr.update(choices=kbs, value=kbs[0] if kbs else None), gr.update(choices=kbs,
                                                                                        value=kbs[0] if kbs else None)


    # 更新知识库文件列表
    def update_kb_files_list(kb_name):
        if not kb_name:
            return "未选择知识库"

        files = get_kb_files(kb_name)
        kb_dir = os.path.join(KB_BASE_DIR, kb_name)
        has_index = os.path.exists(os.path.join(kb_dir, "semantic_chunk.index"))

        if not files:
            files_str = "知识库中暂无文件"
        else:
            files_str = "**文件列表:**\n\n" + "\n".join([f"- {file}" for file in files])

        index_status = "\n\n**索引状态:** " + ("✅ 已建立索引" if has_index else "❌ 未建立索引")

        return f"### 知识库: {kb_name}\n\n{files_str}{index_status}"


    # 同步知识库选择 - 管理界面到对话界面
    def sync_kb_to_chat(kb_name):
        return gr.update(value=kb_name)


    # 同步知识库选择 - 对话界面到管理界面
    def sync_chat_to_kb(kb_name):
        return gr.update(value=kb_name), update_kb_files_list(kb_name)


    # 处理文件上传到指定知识库
    def process_upload_to_kb(files, kb_name):
        if not kb_name:
            return "错误：未选择知识库"

        result = batch_upload_to_kb(files, kb_name)
        # 更新知识库文件列表
        files_list = update_kb_files_list(kb_name)
        return result, files_list


    # 知识库选择变化时
    def on_kb_change(kb_name):
        if not kb_name:
            return "未选择知识库", "选择知识库查看文件..."

        kb_dir = os.path.join(KB_BASE_DIR, kb_name)
        has_index = os.path.exists(os.path.join(kb_dir, "semantic_chunk.index"))
        status = f"已选择知识库: {kb_name}" + (" (已建立索引)" if has_index else " (未建立索引)")

        # 更新文件列表
        files_list = update_kb_files_list(kb_name)

        return status, files_list


    ################## 事件绑定，定义组件触发事件的功能 ##################
    # 创建知识库按钮功能
    create_kb_btn.click(
        fn=create_kb_and_refresh,
        inputs=[new_kb_name],
        outputs=[kb_status, kb_dropdown, kb_dropdown_chat]
    ).then(
        fn=lambda: "",  # 清空输入框
        inputs=[],
        outputs=[new_kb_name]
    )

    # 刷新知识库列表按钮功能
    refresh_kb_btn.click(
        fn=refresh_kb_list,
        inputs=[],
        outputs=[kb_dropdown, kb_dropdown_chat]
    )

    # 删除知识库按钮功能
    delete_kb_btn.click(
        fn=delete_kb_and_refresh,
        inputs=[kb_dropdown],
        outputs=[kb_status, kb_dropdown, kb_dropdown_chat]
    ).then(
        fn=update_kb_files_list,
        inputs=[kb_dropdown],
        outputs=[kb_files_list]
    )

    # 知识库选择变化时 - 管理界面
    kb_dropdown.change(
        fn=on_kb_change,
        inputs=[kb_dropdown],
        outputs=[kb_status, kb_files_list]
    ).then(
        fn=sync_kb_to_chat,
        inputs=[kb_dropdown],
        outputs=[kb_dropdown_chat]
    )

    # 知识库选择变化时 - 对话界面
    kb_dropdown_chat.change(
        fn=sync_chat_to_kb,
        inputs=[kb_dropdown_chat],
        outputs=[kb_dropdown, kb_files_list]
    )

    # 处理文件上传
    file_upload.upload(
        fn=process_upload_to_kb,
        inputs=[file_upload, kb_dropdown],
        outputs=[upload_status, kb_files_list]
    )

    # 清空输入按钮功能
    clear_btn.click(
        fn=lambda: "",
        inputs=[],
        outputs=[question_input]
    )


    # 清空对话历史按钮功能
    def clear_history():
        return [], []


    clear_history_btn.click(
        fn=clear_history,
        inputs=[],
        outputs=[chatbot, chat_history_state]
    )


    # 提交按钮 - 开始流式处理
    def update_status(is_processing=True, is_error=False):
        if is_processing:
            return '<div class="status-box status-processing">正在处理您的问题...</div>'
        elif is_error:
            return '<div class="status-box status-error">处理过程中出现错误</div>'
        else:
            return '<div class="status-box status-success">回答已生成完毕</div>'


    # 处理问题并更新对话历史
    def process_and_update_chat(question, kb_name, use_search, use_table_format, multi_hop, chat_history):
        if not question.strip():
            return chat_history, update_status(False, True), "等待提交问题..."

        try:
            # 首先更新聊天界面，显示用户问题
            chat_history.append([question, "正在思考..."])
            yield chat_history, update_status(True), f"开始处理您的问题，使用知识库: {kb_name}..."

            # 用于累积检索状态和答案
            last_search_display = ""
            last_answer = ""

            # 使用生成器进行流式处理
            for search_display, answer in process_question_with_reasoning(question, kb_name, use_search,
                                                                          use_table_format, multi_hop,
                                                                          chat_history[:-1]):
                # 更新检索状态和答案
                last_search_display = search_display
                last_answer = answer

                # 更新聊天历史中的最后一条（当前的回答）
                if chat_history:
                    chat_history[-1][1] = answer
                    yield chat_history, update_status(True), search_display

            # 处理完成，更新状态
            yield chat_history, update_status(False), last_search_display

        except Exception as e:
            # 发生错误时更新状态和聊天历史
            error_msg = f"处理问题时出错: {str(e)}"
            if chat_history:
                chat_history[-1][1] = error_msg
            yield chat_history, update_status(False, True), f"### 错误\n{error_msg}"


    # 连接提交按钮
    submit_btn.click(
        fn=process_and_update_chat,
        inputs=[question_input, kb_dropdown_chat, web_search_toggle, table_format_toggle, multi_hop_toggle,
                chat_history_state],
        outputs=[chatbot, status_box, search_results_output],
        queue=True
    ).then(
        fn=lambda: "",  # 清空输入框
        inputs=[],
        outputs=[question_input]
    ).then(
        fn=lambda h: h,  # 更新state
        inputs=[chatbot],
        outputs=[chat_history_state]
    )

    # 支持Enter键提交
    question_input.submit(
        fn=process_and_update_chat,
        inputs=[question_input, kb_dropdown_chat, web_search_toggle, table_format_toggle, multi_hop_toggle,
                chat_history_state],
        outputs=[chatbot, status_box, search_results_output],
        queue=True
    ).then(
        fn=lambda: "",  # 清空输入框
        inputs=[],
        outputs=[question_input]
    ).then(
        fn=lambda h: h,  # 更新state
        inputs=[chatbot],
        outputs=[chat_history_state]
    )
