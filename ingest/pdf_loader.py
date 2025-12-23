import fitz  # PyMuPDF

# PDF文本提取
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            page_text = page.get_text()
            # 清理不可打印字符，尝试用 UTF-8 解码，失败时忽略非法字符
            text += page_text.encode('utf-8', errors='ignore').decode('utf-8')
        if not text.strip():
            print(f"警告：PDF文件 {pdf_path} 提取内容为空")
        return text
    except Exception as e:
        print(f"PDF文本提取失败：{str(e)}")
        return ""

