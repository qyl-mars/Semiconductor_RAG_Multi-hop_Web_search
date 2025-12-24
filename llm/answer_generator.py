from typing import Optional
from .llm_client import DeepSeekClient

# 基本的回答生成
def generate_answer_from_deepseek(question: str, system_prompt: str = "你是一名专业半导体助手，请根据背景知识回答问题。", background_info: Optional[str] = None) -> str:
    deepseek_client = DeepSeekClient()
    user_prompt = f"问题：{question}"
    if background_info:
        user_prompt = f"背景知识：{background_info}\n\n{user_prompt}"
    try:
        answer = deepseek_client.generate_answer(system_prompt, user_prompt)
        return answer
    except Exception as e:
        return f"生成回答时出错：{str(e)}"