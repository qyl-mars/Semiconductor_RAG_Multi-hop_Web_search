"""
提示词使用示例

展示如何在代码中使用 config/prompts.py 中的提示词配置
"""

from config.prompts import Prompts, get_default_domain


# ==================== 示例1：基础问答场景 ====================
def example_basic_answer():
    """基础问答场景"""
    # 使用默认领域（半导体）
    system_prompt = Prompts.get_answer_prompt(
        domain=get_default_domain(),
        use_table_format=False,
        consider_history=False
    )
    print("基础问答提示词:", system_prompt)
    # 输出: "你是一名半导体专家。基于提供的背景信息回答用户的问题。"


# ==================== 示例2：带表格格式的问答 ====================
def example_answer_with_table():
    """带表格格式的问答"""
    system_prompt = Prompts.get_answer_prompt(
        domain="semiconductor",
        use_table_format=True,
        consider_history=True
    )
    print("带表格格式的提示词:", system_prompt)
    # 输出: "你是一名半导体专家。请考虑对话历史并回答用户的问题。请尽可能以Markdown表格的形式呈现结构化信息。"


# ==================== 示例3：多跳推理场景 ====================
def example_reasoning_analysis():
    """多跳推理分析"""
    system_prompt = Prompts.get_reasoning_analysis_prompt(domain="semiconductor")
    print("推理分析提示词:", system_prompt[:100] + "...")


# ==================== 示例4：Rerank 场景 ====================
def example_rerank():
    """Rerank 评估"""
    system_prompt = Prompts.get_rerank_evaluation_prompt()
    print("Rerank 提示词:", system_prompt[:100] + "...")


# ==================== 示例5：在现有代码中替换硬编码提示词 ====================
def example_replace_hardcoded():
    """替换硬编码提示词的示例"""
    
    # 原来的硬编码方式：
    # system_prompt = "你是一名半导体专家。基于提供的背景信息和对话历史回答用户的问题。"
    # if use_table_format:
    #     system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"
    
    # 新的使用方式：
    use_table_format = True
    system_prompt = Prompts.get_answer_with_background_prompt(
        domain="semiconductor",
        use_table_format=use_table_format,
        consider_history=True
    )
    print("替换后的提示词:", system_prompt)


# ==================== 示例6：不同领域切换 ====================
def example_different_domains():
    """不同领域的提示词"""
    
    # 半导体领域
    semiconductor_prompt = Prompts.get_answer_prompt(domain="semiconductor")
    print("半导体领域:", semiconductor_prompt)
    
    # 通用领域
    general_prompt = Prompts.get_answer_prompt(domain="general")
    print("通用领域:", general_prompt)


if __name__ == "__main__":
    print("="*60)
    print("提示词使用示例")
    print("="*60)
    
    example_basic_answer()
    print()
    example_answer_with_table()
    print()
    example_reasoning_analysis()
    print()
    example_rerank()
    print()
    example_replace_hardcoded()
    print()
    example_different_domains()

