# 提示词配置迁移指南

## 概述

所有系统提示词已统一提取到 `config/prompts.py` 中，避免硬编码在各个子程序中。

## 文件结构

```
config/
├── prompts.py                    # 提示词配置文件（核心）
├── prompts_usage_example.py     # 使用示例
├── PROMPTS_MIGRATION_GUIDE.md   # 本迁移指南
└── configs.py                   # 添加了 default_domain 配置
```

## 使用方法

### 1. 导入提示词配置

```python
from config.prompts import Prompts, get_default_domain
```

### 2. 替换硬编码的提示词

#### 示例1：基础问答场景

**原来的代码：**
```python
system_prompt = "你是一名医疗专家。基于提供的背景信息回答用户的问题。"
if use_table_format:
    system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"
```

**替换后：**
```python
system_prompt = Prompts.get_answer_prompt(
    domain="medical",
    use_table_format=use_table_format,
    consider_history=False
)
```

#### 示例2：考虑对话历史的问答

**原来的代码：**
```python
system_prompt = "你是一名医疗专家。请考虑对话历史并回答用户的问题。"
if use_table_format:
    system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"
```

**替换后：**
```python
system_prompt = Prompts.get_answer_with_background_prompt(
    domain="medical",
    use_table_format=use_table_format,
    consider_history=True
)
```

#### 示例3：多跳推理分析

**原来的代码：**
```python
system_prompt = """
你是医疗信息检索的专家分析系统。
你的任务是分析检索到的信息块，识别缺失的内容，并提出有针对性的后续查询来填补信息缺口。

重点关注医疗领域知识，如:
- 疾病诊断和症状
- 治疗方法和药物
- 医学研究和临床试验
- 患者护理和康复
- 医疗法规和伦理
"""
```

**替换后：**
```python
system_prompt = Prompts.get_reasoning_analysis_prompt(domain="medical")
```

#### 示例4：Rerank 评估

**原来的代码：**
```python
system_prompt = """你是一个信息检索评估专家。你的任务是根据用户查询，评估每个文档的相关性。
...（很长的提示词）...
"""
```

**替换后：**
```python
system_prompt = Prompts.get_rerank_evaluation_prompt()
```

## 需要迁移的文件

以下文件包含硬编码的提示词，建议逐步迁移：

1. **web_ui/streaming_handler.py**
   - 第 84 行：`"你是一名半导体专家。请考虑对话历史并回答用户的问题。"`
   - 第 152 行：`"你是一名医疗专家。基于提供的背景信息和对话历史回答用户的问题。"`
   - 第 194 行：`"你是一名医疗专家，请整合网络搜索和本地知识库提供全面的解答。请考虑对话历史。"`

2. **rag/multi_hop_rag.py**
   - 第 99-109 行：多跳推理分析提示词
   - 第 202-208 行：合成答案提示词

3. **rag/pipeline.py**
   - 第 9 行：`"你是一名半导体专家。"`
   - 第 44 行：`"你是一名半导体专家。基于提供的背景信息回答用户的问题。"`

4. **rag/service.py**
   - 第 49 行：`"你是一名医疗专家，请整合网络搜索和本地知识库提供全面的解答。"`
   - 第 91 行：`"你是一名医疗专家。"`

5. **search/rerank.py**
   - 第 86-96 行：Rerank 评估提示词

6. **llm/answer_generator.py**
   - 第 5 行：默认提示词 `"你是一名专业半导体助手，请根据背景知识回答问题。"`

## 配置选项

### 默认领域配置

在 `config/configs.py` 中：
```python
default_domain = "medical"  # 或 "semiconductor"
```

### 使用默认领域

```python
from config.prompts import Prompts, get_default_domain

# 使用配置的默认领域
system_prompt = Prompts.get_answer_prompt(
    domain=get_default_domain(),
    use_table_format=False
)
```

## 支持的领域

- `"medical"` - 医疗领域
- `"semiconductor"` - 半导体领域
- `"general"` - 通用领域（默认回退）

## 可用的提示词方法

### 问答场景
- `Prompts.get_answer_prompt()` - 基础问答
- `Prompts.get_answer_with_background_prompt()` - 基于背景信息的问答
- `Prompts.get_combined_answer_prompt()` - 整合网络和本地知识库的问答

### 多跳推理场景
- `Prompts.get_reasoning_analysis_prompt()` - 推理分析
- `Prompts.get_synthesis_answer_prompt()` - 合成答案

### Rerank 场景
- `Prompts.get_rerank_evaluation_prompt()` - 相关性评估

### 工具方法
- `Prompts.get_table_format_instruction()` - 表格格式说明

## 迁移步骤

1. **逐步迁移**：不要一次性修改所有文件，建议逐个文件迁移
2. **测试验证**：每次迁移后测试功能是否正常
3. **保持兼容**：可以先保留旧代码，新代码使用新配置，逐步替换

## 优势

1. **集中管理**：所有提示词在一个文件中，易于维护
2. **灵活配置**：支持不同领域和场景的切换
3. **易于扩展**：添加新领域或场景只需修改配置文件
4. **避免重复**：消除代码中的重复提示词
5. **统一风格**：确保提示词风格一致

