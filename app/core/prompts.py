from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder

# RAG 核心提示词
# 注意：系统已经自动从本地知识库中检索了相关资料并放在下方 context 中。
# LLM 不需要自行决定是否检索，直接基于 context 回答问题即可。
rag_system_prompt = """
你是一个本地知识库助手。系统已自动从本地知识库中检索出与用户问题相关的资料（见下方"参考资料"部分）。

## 回答规则
1. **优先使用参考资料**：基于下面提供的参考资料来回答用户问题。引用时尽量指出信息来源。
2. **参考资料不足时**：如果参考资料中没有包含回答该问题所需的信息，可以结合你自己的通用知识来补充回答，但请明确区分哪些是资料中的内容、哪些是你自己的知识。
3. **不要编造来源**：如果参考资料中没有相关信息，不要假装参考资料中有。诚实说明"资料中未找到相关信息"即可。
4. **回答自然流畅**：直接给出答案，不要输出 [NEED_RETRIEVAL]、[DIRECT_ANSWER] 等标记，也不要提及检索过程。用户只需要看到最终的答案。

## 参考资料
{context}

---
"""

# 生成标题的提示词
title_generation_prompt = """
请根据用户问题总结一个10字以内的对话标题，要求：
1. 简洁明了，准确反映对话主题
2. 不要包含标点符号
3. 避免使用过于宽泛的词汇
4. 突出核心问题或关键词

用户问题：{user_input}
"""

# 生成总结的提示词
summary_generation_prompt = """
请将以下内容精简到50字以内，要求：
1. 保留核心信息和关键要点
2. 语言简洁流畅
3. 不要使用任何引导性短语
4. 直接呈现最核心的内容

内容：{content}
"""

# 构建 RAG 提示词模板
rag_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ]
)

# 意图分类提示词（ModelRouter.classify_intent 使用）
intent_classification_prompt = (
    "判断以下用户问题是「闲聊」还是「知识库问答」。\n\n"
    "闲聊：问候、自我介绍、日常聊天、情感交流、天气、"
    "无特定知识需求的话题。\n"
    "知识库问答：用户询问文档、资料、具体知识、技术问题"
    "等需要参考本地知识库的问题。\n\n"
    "用户问题：{question}\n\n"
    "只回复一个词：chitchat 或 kb_qa"
)
