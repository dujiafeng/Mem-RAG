"""LLM 封装：支持通义千问 / DeepSeek，新增 DynamicModelRunnable 根据问题类型动态选模型。"""
from __future__ import annotations
from functools import lru_cache
from typing import Any, Iterator, AsyncIterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.prompt_values import ChatPromptValue

from app.core.config import get_settings
from app.core.prompts import intent_classification_prompt

# ── 底层构建函数 ──────────────────────────────────
def _build_qwen_model(model_name: str) -> BaseChatModel:
    from langchain_community.chat_models import ChatTongyi
    return ChatTongyi(model=model_name)


def _build_deepseek_model(model_name: str) -> BaseChatModel:
    try:
        from langchain_deepseek import ChatDeepSeek
    except ImportError:
        raise ImportError(
            "使用 DeepSeek 需要安装 langchain-deepseek："
            " pip install langchain-deepseek"
        )
    settings = get_settings()
    return ChatDeepSeek(
        model=model_name,
        api_key=settings.DEEPSEEK_API_KEY,
        api_base=settings.DEEPSEEK_API_BASE,
    )

# ── 兼容旧接口（单供应商模式） ──
def _resolve_provider_and_model(
    model: str | None = None,
    lightweight: bool = False,
) -> tuple[str, str]:
    """根据配置和传参决定使用的供应商和模型名。

    Returns:
        (provider, model_name)
    """
    settings = get_settings()
    if model is not None:
        return settings.LLM_PROVIDER, model
    if settings.LLM_PROVIDER == "deepseek":
        return (
            "deepseek",
            settings.DEEPSEEK_LIGHTWEIGHT_MODEL
            if lightweight
            else settings.DEEPSEEK_CHAT_MODEL,
        )
    return (
        "qwen",
        settings.QWEN_LIGHTWEIGHT_MODEL
        if lightweight
        else settings.QWEN_CHAT_MODEL,
    )


@lru_cache()
def get_chat_model(model: str | None = None) -> BaseChatModel:
    """获取主对话模型（RAG 回答用）。"""
    provider, model_name = _resolve_provider_and_model(model)
    if provider == "deepseek":
        return _build_deepseek_model(model_name)
    return _build_qwen_model(model_name)


@lru_cache()
def get_lightweight_chat_model(
    model: str | None = None,
) -> BaseChatModel:
    """获取轻量模型（标题生成 / 摘要用）。"""
    provider, model_name = _resolve_provider_and_model(
        model, lightweight=True
    )
    if provider == "deepseek":
        return _build_deepseek_model(model_name)
    return _build_qwen_model(model_name)

# ── ModelRouter：意图分类 + 双模型管理 ──
_CHITCHAT_KEYWORDS = frozenset({
    "你好", "嗨", "hello", "hi", "你是谁", "你叫什么",
    "今天天气", "再见", "拜拜", "谢谢", "感谢",
    "哈哈", "不错", "好的", "最近怎么样", "在吗",
    "你吃饭了吗", "厉害", "牛", "棒", "晚安", "早上好",
})

_KB_QA_KEYWORDS = frozenset({
    "文档", "资料", "根据", "说明", "知识库", "文件",
    "文档中", "章节", "页面", "报告", "论文",
    "总结", "概括", "提炼", "归纳",
})
class ModelRouter:
    """双模型管理：意图分类 + 获取对应模型。

    用法::

        router = ModelRouter()
        intent = router.classify_intent("你好")  # → "chitchat"
        model  = router.get_model(intent)         # → qwen_model
    """

    def __init__(self):
        settings = get_settings()
        self.qwen_model = _build_qwen_model(settings.QWEN_CHAT_MODEL)
        self.deepseek_model = _build_deepseek_model(
            settings.DEEPSEEK_CHAT_MODEL
        )
        self.classifier = _build_qwen_model(
            settings.QWEN_LIGHTWEIGHT_MODEL
        )

    def classify_intent(self, question: str) -> str:
        """分类问题意图，返回 ``'chitchat'`` 或 ``'kb_qa'``。

        两级策略：关键词快检 → 轻量 LLM 兜底。
        """
        q = question.strip()
        # 一级：关键词计分
        kb_score = sum(1 for kw in _KB_QA_KEYWORDS if kw in q)
        chat_score = sum(1 for kw in _CHITCHAT_KEYWORDS if kw in q)
        if chat_score >= 2 and kb_score == 0:
            return "chitchat"
        if kb_score >= 2 and chat_score == 0:
            return "kb_qa"
        # 二级：LLM 兜底
        prompt = intent_classification_prompt.format(question=q)
        try:
            resp = self.classifier.invoke(
                [HumanMessage(content=prompt)]
            )
            result = resp.content.strip().lower()
            if "kb_qa" in result:
                return "kb_qa"
        except Exception:
            pass
        return "chitchat"

    def get_model(self, intent: str) -> BaseChatModel:
        """根据意图返回对应模型。"""
        if intent == "kb_qa":
            return self.deepseek_model
        return self.qwen_model

# ── DynamicModelRunnable：拦截模型调用，运行时动态选模型 ──
# 这个 Runnable 等价于 deepagents 中的 wrap_model_call 模式：
# 在模型调用前拦截，根据输入动态决定使用哪个模型实例。
class DynamicModelRunnable(Runnable):
    """Runnable 包装器：拦截 Model 调用，根据消息内容动态路由到 qwen 或 deepseek。

    放在链中替换 ``| model |`` 的位置，保持 streaming / invoke 与普通模型一致。
    """

    def __init__(self, router: ModelRouter):
        self.router = router

    def _extract_question(self, input: Any) -> str:
        """从 PromptValue 或 message list 中提取最后一个用户问题。"""
        if isinstance(input, ChatPromptValue):
            input = input.to_messages()
        for msg in reversed(input):
            if isinstance(msg, HumanMessage):
                return msg.content
        return ""

    def _select_model(self, input: Any) -> BaseChatModel:
        """根据输入选择模型。"""
        question = self._extract_question(input)
        intent = self.router.classify_intent(question)
        model = self.router.get_model(intent)
        return model

    # ── invoke ──
    def invoke(self, input: Any, config: dict | None = None, **kwargs) -> Any:
        model = self._select_model(input)
        return model.invoke(input, config=config, **kwargs)

    async def ainvoke(
        self, input: Any, config: dict | None = None, **kwargs
    ) -> Any:
        model = self._select_model(input)
        return await model.ainvoke(input, config=config, **kwargs)

    # ── stream ──
    def stream(
        self, input: Any, config: dict | None = None, **kwargs
    ) -> Iterator:
        model = self._select_model(input)
        yield from model.stream(input, config=config, **kwargs)

    async def astream(
        self, input: Any, config: dict | None = None, **kwargs
    ) -> AsyncIterator:
        model = self._select_model(input)
        async for chunk in model.astream(input, config=config, **kwargs):
            yield chunk
