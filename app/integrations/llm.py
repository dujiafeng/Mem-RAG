from __future__ import annotations
from functools import lru_cache
from typing import Any, Iterator, AsyncIterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.prompt_values import ChatPromptValue

from app.core.config import get_settings
from app.core.prompts import intent_classification_prompt


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


def _resolve_provider_and_model(
    model: str | None = None,
    lightweight: bool = False,
) -> tuple[str, str]:
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
    provider, model_name = _resolve_provider_and_model(model)
    if provider == "deepseek":
        return _build_deepseek_model(model_name)
    return _build_qwen_model(model_name)


@lru_cache()
def get_lightweight_chat_model(
    model: str | None = None,
) -> BaseChatModel:
    provider, model_name = _resolve_provider_and_model(
        model, lightweight=True
    )
    if provider == "deepseek":
        return _build_deepseek_model(model_name)
    return _build_qwen_model(model_name)


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
    """双模型管理：意图分类 + 获取对应模型（懒初始化）。"""

    def __init__(self):
        self._qwen_model: BaseChatModel | None = None
        self._deepseek_model: BaseChatModel | None = None
        self._classifier: BaseChatModel | None = None
        self._settings = get_settings()

    @property
    def qwen_model(self) -> BaseChatModel:
        if self._qwen_model is None:
            self._qwen_model = _build_qwen_model(
                self._settings.QWEN_CHAT_MODEL
            )
        return self._qwen_model

    @property
    def deepseek_model(self) -> BaseChatModel:
        if self._deepseek_model is None:
            self._deepseek_model = _build_deepseek_model(
                self._settings.DEEPSEEK_CHAT_MODEL
            )
        return self._deepseek_model

    @property
    def classifier(self) -> BaseChatModel:
        if self._classifier is None:
            self._classifier = _build_qwen_model(
                self._settings.QWEN_LIGHTWEIGHT_MODEL
            )
        return self._classifier

    def classify_intent(self, question: str) -> str:
        q = question.strip()
        kb_score = sum(1 for kw in _KB_QA_KEYWORDS if kw in q)
        chat_score = sum(1 for kw in _CHITCHAT_KEYWORDS if kw in q)
        if chat_score >= 2 and kb_score == 0:
            return "chitchat"
        if kb_score >= 2 and chat_score == 0:
            return "kb_qa"
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
        if intent == "kb_qa":
            return self.deepseek_model
        return self.qwen_model


class DynamicModelRunnable(Runnable):
    """Runnable 包装器：一次请求只做一次意图分类，缓存结果供后续 chunk 复用。"""

    def __init__(self, router: ModelRouter):
        self.router = router

    def _extract_question(self, input: Any) -> str:
        if isinstance(input, ChatPromptValue):
            input = input.to_messages()
        for msg in reversed(input):
            if isinstance(msg, HumanMessage):
                return msg.content
        return ""

    def _select_model(self, input: Any) -> BaseChatModel:
        question = self._extract_question(input)
        intent = self.router.classify_intent(question)
        return self.router.get_model(intent)

    def invoke(self, input: Any, config: dict | None = None, **kwargs) -> Any:
        model = self._select_model(input)
        return model.invoke(input, config=config, **kwargs)

    async def ainvoke(
        self, input: Any, config: dict | None = None, **kwargs
    ) -> Any:
        model = self._select_model(input)
        return await model.ainvoke(input, config=config, **kwargs)

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
