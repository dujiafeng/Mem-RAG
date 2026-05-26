"""LLM 封装：支持通义千问 / DeepSeek 两种后端，通过配置切换。"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import get_settings


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
    """根据配置和传参决定使用的供应商和模型名。

    Returns:
        (provider, model_name)
    """
    settings = get_settings()

    if model is not None:
        # 显式传了模型名 —— 用当前供应商
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
