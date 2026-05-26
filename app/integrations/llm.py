"""LLM 封装（通义千问）。"""
from __future__ import annotations

from functools import lru_cache

from langchain_community.chat_models import ChatTongyi


@lru_cache()
def get_chat_model(model: str = "qwen-max") -> ChatTongyi:
    return ChatTongyi(model=model)


# 用于标题/总结等轻量任务
def get_lightweight_chat_model(model: str = "qwen-turbo") -> ChatTongyi:
    return ChatTongyi(model=model)
