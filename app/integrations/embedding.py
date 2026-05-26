"""Embedding 封装。"""
from __future__ import annotations

from functools import lru_cache

from langchain_community.embeddings import DashScopeEmbeddings

from app.core.config import get_settings


@lru_cache()
def get_embedding_model() -> DashScopeEmbeddings:
    """获取 Embedding 模型实例（单例缓存）。"""
    settings = get_settings()
    return DashScopeEmbeddings(model=settings.EMBEDDINGS_MODEL)
