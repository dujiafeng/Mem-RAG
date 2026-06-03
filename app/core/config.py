"""Pydantic Settings — 集中管理所有配置，支持环境变量覆盖。"""
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # ── LLM 供应商选择 ──
    LLM_PROVIDER: str = "qwen"

    # ── DashScope (通义千问) ──
    DASHSCOPE_API_KEY: str = ""
    QWEN_CHAT_MODEL: str = "qwen-max"
    QWEN_LIGHTWEIGHT_MODEL: str = "qwen-turbo"

    # ── DeepSeek ──
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_API_BASE: str = "https://api.deepseek.com"
    DEEPSEEK_CHAT_MODEL: str = "deepseek-chat"
    DEEPSEEK_LIGHTWEIGHT_MODEL: str = "deepseek-chat"

    EMBEDDINGS_MODEL: str = "text-embedding-v4"

    # ── Milvus ──
    MILVUS_URI: str = "http://localhost:19530"
    COLLECTION_NAME: str = "rag_collection"

    # ── 语义分割 ──
    BREAKPOINT_TYPE: str = "percentile"
    BUFFER_SIZE: int = 1

    # ── 混合检索 ──
    BM25_CORPUS_PATH: str = "./database/bm25_corpus.pkl"
    SIMILARITY_THRESHOLD: int = 3
    DENSE_WEIGHT: float = 0.7
    SPARSE_WEIGHT: float = 0.3
    RRF_K: int = 60
    RRF_SCORE_THRESHOLD: float = 97.0

    # ── 文本限制 ──
    MAX_SPLIT_CHAR_NUMBER: int = 1000

    # ── 数据库 ──
    DB_HOST: str = "localhost"
    DB_PORT: str = "3306"
    DB_USER: str = "root"
    DB_PASSWORD: str = ""
    DB_NAME: str = "mem_rag"

    # ── JWT ──
    JWT_SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440

    # ── 路径 ──
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
    LOG_DIR: Path = PROJECT_ROOT / "logs"

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        """从单独的 DB_* 字段构造异步连接 URL。"""
        return (
            f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?charset=utf8"
        )

    @property
    def SYNC_DATABASE_URL(self) -> str:
        """从异步 URL 推导同步 URL。"""
        return self.ASYNC_DATABASE_URL.replace(
            "mysql+aiomysql://", "mysql+pymysql://"
        )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    if not settings.JWT_SECRET_KEY:
        import secrets
        settings.JWT_SECRET_KEY = secrets.token_hex(32)
    if not settings.DASHSCOPE_API_KEY and not settings.DEEPSEEK_API_KEY:
        import warnings
        warnings.warn("未配置任何 API Key (DASHSCOPE_API_KEY / DEEPSEEK_API_KEY)")
    return settings
