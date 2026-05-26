"""Milvus 客户端封装（单例模式）。"""
from __future__ import annotations

from typing import Optional

from pymilvus import MilvusClient

from app.core.config import get_settings
from app.core.logger import logger


class MilvusClientWrapper:
    """Milvus 连接管理，支持懒初始化和重连。"""

    _instance: Optional[MilvusClientWrapper] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_client"):
            return
        self._client: Optional[MilvusClient] = None
        self._connect()

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            self._connect()
        return self._client

    def _connect(self):
        settings = get_settings()
        grpc_options = {
            "grpc.keepalive_time_ms": 30000,
            "grpc.keepalive_timeout_ms": 10000,
            "grpc.keepalive_permit_without_calls": True,
            "grpc.http2.max_pings_without_data": 5,
            "grpc.http2.min_time_between_pings_ms": 5000,
        }
        try:
            self._client = MilvusClient(settings.MILVUS_URI, grpc_options=grpc_options)
            logger.info("[Milvus] 成功连接到数据库")
        except Exception as e:
            logger.error(f"[Milvus] 连接失败: {e}")
            logger.info("[Milvus] 尝试重新创建连接...")
            self._client = MilvusClient(settings.MILVUS_URI, grpc_options=grpc_options)
            logger.info("[Milvus] 成功重连")

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
            logger.info("[Milvus] 连接已关闭")
