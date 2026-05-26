"""记忆 / 上下文管理服务（短期 / 长期）。"""
from __future__ import annotations

from typing import Sequence

from langchain_core.messages import BaseMessage

from app.services.session_service import SessionService
from app.core.logger import logger


class MemoryService:
    """记忆服务：管理对话上下文、token 预算等。"""

    # 最大历史对话轮数
    MAX_HISTORY_ROUNDS = 6

    def __init__(self):
        self.session_svc = SessionService()

    def get_recent_history(
        self, session_uuid: str
    ) -> list[BaseMessage]:
        """获取最近 N 轮对话历史。"""
        all_msgs = self.session_svc.get_messages(session_uuid)
        # 每轮有 user+ai 两条消息
        max_msgs = self.MAX_HISTORY_ROUNDS * 2
        if len(all_msgs) > max_msgs:
            return all_msgs[-max_msgs:]
        return all_msgs

    def truncate_history(
        self, history: Sequence[BaseMessage], max_tokens: int = 2000
    ) -> list[BaseMessage]:
        """截断历史消息使其不超过 max_tokens（粗略按字符估算）。"""
        result = []
        total_chars = 0
        for msg in reversed(history):
            chars = len(msg.content)
            if total_chars + chars > max_tokens * 4:  # 中文字符~4/token
                break
            result.insert(0, msg)
            total_chars += chars
        return result
