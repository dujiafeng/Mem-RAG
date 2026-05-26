"""对话历史管理服务。"""
from __future__ import annotations

import re
from typing import Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from sqlalchemy import create_engine, select, delete
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.core.logger import logger
from app.models.models import ChatMessage, ChatSession


class SessionService:
    """会话 / 历史消息管理。"""

    def __init__(self):
        settings = get_settings()
        self._engine = create_engine(settings.SYNC_DATABASE_URL)
        self._SessionLocal = sessionmaker(bind=self._engine)

    def get_session_id(self, session_uuid: str) -> int | None:
        """通过 session_uuid 获取内部 id。"""
        db = self._SessionLocal()
        try:
            res = db.execute(
                select(ChatSession).where(
                    ChatSession.session_uuid == session_uuid
                )
            )
            session = res.scalars().first()
            return session.id if session else None
        finally:
            db.close()

    def get_messages(self, session_uuid: str) -> list[BaseMessage]:
        """获取会话历史消息列表。

        规则：最近一条返回完整 output_uncode，
        其余返回 streamline_input 节省 token。
        """
        session_id = self.get_session_id(session_uuid)
        if not session_id:
            return []

        db = self._SessionLocal()
        try:
            result = db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.create_time)
            )
            msgs = result.scalars().all()
            if not msgs:
                return []

            history = []
            total = len(msgs)
            for i, msg in enumerate(msgs):
                history.append(HumanMessage(content=msg.user_input))

                if i == total - 1:
                    ai_content = msg.output_uncode or msg.raw_output
                else:
                    ai_content = (
                        msg.streamline_input
                        or msg.output_uncode
                        or msg.raw_output
                    )
                history.append(AIMessage(content=ai_content))

            return history
        finally:
            db.close()

    def clear_messages(self, session_uuid: str):
        """清除会话消息。"""
        session_id = self.get_session_id(session_uuid)
        if not session_id:
            return
        db = self._SessionLocal()
        try:
            db.execute(
                delete(ChatMessage).where(
                    ChatMessage.session_id == session_id
                )
            )
            db.commit()
        finally:
            db.close()

    @staticmethod
    def extract_clean_text(raw: str) -> str:
        """从 raw_output 中提取纯文本（去除代码块）。"""
        clean = re.sub(r"```.*?```", "", raw, flags=re.DOTALL)
        return clean.strip()

    @staticmethod
    def extract_code(raw: str) -> str:
        """从 raw_output 中提取代码块。"""
        codes = re.findall(
            r"```[a-zA-Z0-9+#]*\n(.*?)\n```", raw, re.DOTALL
        )
        return "\n---\n".join(codes) if codes else ""
