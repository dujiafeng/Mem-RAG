"""记忆 / 上下文管理服务（短期 / 长期）。
包含 SessionHistoryStore，实现 LangChain BaseChatMessageHistory 接口。"""
from __future__ import annotations

from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from sqlalchemy import create_engine, select, delete
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.core.logger import logger
from app.models.models import ChatSession, ChatMessage


class SessionHistoryStore(BaseChatMessageHistory):
    """实现 LangChain BaseChatMessageHistory 接口，从 MySQL 读写历史消息。

    注意：RunnableWithMessageHistory 需要这个接口，
    get_history() 回调必须返回 BaseChatMessageHistory 实例，而不是 list。
    """

    def __init__(self, session_uuid: str):
        self.session_uuid = session_uuid
        settings = get_settings()
        self._engine = create_engine(settings.SYNC_DATABASE_URL)
        self._SessionLocal = sessionmaker(bind=self._engine)

    # ── 内部方法 ──

    def _get_session_id(self) -> int | None:
        db = self._SessionLocal()
        try:
            res = db.execute(
                select(ChatSession).where(
                    ChatSession.session_uuid == self.session_uuid
                )
            )
            session = res.scalars().first()
            return session.id if session else None
        finally:
            db.close()

    # ── 历史消息（同步） ──

    @property
    def messages(self) -> Sequence[BaseMessage]:
        """从数据库加载历史消息。

        规则：最近一条返回完整 output_uncode，
        其余返回 streamline_input 节省 token。
        """
        session_id = self._get_session_id()
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

            history: list[BaseMessage] = []
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

    async def aget_messages(self) -> Sequence[BaseMessage]:
        """异步获取消息（实际基于同步实现）。"""
        return self.messages

    # ── 添加消息（同步） ──

    def add_message(self, message: BaseMessage) -> None:
        """向数据库追加一条消息。
        
        注意：当前实现仅支持 HumanMessage 和 AIMessage。
        对话存储由 chat.py 的 _save_after_chat 异步处理，
        这里仅保留接口兼容性。
        """
        logger.debug(
            f"[History] add_message called for "
            f"{self.session_uuid}: "
            f"{type(message).__name__}"
        )

    async def aadd_message(self, message: BaseMessage) -> None:
        """异步追加消息。"""
        self.add_message(message)

    # ── 清除消息 ──

    def clear(self) -> None:
        """清除当前会话的所有消息。"""
        session_id = self._get_session_id()
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

    async def aclear(self) -> None:
        """异步清除消息。"""
        self.clear()


class MemoryService:
    """记忆服务：管理对话上下文、token 预算等。"""

    MAX_HISTORY_ROUNDS = 6

    def __init__(self):
        pass

    def get_history_store(
        self, session_uuid: str
    ) -> SessionHistoryStore:
        """返回符合 LangChain 接口的历史记录对象。"""
        return SessionHistoryStore(session_uuid)

    def truncate_history(
        self,
        history: Sequence[BaseMessage],
        max_tokens: int = 2000,
    ) -> list[BaseMessage]:
        """截断历史消息使其不超过 max_tokens（粗略按字符估算）。"""
        result = []
        total_chars = 0
        for msg in reversed(history):
            chars = len(msg.content)
            if total_chars + chars > max_tokens * 4:
                break
            result.insert(0, msg)
            total_chars += chars
        return result
