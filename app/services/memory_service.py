from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from sqlalchemy import select, delete

from app.core.config import get_settings
from app.core.logger import logger
from app.db.db_client import sync_engine, async_engine
from app.models.models import ChatSession, ChatMessage
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession


class SessionHistoryStore(BaseChatMessageHistory):
    """实现 LangChain BaseChatMessageHistory 接口，从 MySQL 读写历史消息。"""

    def __init__(self, session_uuid: str):
        self.session_uuid = session_uuid
        settings = get_settings()
        self._SyncSessionLocal = sessionmaker(bind=sync_engine)

    def _get_session_id_sync(self) -> int | None:
        db = self._SyncSessionLocal()
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

    async def _get_session_id_async(self) -> int | None:
        async with AsyncSession(async_engine) as db:
            res = await db.execute(
                select(ChatSession).where(
                    ChatSession.session_uuid == self.session_uuid
                )
            )
            session = res.scalars().first()
            return session.id if session else None

    @property
    def messages(self) -> Sequence[BaseMessage]:
        session_id = self._get_session_id_sync()
        if not session_id:
            return []
        db = self._SyncSessionLocal()
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
        session_id = await self._get_session_id_async()
        if not session_id:
            return []
        async with AsyncSession(async_engine) as db:
            result = await db.execute(
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

    def add_message(self, message: BaseMessage) -> None:
        logger.debug(
            f"[History] add_message({self.session_uuid}): "
            f"{type(message).__name__}"
        )

    async def aadd_message(self, message: BaseMessage) -> None:
        logger.debug(
            f"[History] aadd_message({self.session_uuid}): "
            f"{type(message).__name__}"
        )

    def clear(self) -> None:
        session_id = self._get_session_id_sync()
        if not session_id:
            return
        db = self._SyncSessionLocal()
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
        session_id = await self._get_session_id_async()
        if not session_id:
            return
        async with AsyncSession(async_engine) as db:
            await db.execute(
                delete(ChatMessage).where(
                    ChatMessage.session_id == session_id
                )
            )
            await db.commit()


class MemoryService:
    """记忆服务：管理对话上下文。"""

    MAX_HISTORY_ROUNDS = 6

    def get_history_store(self, session_uuid: str) -> SessionHistoryStore:
        return SessionHistoryStore(session_uuid)

    def truncate_history(
        self,
        history: Sequence[BaseMessage],
        max_tokens: int = 2000,
    ) -> list[BaseMessage]:
        result = []
        total_chars = 0
        for msg in reversed(history):
            chars = len(msg.content)
            if total_chars + chars > max_tokens * 4:
                break
            result.insert(0, msg)
            total_chars += chars
        return result
