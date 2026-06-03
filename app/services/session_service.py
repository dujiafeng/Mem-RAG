import re
from typing import Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from sqlalchemy import select, delete

from app.core.logger import logger
from app.db.db_client import SyncSessionLocal
from app.models.models import ChatMessage, ChatSession


class SessionService:
    """会话 / 历史消息管理。"""

    def get_session_id(self, session_uuid: str) -> int | None:
        db = SyncSessionLocal()
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
        session_id = self.get_session_id(session_uuid)
        if not session_id:
            return []
        db = SyncSessionLocal()
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
        session_id = self.get_session_id(session_uuid)
        if not session_id:
            return
        db = SyncSessionLocal()
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
        clean = re.sub(r"```.*?```", "", raw, flags=re.DOTALL)
        return clean.strip()

    @staticmethod
    def extract_code(raw: str) -> str:
        codes = re.findall(
            r"```[a-zA-Z0-9+#]*\n(.*?)\n```", raw, re.DOTALL
        )
        return "\n---\n".join(codes) if codes else ""
