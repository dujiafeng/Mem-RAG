from __future__ import annotations

from fastapi import Cookie, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.db_client import get_db
from app.models.models import User
from app.core.security import verify_jwt_token
from app.services.rag_service import RAGService
from app.services.document_service import DocumentService
from app.services.generation_service import GenerationService
from app.services.session_service import SessionService

_rag_service: RAGService | None = None
_doc_service: DocumentService | None = None
_gen_service: GenerationService | None = None
_session_service: SessionService | None = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def get_document_service() -> DocumentService:
    global _doc_service
    if _doc_service is None:
        _doc_service = DocumentService()
    return _doc_service


def get_generation_service() -> GenerationService:
    global _gen_service
    if _gen_service is None:
        _gen_service = GenerationService()
    return _gen_service


def get_session_service() -> SessionService:
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service


async def get_current_user(
    session_id: str = Cookie(None),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not session_id:
        raise HTTPException(status_code=401, detail="未登录")
    payload = verify_jwt_token(session_id)
    if not payload:
        raise HTTPException(status_code=401, detail="无效会话")
    result = await db.execute(
        select(User).where(User.id == int(payload["sub"]))
    )
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user
