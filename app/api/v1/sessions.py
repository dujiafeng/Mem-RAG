"""会话管理路由。"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Cookie, Depends, HTTPException
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db.db_client import get_db
from app.models.models import ChatSession, ChatMessage, User
from app.models.chat import SessionCreateResponse, SessionItem, HistoryItem

router = APIRouter(prefix="/sessions", tags=["会话管理"])


@router.post("")
async def create_session(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    new_uuid = str(uuid.uuid4())
    db.add(
        ChatSession(session_uuid=new_uuid, user_id=current_user.id)
    )
    await db.commit()
    return {
        "status": "success",
        "data": {"session_id": new_uuid, "title": "新对话"},
    }


@router.get("")
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id)
        .order_by(desc(ChatSession.update_time))
    )
    return {
        "status": "success",
        "data": [
            {
                "session_id": s.session_uuid,
                "title": s.title,
                "update_time": s.update_time.strftime("%m-%d %H:%M"),
            }
            for s in result.scalars().all()
        ],
    }


@router.get("/{session_uuid}")
async def get_history(
    session_uuid: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.session_uuid == session_uuid
        )
    )
    session = result.scalars().first()
    if not session:
        raise HTTPException(status_code=404)

    msg_res = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.create_time)
    )
    return {
        "status": "success",
        "data": [
            {
                "user_input": m.user_input,
                "raw_output": m.raw_output,
            }
            for m in msg_res.scalars().all()
        ],
    }


@router.delete("/{session_uuid}")
async def delete_session(
    session_uuid: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.session_uuid == session_uuid
        )
    )
    session = result.scalars().first()
    if session:
        await db.delete(session)
        await db.commit()
    return {"status": "success"}
