"""对话路由：发消息、获取历史。"""
from __future__ import annotations

import asyncio
import io
from contextlib import redirect_stdout

from fastapi import APIRouter, Body, Cookie, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_rag_service, get_generation_service
from app.db.db_client import get_db
from app.models.models import ChatSession, ChatMessage, User
from app.services.rag_service import RAGService
from app.services.generation_service import GenerationService
from app.services.session_service import SessionService
from app.core.logger import logger

router = APIRouter(prefix="/chat", tags=["对话"])


@router.post("")
async def chat_stream(
    session_uuid: str = Body(..., embed=True),
    input_text: str = Body(..., embed=True),
    current_user: User = Depends(get_current_user),
    rag: RAGService = Depends(get_rag_service),
    gen: GenerationService = Depends(get_generation_service),
    db: AsyncSession = Depends(get_db),
):
    """流式聊天接口。"""
    # 校验 session 存在
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.session_uuid == session_uuid
        )
    )
    session = result.scalars().first()
    if not session:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="会话不存在")

    async def event_generator():
        yield "[状态] 正在处理您的问题...\n"
        await asyncio.sleep(0.3)

        yield "[状态] 正在检索相关资料...\n"
        await asyncio.sleep(0.3)

        yield "[状态] 正在生成回答...\n"
        await asyncio.sleep(0.1)

        stdout_capture = io.StringIO()
        full_out = ""

        # 在调用链之前注入 user_id（绕过 LangChain config 传播问题）
        rag.set_user_id(session_uuid, current_user.id)

        with redirect_stdout(stdout_capture):
            config = {
                "configurable": {
                    "session_id": session_uuid,
                    "user_id": current_user.id,
                }
            }
            async for chunk in rag.chain.astream(
                {"input": input_text}, config=config
            ):
                # 刷新捕获的 stdout
                captured = stdout_capture.getvalue()
                if captured:
                    yield captured
                    stdout_capture.truncate(0)
                    stdout_capture.seek(0)

                content = (
                    chunk
                    if isinstance(chunk, str)
                    else getattr(chunk, "content", "")
                )
                if content.startswith("[状态]"):
                    yield content
                    continue
                full_out += content
                yield content

        # 最后捕获的 stdout
        final = stdout_capture.getvalue()
        if final:
            yield final

        # 异步存储（不阻塞响应）
        asyncio.create_task(
            _save_after_chat(session.id, input_text, full_out, gen)
        )

    return StreamingResponse(event_generator(), media_type="text/plain")


async def _save_after_chat(
    session_id: int,
    user_input: str,
    raw_output: str,
    gen: GenerationService,
):
    """保存聊天记录到数据库。"""
    from app.db.db_client import AsyncSessionLocal
    from app.models.models import ChatMessage
    from sqlalchemy import func, update

    async with AsyncSessionLocal() as db:
        try:
            clean_text = SessionService.extract_clean_text(raw_output)
            code = SessionService.extract_code(raw_output)

            # 是否为第一条消息 → 生成标题
            count_res = await db.execute(
                select(func.count(ChatMessage.id)).where(
                    ChatMessage.session_id == session_id
                )
            )
            msg_count = count_res.scalar()

            if msg_count == 0:
                title = await gen.generate_title(user_input)
                await db.execute(
                    update(ChatSession)
                    .where(ChatSession.id == session_id)
                    .values(title=title)
                )

            summary = await gen.generate_summary(clean_text)

            db.add(
                ChatMessage(
                    session_id=session_id,
                    user_input=user_input,
                    raw_output=raw_output,
                    output_uncode=clean_text,
                    code=code,
                    streamline_input=summary,
                )
            )
            await db.commit()
            logger.info(f"[Backend] 会话 {session_id} 存储完成")
        except Exception as e:
            await db.rollback()
            logger.error(f"[Backend] 存储失败: {e}")
