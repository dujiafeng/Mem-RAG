from fastapi import APIRouter, Body, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_rag_service, get_generation_service
from app.db.db_client import get_db, AsyncSessionLocal
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

        rag.set_user_id(session_uuid, current_user.id)

        config = {
            "configurable": {
                "session_id": session_uuid,
                "user_id": current_user.id,
            }
        }
        full_out = ""

        async for chunk in rag.chain.astream(
            {"input": input_text}, config=config
        ):
            content = (
                chunk
                if isinstance(chunk, str)
                else getattr(chunk, "content", "")
            )
            full_out += content
            yield content

        import asyncio
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
    async with AsyncSessionLocal() as db:
        try:
            clean_text = SessionService.extract_clean_text(raw_output)
            code = SessionService.extract_code(raw_output)

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
