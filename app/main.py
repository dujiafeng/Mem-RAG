"""
应用入口 — App Factory + 路由注册。

替代旧的 api_service.py，使用分层架构：
  main.py → 路由注册 + 中间件 + 生命周期
  api/v1/ → 各模块路由
  services/ → 业务逻辑
  integrations/ → 外部服务适配
  db/ → 数据访问
"""
from __future__ import annotations

import os

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1 import auth, chat, sessions, documents
from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logger import logger
from app.db.postgres_client import get_db, async_engine
from app.models.models import Base, ChatSession, ChatMessage


def create_app() -> FastAPI:
    settings = get_settings()
    os.environ["DASHSCOPE_API_KEY"] = settings.DASHSCOPE_API_KEY

    app = FastAPI(
        title="Mem-RAG",
        description="本地知识库 RAG 系统",
        version="2.0.0",
    )

    # ── 中间件 ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── 注册路由 ──
    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(sessions.router)
    app.include_router(documents.router)

    # ── 向后兼容的旧路由别名 ──
    @app.get("/chat/{session_uuid}")
    async def legacy_get_history(
        session_uuid: str,
        current_user=Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
    ):
        """旧版 GET /chat/{uuid} → 新版 GET /sessions/{uuid}"""
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

    @app.delete("/delete/{session_uuid}")
    async def legacy_delete_session(
        session_uuid: str,
        current_user=Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
    ):
        """旧版 DELETE /delete/{uuid} → 新版 DELETE /sessions/{uuid}"""
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

    # ── 静态文件 ──
    project_root = settings.PROJECT_ROOT
    static_dir = project_root / "html"
    if static_dir.exists():
        app.mount("/html", StaticFiles(directory=str(static_dir)), name="html")

    # ── 异常处理器 ──
    register_exception_handlers(app)

    # ── 生命周期 ──
    @app.on_event("startup")
    async def startup():
        logger.info("[Startup] 初始化数据库表...")
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("[Startup] Mem-RAG v2 启动完成")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
