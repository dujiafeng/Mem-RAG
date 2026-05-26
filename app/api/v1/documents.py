"""文档上传 / 管理路由。"""
from __future__ import annotations

import os

from fastapi import APIRouter, Cookie, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_document_service
from app.db.postgres_client import get_db
from app.models.models import KnowledgeFile, User
from app.services.document_service import DocumentService

router = APIRouter(prefix="/kb", tags=["知识库"])

# 支持的文件扩展名
ALLOWED_EXTS = {".txt", ".pdf", ".docx", ".doc", ".pptx", ".xlsx"}


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    is_shared: str = Form("false"),
    current_user: User = Depends(get_current_user),
    doc_service: DocumentService = Depends(get_document_service),
    db: AsyncSession = Depends(get_db),
):
    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，支持的格式: {', '.join(ALLOWED_EXTS)}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="文件内容为空")

    share_flag = is_shared.lower() == "true"

    try:
        text = doc_service.extract_text(file_bytes, file.filename)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="文件编码错误，请使用 UTF-8")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文本提取失败: {e}")

    # MD5 去重
    import hashlib
    md5_hex = hashlib.md5(text.encode("utf-8")).hexdigest()

    # 检查 MySQL 去重
    from sqlalchemy import select
    result = await db.execute(
        select(KnowledgeFile).where(KnowledgeFile.md5 == md5_hex)
    )
    if result.scalars().first():
        return {"status": "skipped", "message": "【跳过】内容已在库中"}

    # 分块 + 写入
    chunk_count = doc_service.process_text(
        text=text,
        filename=file.filename,
        user_id=current_user.id,
        is_shared=share_flag,
        md5_hex=md5_hex,
    )

    # 保存文件记录到 MySQL（含原始内容，用于预览）
    db.add(
        KnowledgeFile(
            user_id=current_user.id,
            filename=file.filename,
            md5=md5_hex,
            chunk_count=chunk_count,
            is_shared=1 if share_flag else 0,
            content=text,
        )
    )
    await db.commit()

    return {"status": "success", "message": "【成功】内容已载入数据库"}


@router.get("/files")
async def list_files(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select
    result = await db.execute(
        select(KnowledgeFile).where(KnowledgeFile.user_id == current_user.id)
    )
    files = result.scalars().all()
    return {
        "status": "success",
        "data": [
            {
                "id": f.id,
                "filename": f.filename,
                "md5": f.md5,
                "is_shared": bool(f.is_shared),
                "chunk_count": f.chunk_count,
                "create_time": f.create_time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for f in files
        ],
    }


@router.get("/shared")
async def list_shared_files(db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    result = await db.execute(
        select(KnowledgeFile).where(KnowledgeFile.is_shared == 1)
    )
    files = result.scalars().all()
    return {
        "status": "success",
        "data": [
            {
                "id": f.id,
                "filename": f.filename,
                "md5": f.md5,
                "chunk_count": f.chunk_count,
            }
            for f in files
        ],
    }


@router.delete("/files/{file_id}")
async def delete_file(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select
    result = await db.execute(
        select(KnowledgeFile).where(
            KnowledgeFile.id == file_id,
            KnowledgeFile.user_id == current_user.id,
        )
    )
    f = result.scalars().first()
    if not f:
        raise HTTPException(status_code=400, detail="文件不存在或无权删除")
    await db.delete(f)
    await db.commit()
    return {"status": "success", "message": "文件已删除"}


@router.post("/files/{file_id}/share")
async def toggle_share(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select
    result = await db.execute(
        select(KnowledgeFile).where(
            KnowledgeFile.id == file_id,
            KnowledgeFile.user_id == current_user.id,
        )
    )
    f = result.scalars().first()
    if not f:
        raise HTTPException(status_code=400, detail="文件不存在或无权操作")
    f.is_shared = 0 if f.is_shared else 1
    await db.commit()
    return {
        "status": "success",
        "data": {"is_shared": bool(f.is_shared)},
    }


@router.get("/files/{file_id}/preview")
async def preview_file(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select
    result = await db.execute(
        select(KnowledgeFile).where(
            KnowledgeFile.id == file_id,
            KnowledgeFile.user_id == current_user.id,
        )
    )
    f = result.scalars().first()
    if not f:
        raise HTTPException(status_code=400, detail="文件不存在或无权访问")
    if not f.content:
        raise HTTPException(status_code=404, detail="预览内容不存在")
    return {"status": "success", "data": {"content": f.content[:2000]}}
