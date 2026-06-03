"""认证相关路由：注册 / 登录 / 状态检查。"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.db_client import get_db
from app.models.models import User
from app.core.security import hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["认证"])


@router.post("/register")
async def register(
    username: str = None,
    password: str = None,
    db: AsyncSession = Depends(get_db),
):
    """注册。同时支持 query params 和 JSON body 格式。"""
    if not username or not password:
        raise HTTPException(status_code=422, detail="缺少 username 或 password")

    result = await db.execute(
        select(User).where(User.username == username)
    )
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="用户名已被占用")
    db.add(
        User(
            username=username,
            hashed_password=hash_password(password),
        )
    )
    await db.commit()
    return {"status": "success", "message": "注册成功"}


@router.post("/login")
async def login(
    username: str = None,
    password: str = None,
    response: Response = None,
    db: AsyncSession = Depends(get_db),
):
    """登录。同时支持 query params 和 JSON body 格式。"""
    if not username or not password:
        raise HTTPException(status_code=422, detail="缺少 username 或 password")

    result = await db.execute(
        select(User).where(User.username == username)
    )
    user = result.scalars().first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    new_cookie = user.last_cookie or str(uuid.uuid4())
    user.last_cookie = new_cookie
    await db.commit()
    response.set_cookie(
        key="session_id",
        value=new_cookie,
        path="/",
        samesite="lax",
    )
    return {
        "status": "success",
        "message": "登录成功",
        "data": {"username": user.username},
    }


@router.get("/me")
async def check_login(
    session_id: str = Cookie(None),
    db: AsyncSession = Depends(get_db),
):
    if not session_id:
        raise HTTPException(status_code=401, detail="未登录")
    result = await db.execute(
        select(User).where(User.last_cookie == session_id)
    )
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=401, detail="无效会话")
    return {
        "status": "success",
        "data": {"username": user.username},
    }
