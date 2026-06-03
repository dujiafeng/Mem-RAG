import uuid

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.db_client import get_db
from app.models.models import User
from app.core.security import hash_password, verify_password, create_jwt_token

router = APIRouter(prefix="/auth", tags=["认证"])

COOKIE_MAX_AGE = 86400

def _set_jwt_cookie(response: Response, user: User):
    token = create_jwt_token(user.id, user.username)
    response.set_cookie(
        key="session_id",
        value=token,
        max_age=COOKIE_MAX_AGE,
        path="/",
        httponly=True,
        samesite="lax",
    )


@router.post("/register")
async def register(
    username: str = None,
    password: str = None,
    db: AsyncSession = Depends(get_db),
):
    if not username or not password:
        raise HTTPException(status_code=422, detail="缺少 username 或 password")
    result = await db.execute(
        select(User).where(User.username == username)
    )
    if result.scalars().first():
        raise HTTPException(status_code=409, detail="用户名已被占用")
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
    if not username or not password:
        raise HTTPException(status_code=422, detail="缺少 username 或 password")
    result = await db.execute(
        select(User).where(User.username == username)
    )
    user = result.scalars().first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    _set_jwt_cookie(response, user)
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
    from app.core.security import verify_jwt_token
    payload = verify_jwt_token(session_id)
    if not payload:
        raise HTTPException(status_code=401, detail="无效会话")
    result = await db.execute(
        select(User).where(User.id == int(payload["sub"]))
    )
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return {
        "status": "success",
        "data": {"username": user.username},
    }
