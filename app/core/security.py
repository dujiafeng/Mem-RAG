"""密码哈希与 Cookie / JWT 工具函数。"""

import hashlib

from app.core.config import get_settings


def hash_password(password: str) -> str:
    """SHA-256 加盐哈希。"""
    settings = get_settings()
    return hashlib.sha256(
        (password + settings.SALT_SUFFIX).encode("utf-8")
    ).hexdigest()


def verify_password(plain: str, hashed: str) -> bool:
    return hash_password(plain) == hashed
