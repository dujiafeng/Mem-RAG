"""自定义异常类 + 全局异常处理器。"""

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette import status


class AppException(Exception):
    """业务异常基类。"""

    def __init__(self, status_code: int, detail: str, code: str = None):
        self.status_code = status_code
        self.detail = detail
        self.code = code or "app_error"


class NotFoundError(AppException):
    def __init__(self, detail: str = "资源不存在", code: str = "not_found"):
        super().__init__(status.HTTP_404_NOT_FOUND, detail, code)


class UnauthorizedError(AppException):
    def __init__(self, detail: str = "未登录或登录已过期", code: str = "unauthorized"):
        super().__init__(status.HTTP_401_UNAUTHORIZED, detail, code)


class ForbiddenError(AppException):
    def __init__(self, detail: str = "权限不足", code: str = "forbidden"):
        super().__init__(status.HTTP_403_FORBIDDEN, detail, code)


class BadRequestError(AppException):
    def __init__(self, detail: str = "请求参数错误", code: str = "bad_request"):
        super().__init__(status.HTTP_400_BAD_REQUEST, detail, code)


class LLMTimeoutError(AppException):
    def __init__(self, detail: str = "LLM 响应超时", code: str = "llm_timeout"):
        super().__init__(status.HTTP_504_GATEWAY_TIMEOUT, detail, code)


# ── FastAPI 全局异常处理器 ──


def register_exception_handlers(app):
    """在 FastAPI app 上注册全局异常处理器。"""

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": exc.code},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "detail": "服务器内部错误",
                "code": "internal_error",
            },
        )
