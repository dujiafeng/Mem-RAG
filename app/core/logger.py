"""结构化日志。"""

import logging
import uuid
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar

from app.core.config import get_settings

# 每个请求的 trace_id
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


class TraceIDFilter(logging.Filter):
    """向每条日志注入 trace_id。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id_var.get() or "-"
        return True


def get_logger(name: str = "rag_system") -> logging.Logger:
    settings = get_settings()
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger  # 避免重复添加

    # 文件处理器（UTF-8 编码，防止中文乱码）
    file_handler = RotatingFileHandler(
        settings.LOG_DIR / f"{name}.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式化
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(trace_id)s %(name)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    file_handler.addFilter(TraceIDFilter())
    console_handler.addFilter(TraceIDFilter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# 默认 logger
logger = get_logger()
