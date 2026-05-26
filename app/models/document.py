"""文档相关 Pydantic Schema。"""
from pydantic import BaseModel, Field
from typing import Optional


class FileInfo(BaseModel):
    id: int
    filename: str
    md5: str
    is_shared: bool
    chunk_count: int
    create_time: Optional[str] = None


class UploadResponse(BaseModel):
    status: str
    message: str
    file_id: Optional[int] = None
