"""对话相关 Pydantic Schema。"""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_uuid: str = Field(..., description="会话 UUID")
    input_text: str = Field(..., description="用户输入")


class ChatResponse(BaseModel):
    answer: str


class SessionCreateResponse(BaseModel):
    session_id: str
    title: str = "新对话"


class SessionItem(BaseModel):
    session_id: str
    title: str
    update_time: str


class HistoryItem(BaseModel):
    user_input: str
    raw_output: str
