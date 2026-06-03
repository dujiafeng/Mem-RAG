"""LLM 生成服务：调用模型、构建 prompt、管理超时。"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.core.prompts import (
    title_generation_prompt,
    summary_generation_prompt,
)
from app.integrations.llm import get_chat_model, get_lightweight_chat_model
from app.core.logger import logger


class GenerationService:
    """LLM 生成服务。"""

    def __init__(self, model_name: str | None = None):
        self.model: BaseChatModel = get_chat_model(model_name)
        self.lightweight: BaseChatModel = get_lightweight_chat_model()

    async def generate_title(self, user_input: str) -> str:
        """生成对话标题（10字以内）。"""
        try:
            msg = HumanMessage(
                content=title_generation_prompt.format(user_input=user_input)
            )
            resp = await self.lightweight.ainvoke([msg])
            title = (
                resp.content.strip()
                .replace("\u201c", "")
                .replace("\u201d", "")
                .replace("标题：", "")
            )
            return title[:20] or "新对话"
        except Exception as e:
            logger.error(f"[Title] 生成失败: {e}")
            return "新对话"

    async def generate_summary(self, text: str) -> str:
        """生成回复摘要（50字以内）。"""
        try:
            msg = HumanMessage(
                content=summary_generation_prompt.format(content=text)
            )
            resp = await self.lightweight.ainvoke([msg])
            return resp.content.strip()[:100]
        except Exception as e:
            logger.error(f"[Summary] 生成失败: {e}")
            return text[:50]
