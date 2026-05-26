"""RAG 主流程编排：检索 → 生成 → 存储。"""
from __future__ import annotations

from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory

from app.core.logger import logger
from app.core.prompts import rag_prompt_template
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService
from app.services.memory_service import MemoryService
from app.integrations.llm import get_chat_model


class RAGService:
    """RAG 主服务：编排检索、生成、历史管理。"""

    def __init__(self, model_name: str = "qwen-max"):
        self.retrieval = RetrievalService()
        self.generation = GenerationService(model_name=model_name)
        self.memory = MemoryService()
        self.model: ChatTongyi = get_chat_model(model_name)
        self._chain = self._build_chain()

    def _build_chain(self):
        """构建 LangChain 执行链。"""

        def retrieve_context(input_data: dict, run_config=None):
            query = input_data["input"]
            user_id = None
            if run_config and "configurable" in run_config:
                user_id = run_config["configurable"].get("user_id")
            logger.info(
                f"[RAG] 混合检索 query={query[:50]} user_id={user_id}"
            )
            docs = self.retrieval.hybrid_search(query, user_id=user_id)
            return self.retrieval.format_docs_for_prompt(docs)

        chain = (
            {
                "context": RunnableLambda(retrieve_context),
                "input": lambda x: x["input"],
                "history": lambda x: x.get("history", []),
            }
            | rag_prompt_template
            | self.model
            | StrOutputParser()
        )

        # 会话历史支持
        def get_history(session_uuid: str):
            return self.memory.get_recent_history(session_uuid)

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain

    @property
    def chain(self):
        return self._chain

    async def answer(
        self,
        question: str,
        session_uuid: str,
        user_id: int | None = None,
    ) -> tuple[str, list[dict]]:
        """便捷方法：检索 → 生成 → 返回 (answer, sources)。

        Args:
            question: 用户问题
            session_uuid: 会话 UUID
            user_id: 用户 ID（用于知识库隔离）

        Returns:
            (answer_text, sources_list)
        """
        config = {
            "configurable": {
                "session_id": session_uuid,
                "user_id": user_id,
            }
        }

        full_answer = ""
        sources = []

        async for chunk in self._chain.astream(
            {"input": question}, config=config
        ):
            content = chunk if isinstance(chunk, str) else getattr(chunk, "content", "")
            full_answer += content

        return full_answer, sources
