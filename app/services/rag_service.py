"""RAG 主流程编排：检索 → 生成 → 存储。"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory

from app.core.logger import logger
from app.core.prompts import rag_prompt_template
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService
from app.services.memory_service import MemoryService
from app.integrations.llm import ModelRouter, DynamicModelRunnable

class RAGService:
    """RAG 主服务：编排检索、生成、历史管理。

    使用 DynamicModelRunnable 在运行时根据问题意图动态选择模型，
    等价于 deepagents 的 wrap_model_call 模式。
    """
    def __init__(self, model_name: str | None = None):
        self.retrieval = RetrievalService()
        self.generation = GenerationService(model_name=model_name)
        self.memory = MemoryService()
        # 双模型路由
        self.model_router = ModelRouter()
        # 动态模型 Runnable —— 运行时自动选择 qwen 或 deepseek
        self.routed_model = DynamicModelRunnable(self.model_router)
        # 临时存储每个 session 的 user_id（由 chat 端点调用前设置）
        self._session_user_map: dict[str, int] = {}
        self._chain = self._build_chain()

    def set_user_id(self, session_uuid: str, user_id: int):
        """为指定会话设置用户 ID（在调用链之前由 chat 端点设置）。"""
        self._session_user_map[session_uuid] = user_id

    def _get_user_id(self, session_uuid: str) -> int | None:
        return self._session_user_map.get(session_uuid)

    def _build_chain(self) -> RunnableWithMessageHistory:
        """构建单条 LangChain 执行链（模型由 DynamicModelRunnable 动态选择）。"""
        def retrieve_context(input_data: dict, run_config=None):
            query = input_data["input"]
            # 从 run_config 或本地映射表读取 user_id
            user_id = None
            if run_config and "configurable" in run_config:
                user_id = run_config["configurable"].get("user_id")
            if user_id is None:
                session_id = (
                    run_config.get("configurable", {}).get("session_id")
                    if run_config
                    else None
                )
                if session_id:
                    user_id = self._get_user_id(session_id)
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
            | self.routed_model          # ← 动态路由：等价于 wrap_model_call
            | StrOutputParser()
        )

        def get_history(session_uuid: str):
            return self.memory.get_history_store(session_uuid)

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
