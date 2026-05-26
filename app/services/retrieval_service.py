"""混合检索服务：向量 + BM25 + RRF 融合。"""
from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.config import get_settings
from app.db.milvus_client import MilvusClientWrapper
from app.integrations.embedding import get_embedding_model
from app.integrations.bm25 import BM25Engine
from app.core.logger import logger


class CustomMilvusRetriever(BaseRetriever):
    """符合 LangChain Retriever 接口的 Milvus 包装器。"""

    vector_service: "RetrievalService"
    k: int
    user_id: Optional[int] = None

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return self.vector_service.search_milvus(
            query, k=self.k, user_id=self.user_id
        )


class RRFRetriever(BaseRetriever):
    """RRF 倒数秩融合检索器。"""

    retrievers: list
    k: int = 60

    def _get_relevant_documents(self, query: str) -> list[Document]:
        all_results = []
        for i, retriever in enumerate(self.retrievers):
            results = retriever.invoke(query)
            all_results.extend(
                [(doc, i, rank) for rank, doc in enumerate(results, 1)]
            )

        doc_scores: dict = {}
        for doc, retriever_idx, rank in all_results:
            doc_id = str(hash(doc.page_content))
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0}
            doc_scores[doc_id]["score"] += 1 / (rank + self.k)

        sorted_items = sorted(
            doc_scores.values(), key=lambda x: x["score"], reverse=True
        )

        if not sorted_items:
            return []

        max_score = sorted_items[0]["score"]
        relevant_docs = []
        for item in sorted_items:
            normalized = (item["score"] / max_score) * 100
            if normalized >= 97:
                relevant_docs.append(item["doc"])
                if len(relevant_docs) >= 3:
                    break
        return relevant_docs


class RetrievalService:
    """检索服务：统一向量检索、全文检索、混合融合的入口。"""

    def __init__(self):
        self.settings = get_settings()
        self.milvus = MilvusClientWrapper()
        self.embedding = get_embedding_model()
        self.bm25_engine = BM25Engine()

    def search_milvus(
        self,
        query: str,
        k: int = 3,
        user_id: Optional[int] = None,
    ) -> list[Document]:
        """底层 Milvus 向量检索。"""
        client = self.milvus.client
        collection = self.settings.COLLECTION_NAME

        if not client.has_collection(collection):
            logger.info(f"[Milvus] 集合 {collection} 不存在")
            return []

        query_vector = self.embedding.embed_query(query)

        filter_expr = None
        if user_id is not None:
            filter_expr = f"user_id == {user_id} or is_shared == 1"
            logger.info(f"[Milvus] 过滤: {filter_expr}")

        try:
            params = {
                "collection_name": collection,
                "data": [query_vector],
                "limit": k,
                "output_fields": [
                    "text",
                    "filename",
                    "user_id",
                    "is_shared",
                ],
            }
            if filter_expr:
                params["filter"] = filter_expr

            res = client.search(**params)
            docs = []
            for hit in res[0]:
                doc = Document(
                    page_content=hit["entity"]["text"],
                    metadata={
                        "filename": hit["entity"]["filename"],
                        "score": hit["distance"],
                        "user_id": hit["entity"].get("user_id", 0),
                        "is_shared": hit["entity"].get("is_shared", 0),
                    },
                )
                docs.append(doc)
            return docs
        except Exception as e:
            logger.error(f"[Milvus] 搜索失败: {e}")
            return []

    def get_hybrid_retriever(
        self, user_id: Optional[int] = None
    ) -> BaseRetriever:
        """获取混合检索器（向量 + BM25 + RRF）。"""
        dense = CustomMilvusRetriever(
            vector_service=self,
            k=self.settings.SIMILARITY_THRESHOLD,
            user_id=user_id,
        )

        sparse = self.bm25_engine.get_retriever(
            top_k=self.settings.SIMILARITY_THRESHOLD
        )

        if sparse:
            logger.info("[Hybrid] 启动 RRF 倒数秩融合策略")
            return RRFRetriever(
                retrievers=[dense, sparse],
                k=60,
            )

        return dense

    def hybrid_search(
        self,
        query: str,
        user_id: Optional[int] = None,
    ) -> list[Document]:
        """执行混合搜索，返回文档列表。"""
        retriever = self.get_hybrid_retriever(user_id=user_id)
        return retriever.invoke(query)

    def format_docs_for_prompt(self, docs: list[Document]) -> str:
        """将 Document 列表格式化为 prompt 可用的文本。"""
        if not docs:
            return "无相关参考资料"
        parts = []
        for i, doc in enumerate(docs):
            content = (
                f"资料[{i + 1}]: {doc.page_content}\n"
                f"来源: {doc.metadata.get('filename', '未知')}"
            )
            parts.append(content)
        return "\n\n".join(parts)
