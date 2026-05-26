"""BM25 / 关键词检索引擎（基于 pickle 缓存）。"""
from __future__ import annotations

import os
import pickle
from typing import Optional

from langchain_community.retrievers import BM25Retriever

from app.core.config import get_settings
from app.core.logger import logger


class BM25Engine:
    """全文检索引擎（本地 BM25），后续可替换为 Elasticsearch。"""

    def __init__(self):
        self.settings = get_settings()

    @property
    def corpus_path(self) -> str:
        return self.settings.BM25_CORPUS_PATH

    def load_corpus(self) -> list[str]:
        """从 pickle 加载语料库。"""
        if not os.path.exists(self.corpus_path):
            logger.info("[BM25] 语料文件不存在，返回空")
            return []
        try:
            with open(self.corpus_path, "rb") as f:
                corpus = pickle.load(f)
            return corpus if isinstance(corpus, list) else []
        except Exception as e:
            logger.error(f"[BM25] 加载语料失败: {e}")
            return []

    def save_corpus(self, corpus: list[str]):
        """持久化语料库。"""
        os.makedirs(os.path.dirname(self.corpus_path), exist_ok=True)
        with open(self.corpus_path, "wb") as f:
            pickle.dump(corpus, f)
        logger.info(f"[BM25] 语料已保存 ({len(corpus)} 条)")

    def get_retriever(
        self, top_k: int | None = None
    ) -> Optional[BM25Retriever]:
        """从语料构建 BM25Retriever。"""
        corpus = self.load_corpus()
        if not corpus:
            return None
        r = BM25Retriever.from_texts(corpus)
        r.k = top_k or self.settings.SIMILARITY_THRESHOLD
        return r

    def append_and_save(self, chunks: list[str]):
        """追加新分块并保存。"""
        corpus = self.load_corpus()
        corpus.extend(chunks)
        self.save_corpus(corpus)

    def extend_corpus(self, new_chunks: list[str]):
        """直接扩展内存中的 corpus 并持久化（用于 upload_by_str）。"""
        corpus = self.load_corpus()
        corpus.extend(new_chunks)
        self.save_corpus(corpus)
