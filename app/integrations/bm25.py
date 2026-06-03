import os
import pickle
from typing import Optional

from langchain_community.retrievers import BM25Retriever

from app.core.config import get_settings
from app.core.logger import logger


class BM25Engine:
    """全文检索引擎（本地 BM25），支持按 md5 删除。"""

    def __init__(self):
        self.settings = get_settings()

    @property
    def corpus_path(self) -> str:
        return self.settings.BM25_CORPUS_PATH

    @property
    def meta_path(self) -> str:
        return self.corpus_path.replace(".pkl", "_meta.pkl")

    def load_corpus(self) -> list[str]:
        if not os.path.exists(self.corpus_path):
            return []
        try:
            with open(self.corpus_path, "rb") as f:
                corpus = pickle.load(f)
            return corpus if isinstance(corpus, list) else []
        except Exception as e:
            logger.error(f"[BM25] 加载语料失败: {e}")
            return []

    def save_corpus(self, corpus: list[str]):
        os.makedirs(os.path.dirname(self.corpus_path), exist_ok=True)
        with open(self.corpus_path, "wb") as f:
            pickle.dump(corpus, f)

    def load_meta(self) -> dict[str, list[int]]:
        if not os.path.exists(self.meta_path):
            return {}
        try:
            with open(self.meta_path, "rb") as f:
                meta = pickle.load(f)
            return meta if isinstance(meta, dict) else {}
        except Exception as e:
            logger.error(f"[BM25] 加载元数据失败: {e}")
            return {}

    def save_meta(self, meta: dict[str, list[int]]):
        os.makedirs(os.path.dirname(self.meta_path), exist_ok=True)
        with open(self.meta_path, "wb") as f:
            pickle.dump(meta, f)

    def get_retriever(
        self, top_k: int | None = None
    ) -> Optional[BM25Retriever]:
        corpus = self.load_corpus()
        if not corpus:
            return None
        r = BM25Retriever.from_texts(corpus)
        r.k = top_k or self.settings.SIMILARITY_THRESHOLD
        return r

    def extend_corpus(self, chunks: list[str], md5_hex: str = ""):
        corpus = self.load_corpus()
        meta = self.load_meta()
        start_idx = len(corpus)
        corpus.extend(chunks)
        self.save_corpus(corpus)
        if md5_hex:
            indices = list(range(start_idx, start_idx + len(chunks)))
            meta[md5_hex] = indices
            self.save_meta(meta)

    def delete_by_md5(self, md5_hex: str):
        meta = self.load_meta()
        indices = meta.pop(md5_hex, None)
        if indices is None:
            logger.info(f"[BM25] 未找到 md5={md5_hex} 的索引")
            return
        corpus = self.load_corpus()
        remove_set = set(indices)
        new_corpus = [
            t for i, t in enumerate(corpus) if i not in remove_set
        ]
        self.save_corpus(new_corpus)
        self.save_meta(meta)
        logger.info(
            f"[BM25] 已删除 md5={md5_hex[:8]}... "
            f"({len(indices)} 条, 剩余 {len(new_corpus)} 条)"
        )
