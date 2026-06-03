"""文档处理服务：解析、分块、入库。"""
from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from datetime import datetime
from typing import Optional

from pymilvus import MilvusClient

from app.core.config import get_settings
from app.core.exceptions import BadRequestError
from app.core.logger import logger
from app.core.text_splitter import SmartTextSplitter
from app.db.milvus_client import MilvusClientWrapper
from app.integrations.bm25 import BM25Engine
from app.integrations.embedding import get_embedding_model


class DocumentExtractionError(BadRequestError):
    """文档提取失败。"""
    pass


class DocumentService:
    """文档上传、解析、分块、入库。"""

    def __init__(self):
        self.settings = get_settings()
        self.splitter = SmartTextSplitter(
            embeddings=get_embedding_model(),
            chunk_size=self.settings.MAX_SPLIT_CHAR_NUMBER,
        )
        self.bm25_engine = BM25Engine()

    # ── 文件内容提取 ──

    @staticmethod
    def extract_text(file_bytes: bytes, filename: str) -> str:
        """从文件中提取纯文本。"""
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".txt":
            return file_bytes.decode("utf-8")

        supported = {".pdf", ".docx", ".doc", ".pptx", ".xlsx"}
        if ext not in supported:
            raise ValueError(
                f"不支持的文件格式: {ext}，"
                f"支持的格式: txt, pdf, docx, doc, pptx, xlsx"
            )

        from docling.document_converter import DocumentConverter

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            converter = DocumentConverter()
            result = converter.convert(tmp_path)
            text = result.document.export_to_text()

            if not text.strip():
                raise ValueError(
                    f"docling 未能从 {filename} 提取到任何文本内容"
                )

            text = unicodedata.normalize("NFKC", text)
            text = "".join(
                c
                if unicodedata.category(c) != "Cc" or c in "\n\r\t"
                else " "
                for c in text
            )
            return text
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ── 文本分块 + 写入 Milvus + BM25 ──

    def process_text(
        self,
        text: str,
        filename: str,
        user_id: int = 0,
        is_shared: bool = False,
        md5_hex: str = "",
    ) -> int:
        """处理文本：分块 -> 向量化 -> 写入 Milvus -> 更新 BM25。

        Returns:
            写入的块数。
        """
        settings = self.settings

        # 1. 分块
        logger.info("[Splitter] 正在智能分块...")
        chunks = self.splitter.split_text(text)
        if not chunks:
            logger.warning("[Splitter] 分块结果为空")
            return 0

        # 2. 写入 Milvus
        logger.info("[Storage] 写入 Milvus...")
        client = MilvusClient(settings.MILVUS_URI)
        try:
            embeddings = get_embedding_model()
            vectors = embeddings.embed_documents(chunks)
            actual_dim = len(vectors[0])

            if not client.has_collection(settings.COLLECTION_NAME):
                client.create_collection(
                    collection_name=settings.COLLECTION_NAME,
                    dimension=actual_dim,
                    auto_id=True,
                    enable_dynamic_field=True,
                )

            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data = [
                {
                    "vector": v,
                    "text": t,
                    "filename": filename,
                    "create_time": cur_time,
                    "user_id": user_id,
                    "is_shared": 1 if is_shared else 0,
                    "md5": md5_hex,
                }
                for v, t in zip(vectors, chunks)
            ]

            # JSON 序列化归一化（规避 gRPC latin-1 编码问题）
            data = json.loads(json.dumps(data, ensure_ascii=False))

            client.insert(
                collection_name=settings.COLLECTION_NAME, data=data
            )
            logger.info(f"[Storage] 成功存入 {len(data)} 条数据")
        finally:
            client.close()

        # 3. 追加到 BM25
        self.bm25_engine.extend_corpus(chunks, md5_hex=md5_hex)

        return len(chunks)
