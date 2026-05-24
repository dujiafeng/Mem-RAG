import os
import pickle
import hashlib
import time
from datetime import datetime
from typing import Optional

# 核心导入
from pymilvus import connections, MilvusClient
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core import config_data as config
from app.core.logger import logger
from app.models.models import Base, KnowledgeFile

os.environ["DASHSCOPE_API_KEY"] = config.DASHSCOPE_API_KEY

# 同步数据库引擎（用于 check_md5 / save_md5 等非异步操作）
SYNC_DB_URL = config.ASYNC_DATABASE_URL.replace('mysql+aiomysql://', 'mysql+pymysql://')
sync_engine = create_engine(SYNC_DB_URL)
SessionLocal = sessionmaker(bind=sync_engine)


def get_string_md5(string):
    return hashlib.md5(string.encode('utf-8')).hexdigest()


def get_sync_session():
    return SessionLocal()


class KnowledgeBaseService:
    def __init__(self):
        logger.info("[System] 初始化 KnowledgeBaseService...")
        self.embeddings = DashScopeEmbeddings(model=config.EMBEDDINGS_MODEL)

        # 1. 显式初始化本地 Milvus 引擎
        logger.info(f"[Milvus] 正在启动本地引擎: {config.MILVUS_URI}")
        try:
            connections.connect(alias="default", uri=config.MILVUS_URI)
            time.sleep(2)
            logger.info("[Milvus] 引擎就绪。")
        except Exception as e:
            logger.error(f"[Error] 引擎启动失败: {e}")

        # 2. 初始化语义分割器
        logger.info("[Splitter] 加载语义分割器...")
        self.splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=config.BREAKPOINT_TYPE,
            buffer_size=config.BUFFER_SIZE
        )
        self.bm25_corpus = self._load_bm25_corpus()

    def _load_bm25_corpus(self):
        if os.path.exists(config.BM25_CORPUS_PATH):
            try:
                with open(config.BM25_CORPUS_PATH, 'rb') as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError) as e:
                logger.error(f"[Error] 加载 BM25 语料库失败: {e}")
                logger.info("[BM25] 使用空语料库")
                return []
        return []

    def _save_bm25_corpus(self):
        with open(config.BM25_CORPUS_PATH, 'wb') as f:
            pickle.dump(self.bm25_corpus, f)

    def check_md5(self, md5_str):
        """查询 MySQL 判断 MD5 是否已存在"""
        db = get_sync_session()
        try:
            existing = db.query(KnowledgeFile).filter(KnowledgeFile.md5 == md5_str).first()
            return existing is not None
        finally:
            db.close()

    def save_file_record(self, user_id: int, filename: str, md5: str, chunk_count: int, is_shared: bool = False):
        """将文件记录存入 MySQL"""
        db = get_sync_session()
        try:
            record = KnowledgeFile(
                user_id=user_id,
                filename=filename,
                md5=md5,
                chunk_count=chunk_count,
                is_shared=1 if is_shared else 0
            )
            db.add(record)
            db.commit()
            logger.info(f"[MySQL] 文件记录已保存: {filename}, md5={md5}")
            return record
        except Exception as e:
            db.rollback()
            logger.error(f"[MySQL] 保存文件记录失败: {e}")
            raise
        finally:
            db.close()

    def get_user_files(self, user_id: int):
        """获取用户上传的所有文件列表"""
        db = get_sync_session()
        try:
            records = db.query(KnowledgeFile).filter(
                KnowledgeFile.user_id == user_id
            ).order_by(KnowledgeFile.create_time.desc()).all()
            return [
                {
                    "id": r.id,
                    "filename": r.filename,
                    "md5": r.md5,
                    "is_shared": bool(r.is_shared),
                    "chunk_count": r.chunk_count,
                    "create_time": r.create_time.strftime("%Y-%m-%d %H:%M:%S") if r.create_time else ""
                }
                for r in records
            ]
        finally:
            db.close()

    def get_shared_files(self):
        """获取所有公开共享的知识库文件"""
        db = get_sync_session()
        try:
            records = db.query(KnowledgeFile).filter(
                KnowledgeFile.is_shared == 1
            ).order_by(KnowledgeFile.create_time.desc()).all()
            return [
                {
                    "id": r.id,
                    "filename": r.filename,
                    "md5": r.md5,
                    "chunk_count": r.chunk_count,
                    "create_time": r.create_time.strftime("%Y-%m-%d %H:%M:%S") if r.create_time else "",
                    "username": r.user.username if r.user else "未知"
                }
                for r in records
            ]
        finally:
            db.close()

    def toggle_share(self, file_id: int, user_id: int):
        """切换文件的共享状态（仅文件所有者可操作）"""
        db = get_sync_session()
        try:
            record = db.query(KnowledgeFile).filter(KnowledgeFile.id == file_id).first()
            if not record:
                return None, "文件不存在"
            if record.user_id != user_id:
                return None, "无权操作该文件"
            record.is_shared = 0 if record.is_shared else 1
            db.commit()
            return {"id": record.id, "is_shared": bool(record.is_shared)}, None
        finally:
            db.close()

    def delete_file(self, file_id: int, user_id: int):
        """删除文件记录（仅文件所有者可操作）"""
        db = get_sync_session()
        try:
            record = db.query(KnowledgeFile).filter(KnowledgeFile.id == file_id).first()
            if not record:
                return False, "文件不存在"
            if record.user_id != user_id:
                return False, "无权删除该文件"

            md5 = record.md5
            db.delete(record)
            db.commit()

            # 从 BM25 语料中移除该文件的数据
            # 注意：由于 BM25 存储的是纯文本块，无法精确按文件删除，
            # 我们记录已删除的 MD5，搜索时再过滤（简单方案）
            self._mark_bm25_deletion(md5)

            logger.info(f"[Delete] 文件记录已删除: {record.filename}, md5={md5}")
            return True, None
        except Exception as e:
            db.rollback()
            logger.error(f"[Delete] 删除文件记录失败: {e}")
            return False, str(e)
        finally:
            db.close()

    def _save_bm25_corpus(self):
        os.makedirs(os.path.dirname(config.BM25_CORPUS_PATH), exist_ok=True)
        with open(config.BM25_CORPUS_PATH, 'wb') as f:
            pickle.dump(self.bm25_corpus, f)

    def _mark_bm25_deletion(self, md5: str):
        """记录已删除文件的 MD5，搜索时排除"""
        deleted_dir = os.path.dirname(config.BM25_CORPUS_PATH)
        os.makedirs(deleted_dir, exist_ok=True)
        deleted_file = os.path.join(deleted_dir, "deleted_md5.txt")
        with open(deleted_file, 'a', encoding='utf-8') as f:
            f.write(md5 + '\n')

    def _load_deleted_md5s(self):
        deleted_file = os.path.join(os.path.dirname(config.BM25_CORPUS_PATH), "deleted_md5.txt")
        if os.path.exists(deleted_file):
            with open(deleted_file, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f.readlines())
        return set()

    def get_file_preview(self, file_id: int, user_id: int):
        """获取文件预览内容"""
        db = get_sync_session()
        try:
            record = db.query(KnowledgeFile).filter(KnowledgeFile.id == file_id).first()
            if not record:
                return None, "文件不存在"
            # 公开文件或自己的文件可预览
            if not record.is_shared and record.user_id != user_id:
                return None, "无权预览该文件"

            upload_dir = os.path.join(os.path.dirname(config.BM25_CORPUS_PATH), "uploads")
            file_path = os.path.join(upload_dir, f"{record.md5}.txt")
            if not os.path.exists(file_path):
                return None, "文件内容不存在"

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 截取前 2000 字符作为预览
            preview = content[:2000]
            if len(content) > 2000:
                preview += "\n\n...（内容过长，仅显示前 2000 字符）"

            return {
                "id": record.id,
                "filename": record.filename,
                "content": preview,
                "total_chars": len(content)
            }, None
        finally:
            db.close()

    def upload_by_str(self, data: str, filename: str, user_id: int = 0, is_shared: bool = False):
        """
        上传知识库内容
        :param data: 文本内容
        :param filename: 文件名
        :param user_id: 上传用户 ID
        :param is_shared: 是否公开共享
        """
        logger.info(f"\n[Process] 开始处理文件: {filename} (user_id={user_id})")
        md5_hex = get_string_md5(data)

        # MySQL 校验 MD5 去重
        if self.check_md5(md5_hex):
            return "【跳过】内容已在库中"

        # 1. 语义分割
        logger.info("[Semantic Split] 正在执行语义分割...")
        knowledge_chunks = self.splitter.split_text(data)

        # 2. 写入 Milvus（带上 user_id 和 is_shared 字段）
        logger.info("[Storage] 正在通过底层 Client 写入 Milvus...")
        try:
            client = MilvusClient(config.MILVUS_URI)

            # 生成向量
            logger.info("[Storage] 正在生成向量并自动获取维度...")
            vectors = self.embeddings.embed_documents(knowledge_chunks)
            actual_dim = len(vectors[0])
            logger.info(f"[Storage] 检测到模型输出维度为: {actual_dim}")

            # 如果表不存在，使用实际维度建表
            if not client.has_collection(config.COLLECTION_NAME):
                logger.info(f"[Storage] 创建集合: {config.COLLECTION_NAME}")
                client.create_collection(
                    collection_name=config.COLLECTION_NAME,
                    dimension=actual_dim,
                    auto_id=True,
                    enable_dynamic_field=True
                )

            # 构造数据插入（新增 user_id、is_shared 字段）
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data_to_insert = [
                {
                    "vector": v,
                    "text": t,
                    "filename": filename,
                    "create_time": cur_time,
                    "user_id": user_id,
                    "is_shared": 1 if is_shared else 0,
                    "md5": md5_hex
                }
                for v, t in zip(vectors, knowledge_chunks)
            ]

            client.insert(collection_name=config.COLLECTION_NAME, data=data_to_insert)
            client.close()
            logger.info(f"[Storage] 成功存入 {len(data_to_insert)} 条数据！")

        except Exception as e:
            logger.error(f"[Error] 写入失败: {e}")
            return f"【失败】{e}"

        # 3. 保存原始文件到磁盘（用于预览）
        upload_dir = os.path.join(os.path.dirname(config.BM25_CORPUS_PATH), "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"{md5_hex}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(data)
        logger.info(f"[Storage] 原始文件已保存: {file_path}")

        # 4. 写入 BM25
        self.bm25_corpus.extend(knowledge_chunks)
        self._save_bm25_corpus()

        # 5. 保存文件记录到 MySQL
        self.save_file_record(
            user_id=user_id,
            filename=filename,
            md5=md5_hex,
            chunk_count=len(knowledge_chunks),
            is_shared=is_shared
        )

        return "【成功】内容已载入数据库"


if __name__ == '__main__':
    service = KnowledgeBaseService()
    test_text = "周杰伦出生于1979年，代表作有《青花瓷》。"
    logger.info(service.upload_by_str(test_text, "jay_chou_test", user_id=1))
