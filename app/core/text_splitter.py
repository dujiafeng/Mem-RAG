"""
文本智能分块器 (SmartTextSplitter)

提供多种分块策略，根据文本内容特征自动选择最合适的分块方法。
支持：语义分割、递归字符分割、段落分割、句子分割、固定大小分割，
以及结构化文本：JSON 分割、HTML 分割、Markdown 分割。

用法:
    splitter = SmartTextSplitter(embeddings=embeddings)
    chunks = splitter.split_text(long_text)

    # 或强制指定策略
    chunks = splitter.split_with_method(long_text, method="json")
"""

import re
import json
import math
from typing import Optional, Any

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker

from app.core.logger import logger


class SmartTextSplitter:
    """智能文本分块器，根据文本内容自动选择最佳分块策略。"""

    # ──────────────────────────── 内置分块方法 ────────────────────────────

    METHOD_SEMANTIC = "semantic"       # 语义分割（依赖 embedding 模型）
    METHOD_RECURSIVE = "recursive"     # 递归字符分割（支持多级分隔符）
    METHOD_PARAGRAPH = "paragraph"     # 段落分割（按双换行）
    METHOD_SENTENCE = "sentence"       # 句子级分割（按句号/问号/感叹号）
    METHOD_FIXED = "fixed"             # 固定长度分割（字符数）
    METHOD_JSON = "json"               # JSON 结构化分割
    METHOD_HTML = "html"               # HTML 语义块分割
    METHOD_MARKDOWN = "markdown"       # Markdown 标题层级分割

    # ──────────────────────────── 构造 ────────────────────────────

    def __init__(
        self,
        embeddings: Any,
        *,
        # 通用参数
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        # 语义分割参数
        breakpoint_threshold_type: str = "percentile",
        buffer_size: int = 1,
        # 递归分割参数
        recursive_separators: Optional[list[str]] = None,
        # 自动检测阈值
        min_paragraph_chars: int = 100,
        min_semantic_chars: int = 500,
        # 中文/英文偏好
        prefer_semantic_length: int = 3000,
        # JSON 分割参数
        json_max_items_per_chunk: int = 50,
        json_max_depth: int = 5,
        # HTML/Markdown 分割参数
        html_preserve_context: bool = True,
        markdown_heading_level: int = 2,
    ):
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.buffer_size = buffer_size
        self.min_paragraph_chars = min_paragraph_chars
        self.min_semantic_chars = min_semantic_chars
        self.prefer_semantic_length = prefer_semantic_length
        self.json_max_items_per_chunk = json_max_items_per_chunk
        self.json_max_depth = json_max_depth
        self.html_preserve_context = html_preserve_context
        self.markdown_heading_level = markdown_heading_level

        # 递归分割默认分隔符
        self.recursive_separators = recursive_separators or [
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            ".",
            "!",
            "?",
            "；",
            ";",
            "，",
            ",",
            " ",
            "",
        ]

        # 缓存懒初始化的 splitter 实例
        self._semantic_splitter: Optional[SemanticChunker] = None
        self._recursive_splitter: Optional[RecursiveCharacterTextSplitter] = None

    # ──────────────────────────── 公共入口 ────────────────────────────

    def split_text(self, text: str) -> list[str]:
        """
        根据文本内容自动选择最佳分块策略。

        判断逻辑（按优先级）：
          1. 文本很短（< min_paragraph_chars）→ FIXED
          2. 检测到 JSON 结构 → JSON
          3. 检测到 HTML 结构 → HTML
          4. 检测到 Markdown 标题 → MARKDOWN
          5. 包含明显代码特征 → RECURSIVE
          6. 有清晰段落结构（双换行多）→ PARAGRAPH
          7. 文本较长且语义连贯 → SEMANTIC
          8. 句子特征明显 → SENTENCE
          9. 默认回退 → RECURSIVE
        """
        method = self._detect_method(text)
        logger.info(
            f"[Splitter] 自动选择分块策略: {method}  "
            f"(文本长度={len(text)})"
        )
        return self.split_with_method(text, method)

    def split_with_method(self, text: str, method: str) -> list[str]:
        """使用指定的分块方法对文本进行分割。"""
        method_map = {
            self.METHOD_SEMANTIC: self._semantic_split,
            self.METHOD_RECURSIVE: self._recursive_split,
            self.METHOD_PARAGRAPH: self._paragraph_split,
            self.METHOD_SENTENCE: self._sentence_split,
            self.METHOD_FIXED: self._fixed_split,
            self.METHOD_JSON: self._json_split,
            self.METHOD_HTML: self._html_split,
            self.METHOD_MARKDOWN: self._markdown_split,
        }

        splitter_fn = method_map.get(method)
        if splitter_fn is None:
            logger.warning(
                f"[Splitter] 未知的分块方法 '{method}'，回退到 recursive"
            )
            splitter_fn = self._recursive_split

        return splitter_fn(text)

    # ──────────────────────────── 自动检测策略 ────────────────────────────

    def detect_method(self, text: str) -> str:
        """公开的检测方法，供外部调用。"""
        return self._detect_method(text)

    def _detect_method(self, text: str) -> str:
        length = len(text)
        stripped = text.strip()

        # 1. 非常短的文本 -> 固定大小
        if length < self.min_paragraph_chars:
            return self.METHOD_FIXED

        # 2. JSON 结构检测
        if self._looks_like_json(stripped):
            return self.METHOD_JSON

        # 3. HTML 结构检测
        if self._looks_like_html(stripped):
            return self.METHOD_HTML

        # 4. Markdown 标题检测
        if self._looks_like_markdown(stripped):
            return self.METHOD_MARKDOWN

        # 5. 代码特征检测
        if self._has_code_features(text):
            return self.METHOD_RECURSIVE

        # 6. 段落结构检测
        para_count = len(re.findall(r"\n\s*\n", text))
        para_density = para_count / max(length, 1) * 10000

        # 7. 句子密度
        sentence_count = len(self._split_sentences(text))
        sentence_density = sentence_count / max(length, 1) * 10000

        if para_density > 5.0 and length > self.min_paragraph_chars * 2:
            return self.METHOD_PARAGRAPH

        if length >= self.prefer_semantic_length:
            return self.METHOD_SEMANTIC

        if sentence_density > 8.0:
            return self.METHOD_SENTENCE

        return self.METHOD_RECURSIVE

    # ──────────────────────────── 格式特征检测 ────────────────────────────

    @staticmethod
    def _looks_like_json(text: str) -> bool:
        """检测文本是否看起来像 JSON（尝试解析，不抛异常）。"""
        stripped = text.strip()
        if not (stripped.startswith("{") or stripped.startswith("[")):
            return False
        # 快速关键特征：包含 JSON 典型的键值对结构
        if re.search(r'"[^"]+"\s*:', stripped[:500]):
            return True
        # 尝试完整解析（防御性）
        try:
            json.loads(stripped[:5000])
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    _HTML_DETECT_PATTERNS = [
        r"(?i)<!DOCTYPE\s+html",
        r"(?i)<html[\s>]",
        r"(?i)<head[\s>]",
        r"(?i)<body[\s>]",
        r"<(div|section|article|header|footer|nav|main|aside)[\s>]",
        r"<(table|tr|td|th)[\s>]",
        r"<(form|input|select|button)[\s>]",
    ]

    @classmethod
    def _looks_like_html(cls, text: str) -> bool:
        """检测文本是否包含 HTML 结构。"""
        score = 0
        for pattern in cls._HTML_DETECT_PATTERNS:
            if re.search(pattern, text):
                score += 1
                if score >= 2:  # 两个以上特征才判定
                    return True
        return False

    _MARKDOWN_DETECT_PATTERNS = [
        r"^#{1,6}\s+\S",               # ATX 标题
        r"(?m)^={3,}\s*$",             # Setext 标题 level 1
        r"(?m)^-{3,}\s*$",             # Setext 标题 level 2
        r"```[\s\S]*?```",             # 代码块
        r"\|[^|]+\|[^|]+\|",           # 表格行
        r"(?m)^[-*+]\s+",              # 无序列表
        r"(?m)^\d+\.\s+",              # 有序列表
        r"\[.+?\]\(.+?\)",             # 链接
        r"!\[.*?\]\(.+?\)",            # 图片
        r"^>\s+",                      # 引用块
        r"\*\*.*?\*\*|__.*?__",        # 粗体
        r"\*.*?\*|_.*?_",              # 斜体
    ]

    @classmethod
    def _looks_like_markdown(cls, text: str) -> bool:
        """检测文本是否包含 Markdown 格式特征。"""
        score = 0
        for pattern in cls._MARKDOWN_DETECT_PATTERNS:
            if re.search(pattern, text, re.MULTILINE):
                score += 1
                if score >= 2:
                    return True
        return False

    # ──────────────────────────── 代码特征检测 ────────────────────────────

    _CODE_PATTERNS = [
        r"def\s+\w+\s*\(",           # Python function
        r"class\s+\w+",              # class definition
        r"import\s+\w+",             # import statement
        r"function\s+\w+\s*\(",      # JS/TS function
        r"const\s+\w+\s*=",          # JS const
        r"#\s*include",              # C include
        r"\{[\s\S]*?\}",             # brace blocks (common in code)
        r"\w+\s*:\s*\w+\s*\{",       # CSS/JSON-like
        r"@app\.\w+|@route",         # Flask/FastAPI decorators
        r"print\s*\(|console\.log",  # debug prints
        r"for\s+\w+\s+in\s+",       # loops
        r"if\s+__name__\s*==",       # Python main guard
    ]

    @staticmethod
    def _has_code_features(text: str) -> bool:
        """检测文本是否包含明显的代码特征。"""
        score = 0
        for pattern in SmartTextSplitter._CODE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                score += 1
                if score >= 2:
                    return True
        return False

    @staticmethod
    def _is_chinese_char(ch: str) -> bool:
        return '\u4e00' <= ch <= '\u9fff' or '\u3000' <= ch <= '\u303f'

    @classmethod
    def _chinese_ratio(cls, text: str) -> float:
        if not text:
            return 0.0
        chinese_count = sum(1 for ch in text if cls._is_chinese_char(ch))
        return chinese_count / len(text)

    # ──────────────────────────── 通用分块方法 ────────────────────────────

    def _semantic_split(self, text: str) -> list[str]:
        """语义分割：利用 embedding 模型的语义相似度确定断点。"""
        if self._semantic_splitter is None:
            self._semantic_splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type=self.breakpoint_threshold_type,
                buffer_size=self.buffer_size,
            )
        try:
            chunks = self._semantic_splitter.split_text(text)
            return chunks if chunks else [text]
        except Exception as e:
            logger.warning(f"[Splitter] 语义分割失败 ({e})，回退到 recursive")
            return self._recursive_split(text)

    def _get_recursive_splitter(self) -> RecursiveCharacterTextSplitter:
        if self._recursive_splitter is not None:
            return self._recursive_splitter
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.recursive_separators,
            length_function=len,
        )
        return self._recursive_splitter

    def _recursive_split(self, text: str) -> list[str]:
        """递归字符分割：按多级分隔符递归切割。"""
        splitter = self._get_recursive_splitter()
        try:
            docs = splitter.split_text(text)
            return docs if docs else [text]
        except Exception as e:
            logger.warning(f"[Splitter] 递归分割失败 ({e})，回退到 fixed")
            return self._fixed_split(text)

    def _paragraph_split(self, text: str) -> list[str]:
        """段落分割：按双换行符（段落边界）分割。"""
        if not text.strip():
            return []
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        merged = self._merge_short_chunks(paragraphs)
        return merged if merged else [text]

    def _sentence_split(self, text: str) -> list[str]:
        """句子级分割：按句末标点分割。"""
        if not text.strip():
            return []
        sentences = self._split_sentences(text)
        chunks = self._merge_by_size(sentences)
        return chunks if chunks else [text]

    def _fixed_split(self, text: str) -> list[str]:
        """固定长度分割：按字符数强行切割，带 overlap。"""
        if not text.strip():
            return []
        if len(text) <= self.chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        return chunks

    # ════════════════════════════════════════════════════════════════
    # 结构化文本分块方法
    # ════════════════════════════════════════════════════════════════

    # ──────────── JSON 分块 ────────────

    def _json_split(self, text: str) -> list[str]:
        """
        JSON 结构化分割：解析 JSON，按顶层键或数组元素分块。

        特点：
        - dict → 每个 top-level key 为一个独立块，保留 key: value 结构
        - array → 按元素分块，长元素进一步按 chunk_size 切割
        - 嵌套结构 → 保留路径上下文（如 "user.address.city"）
        - 大数组 → 每 max_items_per_chunk 个元素合并为一块
        """
        if not text.strip():
            return []

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"[Splitter] JSON 解析失败 ({e})，回退到 recursive")
            return self._recursive_split(text)

        chunks = self._json_flatten(data, path="", depth=0)

        if not chunks:
            return [text]

        # 如果块太大，进一步切割
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size * 1.5:
                sub_chunks = self._recursive_split(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _json_flatten(
        self,
        data: Any,
        path: str = "",
        depth: int = 0,
    ) -> list[str]:
        """
        递归展平 JSON 结构为文本块列表。

        Args:
            data: 当前 JSON 节点
            path: 当前路径（如 "user.address"）
            depth: 当前递归深度，超过 max_depth 则序列化为完整字符串
        """
        if depth > self.json_max_depth:
            # 超过深度限制，整个序列化为一个块
            return [json.dumps(data, ensure_ascii=False)]

        if isinstance(data, dict):
            return self._json_flatten_dict(data, path, depth)
        elif isinstance(data, list):
            return self._json_flatten_list(data, path, depth)
        elif isinstance(data, str):
            if path:
                return [f'"{path}": "{data}"']
            return [data]
        else:
            # bool / int / float / None
            value = json.dumps(data, ensure_ascii=False)
            if path:
                return [f'"{path}": {value}']
            return [value]

    def _json_flatten_dict(
        self,
        data: dict,
        path: str = "",
        depth: int = 0,
    ) -> list[str]:
        """将 JSON dict 展平为块列表，每个 key 一个块。"""
        chunks = []
        for key, value in data.items():
            child_path = f"{path}.{key}" if path else str(key)

            # 标量值（str/number/bool/None）可以直接成块
            if isinstance(value, (str, int, float, bool)) or value is None:
                val_str = json.dumps(value, ensure_ascii=False)
                chunks.append(f'"{child_path}": {val_str}')
            else:
                # 嵌套结构：递归
                sub_chunks = self._json_flatten(value, child_path, depth + 1)
                chunks.extend(sub_chunks)
        return chunks

    def _json_flatten_list(
        self,
        data: list,
        path: str = "",
        depth: int = 0,
    ) -> list[str]:
        """将 JSON 数组展平为块列表，按元素分组。"""
        if not data:
            return [f'"{path}": []'] if path else ["[]"]

        # 数组元素类型单一化检测（都是标量 → 批量打包）
        all_scalar = all(
            isinstance(v, (str, int, float, bool)) or v is None
            for v in data
        )

        if all_scalar:
            return self._json_pack_scalar_array(data, path)

        # 元素比较多时按 max_items_per_chunk 分批
        batches = []
        for i in range(0, len(data), self.json_max_items_per_chunk):
            batch = data[i:i + self.json_max_items_per_chunk]
            batch_label = f"{path}[{i}:{i + len(batch) - 1}]" if path else ""
            for idx, item in enumerate(batch):
                actual_idx = i + idx
                item_path = f"{path}[{actual_idx}]" if path else ""
                sub_chunks = self._json_flatten(item, item_path, depth + 1)
                batches.extend(sub_chunks)

        return batches

    def _json_pack_scalar_array(
        self,
        items: list,
        path: str,
    ) -> list[str]:
        """将标量数组打包成带上下文的块。"""
        serialized = json.dumps(items, ensure_ascii=False)
        if not path:
            return [serialized]
        if len(serialized) <= self.chunk_size:
            return [f'"{path}": {serialized}']
        # 太长则分批
        sub_batches = []
        batch_size = max(self.json_max_items_per_chunk, 1)
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_str = json.dumps(batch, ensure_ascii=False)
            sub_batches.append(f'"{path}[{i}:{i + len(batch)}]": {batch_str}')
        return sub_batches

    # ──────────── HTML 分块 ────────────

    def _html_split(self, text: str) -> list[str]:
        """
        HTML 语义块分割：按 HTML 语义标签提取独立块。

        支持的语义块（按优先级从高到低）：
        - <h1>~<h6> 标题及其后续内容
        - <table> 表格
        - <pre><code> / <code> 代码块
        - <ul> / <ol> 列表
        - <blockquote> 引用
        - <p> 段落
        - <div>/<section> 区块（带有 id/class 的）
        - 剩余文本按 recursive 分割
        """
        if not text.strip():
            return []

        chunks = []
        remaining = text

        # 按顺序提取各语义块，提取后从剩余文本中移除
        extractors = [
            ("表格", self._html_extract_tables),
            ("代码块", self._html_extract_code_blocks),
            ("标题", self._html_extract_headings),
            ("列表", self._html_extract_lists),
            ("引用", self._html_extract_blockquotes),
        ]

        for label, extractor in extractors:
            extracted, remaining = extractor(remaining)
            if extracted:
                logger.debug(f"[Splitter HTML] 提取到 {len(extracted)} 个{label}块")
                chunks.extend(extracted)

        # 确保所有块不超过 chunk_size
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size * 1.5:
                final_chunks.extend(self._recursive_split(chunk))
            else:
                final_chunks.append(chunk)

        # 处理剩余文本（剥离 HTML 标签后用 recursive）
        plain_text = re.sub(r"<[^>]*>", "", remaining).strip()
        if plain_text:
            plain_chunks = self._recursive_split(plain_text)
            final_chunks.extend(plain_chunks)

        # 如果什么都没提取到（纯 HTML 但没有匹配的语义块）
        if not final_chunks and remaining.strip():
            plain_text = re.sub(r"<[^>]*>", "", remaining).strip()
            if plain_text:
                return self._recursive_split(plain_text)

        return final_chunks if final_chunks else [plain_text] if plain_text else [text]

    @staticmethod
    def _strip_html(html: str) -> str:
        """去除 HTML 标签，保留文本内容。"""
        text = re.sub(r"<[^>]*>", "", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _html_extract_matches(
        html: str,
        pattern: re.Pattern,
        wrap_tag: str = "",
    ) -> tuple[list[str], str]:
        """
        通用 HTML 块提取：匹配正则，从剩余文本移除匹配部分。

        Returns:
            (提取的块列表, 剩余文本)
        """
        extracted = []
        remaining = html

        while True:
            match = re.search(pattern, remaining, re.DOTALL | re.IGNORECASE)
            if not match:
                break
            raw = match.group(0)
            text = SmartTextSplitter._strip_html(raw)
            if text:
                if wrap_tag:
                    text = f"[{wrap_tag}]: {text}"
                extracted.append(text)
            # 从 remaining 中移除匹配部分
            remaining = remaining[:match.start()] + remaining[match.end():]

        return extracted, remaining

    @classmethod
    def _html_extract_tables(cls, html: str) -> tuple[list[str], str]:
        """提取 <table> 块。"""
        pattern = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)
        extracted, remaining = cls._html_extract_matches(html, pattern, wrap_tag="表格")
        return extracted, remaining

    @classmethod
    def _html_extract_code_blocks(cls, html: str) -> tuple[list[str], str]:
        """提取 <pre><code> 或 <code> 代码块。"""
        # 优先匹配 <pre><code>
        pattern = re.compile(
            r"<pre[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>",
            re.DOTALL | re.IGNORECASE,
        )
        extracted, remaining = cls._html_extract_matches(html, pattern)
        # 补上 wrap 标签
        extracted = [f"[代码块]: {c}" for c in extracted if c.strip()]

        # 再匹配独立的 <code>
        pattern2 = re.compile(
            r"<code[^>]*>(.*?)</code>",
            re.DOTALL | re.IGNORECASE,
        )
        code2, remaining = cls._html_extract_matches(remaining, pattern2)
        code2 = [f"[代码]: {c}" for c in code2 if c.strip()]
        extracted.extend(code2)

        return extracted, remaining

    @classmethod
    def _html_extract_headings(cls, html: str) -> tuple[list[str], str]:
        """提取 <h1>~<h6> 标题及其后续内容直到下一个标题。"""
        pattern = re.compile(
            r"<h([1-6])[^>]*>(.*?)</h\1>",
            re.DOTALL | re.IGNORECASE,
        )
        extracted = []
        remaining = html

        while True:
            match = re.search(pattern, remaining, re.DOTALL | re.IGNORECASE)
            if not match:
                break
            level = match.group(1)
            title_text = cls._strip_html(match.group(2))
            start = match.start()
            end = match.end()

            # 找下一个标题的位置
            next_heading = re.search(
                r"<h[1-6][^>]*>.*?</h[1-6]>",
                remaining[end:],
                re.DOTALL | re.IGNORECASE,
            )
            if next_heading:
                section_end = end + next_heading.start()
            else:
                section_end = len(remaining)

            # 提取标题 + 内容
            section_html = remaining[end:section_end]
            section_text = cls._strip_html(section_html)
            chunk = f"[H{level}] {title_text}"
            if section_text:
                chunk += f"\n{section_text}"

            # 限制块大小
            if len(chunk) > 5000:
                chunk = chunk[:5000] + "..."

            extracted.append(chunk)
            remaining = remaining[:start] + remaining[section_end:]

        return extracted, remaining

    @classmethod
    def _html_extract_lists(cls, html: str) -> tuple[list[str], str]:
        """提取 <ul> / <ol> 列表块。"""
        pattern = re.compile(
            r"<(ul|ol)[^>]*>.*?</\1>",
            re.DOTALL | re.IGNORECASE,
        )
        extracted, remaining = cls._html_extract_matches(html, pattern, wrap_tag="列表")
        return extracted, remaining

    @classmethod
    def _html_extract_blockquotes(cls, html: str) -> tuple[list[str], str]:
        """提取 <blockquote> 引用块。"""
        pattern = re.compile(
            r"<blockquote[^>]*>.*?</blockquote>",
            re.DOTALL | re.IGNORECASE,
        )
        extracted, remaining = cls._html_extract_matches(html, pattern, wrap_tag="引用")
        return extracted, remaining

    # ──────────── Markdown 分块 ────────────

    def _markdown_split(self, text: str) -> list[str]:
        """
        Markdown 标题层级分割：按标题（# / ## / ###）分割文档。

        特点：
        - 以标题为分割点，每个标题及其后续内容为一个独立块
        - 标题行本身保留在块中
        - 文档开头的无标题内容作为导言块
        - 根据 markdown_heading_level 控制分割粒度（默认 = 2，即 ##）
        - 支持 ATX（#）和 Setext（=== / ---）两种标题格式
        """
        if not text.strip():
            return []

        lines = text.split("\n")
        chunks = []
        current_chunk_lines = []
        heading_found = False

        target_level = self.markdown_heading_level

        i = 0
        while i < len(lines):
            line = lines[i]

            # 检测 ATX 标题（指定级别）
            atx_match = re.match(
                r"^(#{1,6})\s+(.*)",
                line,
            )
            is_target_heading = False

            if atx_match:
                level = len(atx_match.group(1))
                if level <= target_level:
                    is_target_heading = True
            else:
                # 检测 Setext 标题（下一行是 === 或 ---）
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if re.match(r"^={3,}\s*$", next_line):
                        # level 1 setext
                        is_target_heading = (1 <= target_level)
                        if is_target_heading:
                            # 将当前行和下一行作为标题
                            title_line = line.strip()
                            lines[i + 1] = ""  # 标记为已处理
                            line = f"# #  {title_line}"
                    elif re.match(r"^-{3,}\s*$", next_line) and not line.startswith(" "):
                        # level 2 setext (避免和水平线混淆)
                        is_target_heading = (2 <= target_level)
                        if is_target_heading:
                            title_line = line.strip()
                            lines[i + 1] = ""
                            line = f"##   {title_line}"

            if is_target_heading:
                heading_found = True
                # 保存之前的块
                if current_chunk_lines:
                    chunks.append("\n".join(current_chunk_lines))
                current_chunk_lines = [line]
            else:
                current_chunk_lines.append(line)

            i += 1

        # 最后的块
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines).strip()
            if chunk_text:
                chunks.append(chunk_text)

        # 如果没有检测到标题，回退到 recursive
        if not heading_found or len(chunks) <= 1:
            return self._recursive_split(text)

        # 如果块太大，进一步分割
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size * 1.5:
                final_chunks.extend(self._recursive_split(chunk))
            else:
                final_chunks.append(chunk)

        return final_chunks

    # ──────────────────────────── 通用辅助方法 ────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """将文本分割成句子列表（支持中英文混排）。"""
        sentence_end = r"(?<=[。！？.!?])\s*"
        parts = re.split(sentence_end, text)
        return [p.strip() for p in parts if p.strip()]

    def _merge_short_chunks(
        self,
        chunks: list[str],
        min_size: Optional[int] = None,
    ) -> list[str]:
        """合并过短的文本块到相邻块中。"""
        if not chunks:
            return []
        min_size = min_size or max(self.chunk_size // 3, 50)
        merged = []
        for chunk in chunks:
            if not merged:
                merged.append(chunk)
            elif (
                len(chunk) < min_size
                and len(merged[-1]) + len(chunk) <= self.chunk_size * 1.5
            ):
                merged[-1] += "\n" + chunk
            else:
                merged.append(chunk)
        return merged

    def _merge_by_size(
        self,
        items: list[str],
        max_size: Optional[int] = None,
    ) -> list[str]:
        """将列表中的短项按 max_size 合并成块。"""
        max_size = max_size or self.chunk_size
        result = []
        current = ""
        for item in items:
            if not current:
                current = item
            elif len(current) + len(item) + 1 <= max_size:
                current += item
            else:
                result.append(current)
                current = item
        if current:
            result.append(current)
        return result

    def describe(self, text: str) -> dict:
        """返回对文本的分析报告。"""
        return {
            "length": len(text),
            "recommended_method": self._detect_method(text),
            "looks_like_json": self._looks_like_json(text.strip()),
            "looks_like_html": self._looks_like_html(text),
            "looks_like_markdown": self._looks_like_markdown(text),
            "is_code": self._has_code_features(text),
            "chinese_ratio": round(self._chinese_ratio(text), 4),
            "paragraph_count": len(re.findall(r"\n\s*\n", text)),
            "sentence_count": len(self._split_sentences(text)),
            "chunk_size": self.chunk_size,
        }
