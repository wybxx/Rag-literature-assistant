"""
loader.py —— 第 1 步：文档加载 + 第 2 步：文本分块

RAG 链路：文档加载 → 分块 → 向量化 → 检索 → 生成
本文件负责前两步。

针对"文献检索助手"场景的关键改动：
1. PDF 不再走 PyPDFLoader，改用 pdf_extractor 做结构化抽取
   （正文 + 表格 Markdown + 插图 PNG，全部带页码）；
2. 分块时【只切正文】——表格被拦腰切断后行列对不上号、
   插图只有一两句话也没有切的必要，所以这两类整块保留。
"""
import hashlib
import json
import os
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

import config
from pdf_extractor import EXTRACTOR_VERSION, extract_pdf


# ---------------------------------------------------------------------------
# 抽取缓存：PDF 结构化抽取很慢（实测 4 篇 ~70s，大头是渲染插图 PNG 和
# 全页矢量线条分析），但只要文件没变，抽取结果就一个字都不会变。
# 以【文件内容 sha256 + 抽取器版本号】为键存 JSON：
#   - 文件改了 → 哈希变 → 自动重新抽取；
#   - 抽取代码升级了 → 版本号 +1 → 旧缓存整体失效（防脏缓存）；
#   - assets/ 下的插图 PNG 被误删 → 命中时校验资产文件存在性，缺失即重抽。
# ---------------------------------------------------------------------------
def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _extract_cache_path(path: str) -> str:
    key = f"{_file_hash(path)}-v{EXTRACTOR_VERSION}"
    return os.path.join(config.EXTRACT_CACHE_DIR, f"{key}.json")


def _extract_pdf_cached(path: str) -> list:
    if not config.EXTRACT_CACHE_ENABLED:
        return extract_pdf(path)

    cache_file = _extract_cache_path(path)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, encoding="utf-8") as f:
                items = json.load(f)
            # 校验插图/表格 PNG 资产仍在（缓存文本引用的是文件路径）
            missing = [it["metadata"].get("asset") for it in items
                       if it["metadata"].get("asset") and not os.path.exists(it["metadata"]["asset"])]
            if not missing:
                print(f"  [解析] {os.path.basename(path)[:40]} → 缓存命中（{len(items)} 块）")
                return [Document(page_content=it["page_content"], metadata=it["metadata"])
                        for it in items]
            print(f"  [解析] {os.path.basename(path)[:40]} → 缓存引用的资产缺失（{len(missing)} 个），重新抽取")
        except Exception as e:
            print(f"  [解析] 缓存读取失败（{e}），重新抽取")

    docs = extract_pdf(path)
    os.makedirs(config.EXTRACT_CACHE_DIR, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump([{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
                  f, ensure_ascii=False)
    return docs


def _plain_doc(path: str, filename: str, loader_cls=TextLoader) -> list:
    """txt / md / docx 这类无页码文档：整篇作为一条 text 文档。"""
    try:
        docs = loader_cls(path, encoding="utf-8").load()
    except TypeError:  # 有的 Loader 不接受 encoding 参数
        docs = loader_cls(path).load()

    # 从 Markdown 的首个一级标题里猜文献标题（文献库列表展示用）
    title = ""
    if filename.lower().endswith(".md"):
        m = re.search(r"^#\s+(.+)$", docs[0].page_content, re.M)
        if m:
            title = m.group(1).strip()[:120]

    for d in docs:
        d.metadata.update({
            "source": path,
            "doc_name": filename,
            "title": title,
            "page": 0,                       # 0 = 不适用（非分页文档）
            "content_type": "text",
            "asset": "",
            "caption": "",
        })
    return docs


def load_documents(docs_dir: str = config.DOCS_DIR) -> list:
    """加载 docs 目录下的所有文献，返回 Document 列表（未切分）。

    每条 Document 的 metadata 统一包含：
      doc_name / title / page / content_type(text|table|image) / asset / caption
    """
    documents = []

    for filename in sorted(os.listdir(docs_dir)):
        path = os.path.join(docs_dir, filename)
        low = filename.lower()

        if low.endswith(".pdf"):
            # 结构化抽取：正文 + 表格 + 插图（见 pdf_extractor.py）；带缓存
            documents.extend(_extract_pdf_cached(path))
        elif low.endswith((".txt", ".md")):
            # Markdown 本质是纯文本，直接用 TextLoader
            documents.extend(_plain_doc(path, filename))
        elif low.endswith(".docx"):
            # Word 需要额外解析器，优雅降级：没装库就跳过并提示，而不是崩掉
            try:
                from langchain_community.document_loaders import Docx2txtLoader
                documents.extend(_plain_doc(path, filename, Docx2txtLoader))
            except ImportError:
                print(f"[跳过] {filename}：读取 .docx 需要先执行 pip install python-docx")

    if not documents:
        raise FileNotFoundError(
            f"{docs_dir} 下没有找到可加载的文献（支持 .pdf / .txt / .md / .docx）"
        )
    return documents


def split_documents(documents: list) -> list:
    """把长文档切成小块。为什么必须切？
    1. Embedding 模型有输入长度限制；
    2. 检索粒度小才精准——召回"最相关的段落"比"整篇文章"效果好；
    3. 拼进 Prompt 的上下文有限，省 token 就是省钱。

    【文献场景的关键差异】只切正文，表格/插图整块保留：
    - 一张 20 行的表格被切成两半，后半段没有表头，人和模型都读不懂；
    - 插图的检索文本就是"图注 + 位置"，本来就短。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        # 优先按这些分隔符切：段落 > 换行 > 句号 > 空格，尽量保住语义完整
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )

    text_docs = [d for d in documents if d.metadata.get("content_type", "text") == "text"]
    struct_docs = [d for d in documents if d.metadata.get("content_type", "text") != "text"]

    chunks = splitter.split_documents(text_docs) + struct_docs

    # 给每个块标上编号，方便回答时引用溯源（来源[1]、来源[2]…）
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


if __name__ == "__main__":
    # 直接运行本文件可以单独测试这两步
    docs = load_documents()
    chunks = split_documents(docs)
    kinds = {}
    for c in chunks:
        k = c.metadata.get("content_type", "text")
        kinds[k] = kinds.get(k, 0) + 1
    print(f"共加载 {len(docs)} 条原始内容，切分为 {len(chunks)} 块 -> {kinds}")
    print(f"示例块: [{chunks[0].metadata.get('content_type')}] "
          f"{chunks[0].metadata.get('doc_name')} p{chunks[0].metadata.get('page')} | "
          f"{chunks[0].page_content[:80]}...")
