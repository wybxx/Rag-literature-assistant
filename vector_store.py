"""
vector_store.py —— 第 3 步：向量化存储 + 第 4 步：相似度检索

核心概念：
- Embedding: 把文本变成一串数字（向量），语义相近的文本向量距离也近。
  "猫在睡觉" 和 "小猫入睡了" 的向量距离，远小于 "猫" 和 "股票行情"。
- 相似度检索: 用户问题也转成向量，在库里找距离最近的 K 个块（TOP_K）。
"""
import os
import shutil

import config
from loader import load_documents, split_documents

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def get_embeddings():
    """创建 Embedding 模型客户端。

    走的是硅基流动的 OpenAI 兼容接口（DeepSeek 没有 Embedding 服务，
    所以对话和向量化用两家不同的供应商——这是国内 RAG 项目的常见架构）。
    """
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.EMBEDDING_API_KEY,
        openai_api_base=config.EMBEDDING_BASE_URL,
        # 【关键修复】默认情况下 LangChain 会先用 OpenAI 的分词器把文本
        # 转成 token 编号数组再发给 API（为了防超长）。但硅基流动等第三方
        # 接口对 token 数组输入的支持有问题，返回的向量语义错乱——
        # 表现为"中文问题和无关英文文本相似度反而更高"。
        # 关掉这个预处理，直接发送原文文本，第三方接口就正常了。
        check_embedding_ctx_length=False,
    )


def build_vector_store():
    """【入库】完整执行：加载文档 → 分块 → 生成文献摘要 → 向量化 → 存入 Chroma。

    只需在新增/修改文档后运行一次，之后问答直接走 load_vector_store()。
    每个阶段打印耗时——重建慢的时候能一眼看出瓶颈在哪（摘要 or 向量化）。
    """
    import time

    t0 = time.time()
    raw_docs = load_documents()
    print(f"[1/4] 文档解析完成：{len(raw_docs)} 个原始块（{time.time() - t0:.1f}s）")

    t1 = time.time()
    chunks = split_documents(raw_docs)
    print(f"[2/4] 切块完成：{len(chunks)} 个文本块（{time.time() - t1:.1f}s）")

    # 【新增·解决"全库共性/综述类问题答不了"】为每篇文献生成一份结构化摘要，
    # 作为 content_type="summary" 的块一起入库。详见 summarizer.py 的说明。
    if config.SUMMARY_ENABLED:
        from summarizer import build_summaries, save_summary_files, to_documents

        print("[3/4] 开始生成文献摘要（供全库汇总型问答使用；未改动的文献走缓存，不调 LLM）...")
        t2 = time.time()
        summaries = build_summaries(raw_docs)
        if summaries:
            save_summary_files(summaries)     # 同时存 Markdown，便于人工检查摘要质量
            chunks = chunks + to_documents(summaries)
            print(f"已生成 {len(summaries)} 篇文献摘要（同时另存于 {config.SUMMARY_DIR}，{time.time() - t2:.1f}s）")

    kinds = {}
    for c in chunks:
        k = c.metadata.get("content_type", "text")
        kinds[k] = kinds.get(k, 0) + 1
    print(f"[4/4] 共 {len(chunks)} 个文本块 {kinds}，开始向量化入库...")

    # 【关键】先清空旧库再入库！
    # Chroma.from_documents 是"追加"写入，如果不清空，
    # 每次重建都会把新块堆在旧块上，造成大量重复数据，
    # 检索结果会被重复块污染（这是真实项目里很常见的坑）。
    # 用 Chroma 自带的 delete_collection 接口删除，比删文件夹更干净。
    import chromadb
    client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
    try:
        client.delete_collection(config.COLLECTION_NAME)
        print("已清空旧向量库")
    except Exception:
        pass  # 集合不存在（第一次入库），无需处理

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=config.VECTOR_DB_PATH,
        collection_name=config.COLLECTION_NAME,
    )
    print(f"入库完成，总耗时 {time.time() - t0:.1f}s，向量库保存在 {config.VECTOR_DB_PATH}")
    return vectordb


def load_vector_store():
    """【读取】加载已存在的向量库（每次问答都调这个，不重复入库）。"""
    return Chroma(
        persist_directory=config.VECTOR_DB_PATH,
        embedding_function=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
    )


def retrieve(question: str, vectordb=None, k: int = None) -> list:
    """【检索·第一阶段：召回】根据问题捞回最相关的 k 个文本块。

    两阶段检索里，这一步负责"宁多勿漏"——k 传大一点（如 RECALL_K=12），
    让 Rerank 有足够候选可以精挑细选。
    """
    if vectordb is None:
        vectordb = load_vector_store()
    if k is None:
        k = config.TOP_K
    return vectordb.similarity_search(question, k=k)


def get_all_chunks(vectordb=None) -> list:
    """【取出全部文本块】用于给 BM25 建关键词索引（混合检索的关键词通道）。

    注意：这里拿到的是"文本 + 元数据"，不是向量——BM25 是纯文本算法，
    不关心向量，所以它和向量通道是两条完全独立的召回路径。
    """
    from langchain_core.documents import Document

    if vectordb is None:
        vectordb = load_vector_store()
    data = vectordb.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]


if __name__ == "__main__":
    # 直接运行本文件 = 执行入库
    build_vector_store()
