"""
hybrid_retriever.py —— 混合检索（Hybrid Search）：向量通道 + 关键词通道 + RRF 融合

为什么需要混合检索：
1. 向量检索（语义通道）擅长"意思相近"：
   问"怎么避免模型瞎编"，能找到写着"缓解幻觉"的段落。
   但它对【专有名词、型号、代码符号】不敏感——
   问"bge-reranker 是什么"，它可能给你返回"embedding 模型简介"，
   因为在这两种文本的向量看来都很"像模型介绍"。

2. BM25（关键词通道）擅长"字面命中"：
   文档里出现 "bge-reranker" 这个词，BM25 立刻就锁定它。
   但它不懂同义词——文档写"幻觉"，你问"瞎编"，它一条都找不到。

两者刚好互补，所以工业界的标准做法是【双通道召回 + 结果融合】。

怎么融合？用 RRF（Reciprocal Rank Fusion，倒数排名融合）：
- 核心思想：只看"名次"，不看"分数"。
- 为什么不能直接比分数？因为 BM25 的分数可能是 8.7，向量距离是 1.21，
  两者尺度完全不同、根本不可比。但"在各自通道里排第几"是可比的。
- 公式：score(文档) = Σ 权重 / (RRF_K + 该文档在第 i 个通道里的名次)
  排名越靠前贡献越大；一个文档在两条通道里都靠前，总分就特别高。
（这也是微信搜一搜、Elasticsearch 等系统的常规做法。）
"""
import re

import config
from vector_store import get_all_chunks, load_vector_store

try:
    import jieba  # 中文分词库，装了就用它，效果比单字切分好
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


def tokenize(text: str) -> list:
    """把一段文本切成词列表——BM25 是按"词"统计的，中文必须先分词。

    装了 jieba 就用 jieba（"检索增强生成" → ['检索', '增强', '生成']）；
    没装则退化成"单字 + 相邻双字"切分（效果差一些但行为可用）。
    """
    if _HAS_JIEBA:
        return [w.strip() for w in jieba.lcut(text) if w.strip()]

    tokens = []
    # 英文单词/数字整体保留，中文单字逐个保留
    for piece in re.findall(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]", text):
        tokens.append(piece.lower())
    # 补上相邻中文双字组合："检索增强" → ['检索', '索增', '增强']
    # 中文双字词信息量最大，这样能近似还原大部分词
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        for i in range(len(seg) - 1):
            tokens.append(seg[i : i + 2])
    return tokens


_BM25_CACHE = {}  # 简单缓存：避免每次提问都重新建索引（建索引是有成本的）


def build_bm25(chunks: list):
    """给全部文本块建立 BM25 索引。"""
    from rank_bm25 import BM25Okapi  # 延迟导入：没装这个库也不影响其他功能

    corpus = [tokenize(c.page_content) for c in chunks]
    return BM25Okapi(corpus)


def _get_bm25(chunks: list):
    """带缓存的索引获取。key 用"块数量 + 首块指纹"，
    这样只要知识库没变，索引就只建一次。"""
    key = (len(chunks), hash(chunks[0].page_content[:50]) if chunks else 0)
    if key not in _BM25_CACHE:
        _BM25_CACHE[key] = build_bm25(chunks)
    return _BM25_CACHE[key]


def bm25_search(question: str, chunks: list, k: int) -> list:
    """【关键词通道】BM25 检索，返回 [(文档, 分数)] 按分数从高到低。"""
    if not chunks:
        return []
    bm25 = _get_bm25(chunks)
    scores = bm25.get_scores(tokenize(question))
    # 取分数最高的 k 个下标
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(chunks[i], float(scores[i])) for i in order]


def _doc_key(doc):
    """给文档算一个"身份标识"，用于在融合时识别"这是同一块内容"。

    用 来源文件 + 块编号 做 key——同一块内容在两条通道里出现时，
    必须能被认出是同一个，才能把两边的名次贡献加在一起。
    """
    return (doc.metadata.get("source", ""), doc.metadata.get("chunk_id", doc.page_content[:30]))


def rrf_fuse(rank_lists: list, weights: list = None, top_n: int = None) -> list:
    """RRF 融合：把多路召回结果按"名次"合并成一个最终排名。

    参数:
        rank_lists: [[通道1的文档列表(已按相关度排序)], [通道2的...], ...]
        weights:    每个通道的权重
        top_n:      返回前多少条
    """
    if weights is None:
        weights = [1.0] * len(rank_lists)

    scores = {}   # 文档 key → 融合分数
    store = {}    # 文档 key → 文档对象本身
    for channel_idx, docs in enumerate(rank_lists):
        for rank, doc in enumerate(docs, start=1):  # rank 从 1 开始
            key = _doc_key(doc)
            # 公式：权重 / (K + 名次)。名次越靠前，贡献越大
            scores[key] = scores.get(key, 0.0) + weights[channel_idx] / (config.RRF_K + rank)
            store[key] = doc

    ordered = sorted(scores, key=lambda k: scores[k], reverse=True)
    if top_n:
        ordered = ordered[:top_n]
    return [store[key] for key in ordered]


def _type_quota(pool: list, full_ranking: list) -> list:
    """【类型保底配额】候选池里表格/插图不足时，从完整融合排名里补进来。

    为什么需要（真实踩坑）：
    问"DHyMamba 的实验结果数据是多少"，结果表在向量通道排 36 名、
    BM25 排 30 名——被中文摘要块和大量正文块挤出 RECALL_K=12 的候选池，
    Rerank 根本没机会看到它。可一旦让 Rerank 看到，它能打出 0.74 的高分
    （交叉编码器精排能力远强于双塔召回）。

    教训：召回阶段的"分数低"不等于"不相关"，可能只是"被同类内容淹没"。
    所以给表格/插图留保底名额，让精排阶段做最终裁决。

    参数:
        pool:         融合后的前 K 名候选
        full_ranking: 未截断的完整融合排名
    """
    from utils import doc_type, short_name

    quotas = {"table": config.RECALL_QUOTA_TABLE, "image": config.RECALL_QUOTA_IMAGE}
    added = []
    pool_keys = {_doc_key(d) for d in pool}
    for ctype, quota in quotas.items():
        have = sum(1 for d in pool if doc_type(d) == ctype)
        for d in full_ranking:
            if have >= quota:
                break
            if doc_type(d) == ctype and _doc_key(d) not in pool_keys:
                pool.append(d)
                pool_keys.add(_doc_key(d))
                have += 1
                added.append(f"{ctype}:{short_name(d)} p{d.metadata.get('page')}")
    return pool, added


def retrieve_candidates(question: str, vectordb=None, k: int = None, chunks: list = None,
                        extra_queries: list = None):
    """【混合检索主入口】返回 (候选文档列表, 调试信息)。

    extra_queries: 额外的查询变体（如中文问题的英文翻译）。
    多查询时，向量/BM25 通道都对每条查询各检索一次，查询之间先用
    RRF 按名次融合，再走双通道融合（多查询检索，Multi-Query Retrieval）。

    调试信息里保留了两个通道各自的原始结果——界面上要展示给用户看，
    这样"为什么这块被召回了"是可解释的（可解释性是 RAG 工程的重要指标）。
    """
    if vectordb is None:
        vectordb = load_vector_store()
    if k is None:
        k = config.RECALL_K

    from utils import preview, short_name

    queries = [question] + [q for q in (extra_queries or []) if q and q != question]
    debug = {"hybrid": config.HYBRID_ENABLED, "vector": [], "bm25": [], "queries": queries}

    # 通道召回深度要比最终候选数大（真实踩坑：结果表在通道里排 30/36 名，
    # 按 k=12 截断的话它根本进不了融合环节，后面的类型保底配额也无米下锅。
    # 所以先"宽捞"再精简，把裁决权交给下一阶段的 Rerank 精排）
    deep_k = max(k, config.RECALL_DEEP_K)

    # ---- 通道 1：向量检索（语义）----
    # 用 with_score 版本把距离留下来——界面上可以和 Rerank 分数对比。
    # 多查询时：每条查询各检索一次，查询间用 RRF 融合（只看名次，尺度无关）
    per_query = []
    for qi, q in enumerate(queries):
        v_hits = vectordb.similarity_search_with_score(q, k=deep_k)
        per_query.append([doc for doc, _ in v_hits])
        if qi == 0:  # 调试信息展示主查询的原始结果
            debug["vector"] = [
                {"name": short_name(doc), "score": round(float(s), 3), "preview": preview(doc, 30)}
                for doc, s in v_hits
            ]
    vector_docs = rrf_fuse(per_query, top_n=deep_k) if len(per_query) > 1 else per_query[0]

    if not config.HYBRID_ENABLED:
        return vector_docs, debug

    # ---- 通道 2：BM25 检索（关键词）----
    try:
        if chunks is None:
            chunks = get_all_chunks(vectordb)
        per_query_bm25 = [bm25_search(q, chunks, deep_k) for q in queries]
    except ImportError:
        # 没装 rank_bm25：优雅降级为纯向量检索，而不是让整个程序崩掉
        print("[提示] 未安装 rank_bm25，已自动降级为纯向量检索（pip install rank_bm25 可启用混合检索）")
        return vector_docs, debug

    debug["bm25"] = [
        {"name": short_name(doc), "score": round(s, 2), "preview": preview(doc, 30)}
        for doc, s in per_query_bm25[0]
    ]
    bm25_docs = (
        rrf_fuse([[doc for doc, _ in hits] for hits in per_query_bm25], top_n=deep_k)
        if len(per_query_bm25) > 1
        else [doc for doc, _ in per_query_bm25[0]]
    )

    # ---- 融合：两条通道的名次加权合并 ----
    # 先拿到"未截断"的完整融合排名：类型保底配额需要知道表格/插图在全库中
    # 的最好名次——只看截断后的前 K 名就无从补起了
    full = rrf_fuse(
        [vector_docs, bm25_docs],
        weights=[config.HYBRID_WEIGHT_VECTOR, config.HYBRID_WEIGHT_BM25],
    )
    fused, quota_added = _type_quota(full[:k], full)
    debug["quota"] = quota_added
    return fused, debug


if __name__ == "__main__":
    # 单独测试：python hybrid_retriever.py
    q = "RAG 的幻觉问题"
    cands, info = retrieve_candidates(q)
    print(f"问题: {q}")
    print(f"向量通道 Top5: {[h['name'] for h in info['vector'][:5]]}")
    print(f"BM25 通道 Top5: {[h['name'] for h in info['bm25'][:5]]}")
    print(f"融合后候选数: {len(cands)}")
