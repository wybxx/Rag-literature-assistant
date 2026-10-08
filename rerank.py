"""
rerank.py —— 重排序（Rerank）：两阶段检索的第二阶段

为什么需要 Rerank：
向量检索是"双塔"（Bi-Encoder）——问题和文档【分别】编码成向量再比距离。
优点：文档向量可以提前算好，检索速度极快，适合百万级文档初筛。
缺点：两边编码时互相看不见对方，只学到了"大致语义"，细粒度对应关系抓不住。
  典型翻车现场：问"什么是幻觉"，召回的第一名是无关英文论文，
  真正提到幻觉定义的中文块只排第二——因为那个信号被稀释在 500 字里。

Rerank 模型是"交叉编码器"（Cross-Encoder）——把【问题 + 文档】拼成
一整个输入送进模型，每个词都能注意到另一边的词，直接输出相关性分数。
精度远高于向量检索，但每个候选都要跑一次模型，没法对全库做。

所以业界标准做法是两阶段：
  第一阶段【召回 Recall】：向量检索从全库快速捞出 TOP 20~50（宁多勿漏）
  第二阶段【精排 Rerank】：交叉编码器从候选里精选 TOP 3~5（宁缺毋滥）
"""
import requests

import config


def rerank(question: str, docs: list, top_k: int = None) -> list:
    """调用硅基流动的 Rerank 接口，对候选文本块按相关性重新排序。

    参数:
        question: 用户问题
        docs:     向量检索召回的候选文本块（LangChain Document 对象列表）
        top_k:    精排后保留的数量，默认取 config.TOP_K
    返回:
        [(文档, 分数)] 元组列表，按相关性从高到低排列
        分数要带出去——外面要靠它做阈值过滤（过滤无关候选）
    """
    if top_k is None:
        top_k = config.TOP_K
    if not docs:
        return []

    response = requests.post(
        # EMBEDDING_BASE_URL 是 ".../v1"，rerank 接口就挂在它下面
        f"{config.EMBEDDING_BASE_URL}/rerank",
        headers={"Authorization": f"Bearer {config.EMBEDDING_API_KEY}"},
        json={
            "model": config.RERANK_MODEL,
            "query": question,
            "documents": [d.page_content for d in docs],
            "top_n": top_k,
            # 只要分数和下标，不要返回原文（原文我们本地就有，省流量）
            "return_documents": False,
        },
        timeout=30,
    )
    response.raise_for_status()  # 网络层没问题但接口报错时（如 401 密钥错误）抛异常

    # 返回格式: {"results": [{"index": 3, "relevance_score": 0.98}, ...]}
    # index 对应我们传入 documents 列表的下标，已按分数从高到低排好
    results = response.json()["results"]
    # 返回 (文档, 分数) 元组列表——分数必须带出去，
    # 外面要用它做阈值过滤（低于阈值的块相关性太差，直接丢弃）
    return [(docs[item["index"]], item["relevance_score"]) for item in results]
