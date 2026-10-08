"""
summarizer.py —— 文献摘要生成 + "全库汇总问答"的触发判断

【解决了什么问题】
标准 RAG 的检索单元是"文本块"，所以它只能回答"答案藏在少数几块里"的问题。
一旦问题变成"这些文献的共性是什么"，检索必然失败——因为：
  - 答案分散在 5 篇文献 × 上百个块里，不存在于任何一个块；
  - 拿"共性是什么"去和"某段正文"算相似度，分数自然全都低于阈值。
（这不是项目写错了，是所有 Top-K 检索式 RAG 的固有边界。）

【解法：给每篇文献做一张"名片"】
入库时额外让 LLM 读一遍每篇文献，产出结构化摘要（研究对象/方法/数据/结论/关键词），
作为一个 content_type="summary" 的特殊块存进向量库。于是：
  - 汇总型问题 → 把所有文献的「摘要」一起送进 LLM，模型就"纵览全库"了；
  - 明细型问题 → 摘要也会被召回，帮模型先建立全局认知（相当于自动的"背景知识"）。

长文献怎么摘要？用 map-reduce：
  map   ：把全文切成 N 段，每段各自摘要（避免超出模型上下文）
  reduce：把 N 段摘要合并成一份完整摘要
"""
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import config

SUMMARY_PROMPT = """你是学术文献信息抽取助手。请阅读下面这篇文献的正文片段，抽取关键信息。

输出格式（严格遵守，只输出这五行，不要任何额外说明）：
研究对象：……
核心方法：……
数据与实验：……
主要结论：……
关键词：词1, 词2, 词3

要求：忠于原文，不要编造数据；每行不超过 80 字；信息缺失就写"未提及"。

文献名：{name}
正文片段：
{text}

摘要："""

MERGE_PROMPT = """下面是同一篇文献不同部分的摘要，字段相同但内容各有侧重。
请把它们合并成一份完整摘要（去重、互补、不要遗漏关键信息）。

输出格式（严格遵守，只输出这五行）：
研究对象：……
核心方法：……
数据与实验：……
主要结论：……
关键词：词1, 词2, 词3

文献名：{name}
各部分摘要：
{parts}

合并后的摘要："""


def _client():
    """延迟导入 LLM 客户端——避免"没配 key 也强行初始化"的边缘问题。"""
    from llm import client

    return client


def _chat(prompt: str) -> str:
    response = _client().chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def split_for_summary(text: str, parts: int = None) -> list:
    """把长文本按【连续段落】切成 N 段（不是随机采样，保持原文顺序和逻辑）。"""
    if parts is None:
        parts = config.SUMMARY_PARTS
    text = text.strip()
    if len(text) <= config.SUMMARY_MAX_INPUT_CHARS:
        return [text]
    n = min(parts, max(1, len(text) // config.SUMMARY_MAX_INPUT_CHARS + 1))
    step = len(text) // n
    segs = []
    for i in range(n):
        start = i * step
        end = len(text) if i == n - 1 else (i + 1) * step
        segs.append(text[start:end])
    return [s for s in segs if s.strip()]


def summarize_text(name: str, text: str) -> str:
    """对一篇文献的全文做 map-reduce 摘要（一次 map 时直接返回，省一次调用）。"""
    segs = split_for_summary(text)
    if len(segs) == 1:
        return _chat(SUMMARY_PROMPT.format(name=name, text=segs[0]))

    part_summaries = []
    for i, seg in enumerate(segs, 1):
        try:
            part_summaries.append(_chat(SUMMARY_PROMPT.format(name=name, text=seg)))
        except Exception as e:  # 单段失败不影响整篇（可用性优先）
            print(f"  [提示] {name} 第 {i} 段摘要失败，已跳过：{e}")
    if not part_summaries:
        raise RuntimeError(f"{name} 所有分段的摘要都失败")
    if len(part_summaries) == 1:
        return part_summaries[0]

    joined = "\n\n".join(f"【第{i}部分】\n{s}" for i, s in enumerate(part_summaries, 1))
    return _chat(MERGE_PROMPT.format(name=name, parts=joined))


def collect_doc_texts(documents: list) -> dict:
    """把原始文档按【文献名】归组，拼出每篇的可摘要全文。

    正文和表格都进来（表格 Markdown 里常有实验数据，摘要需要），
    插图块只有一句"位于第几页"，对摘要没价值，跳过。
    每篇控制在 SUMMARY_MAX_INPUT_CHARS × SUMMARY_PARTS 字以内，防止单篇过长。
    """
    limit = config.SUMMARY_MAX_INPUT_CHARS * config.SUMMARY_PARTS
    grouped = {}
    for doc in documents:
        if doc.metadata.get("content_type") == "image":
            continue
        name = doc.metadata.get("doc_name") or doc.metadata.get("source", "未知")
        grouped.setdefault(name, []).append(doc)

    out = {}
    for name, docs in grouped.items():
        text = "\n".join(d.page_content for d in docs)
        out[name] = text[:limit]
    return out


def normalize(summary: str) -> str:
    """清洗模型输出：去掉代码块围栏、多余前缀，保证存进库的是干净文本。"""
    text = summary.strip()
    text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text).strip()
    return text


# ---------------------------------------------------------------------------
# 摘要缓存：文献内容没变 → 直接复用上次的摘要，重建时零 LLM 调用
#
# 为什么值得做：摘要是重建流程里最慢的一步（4 篇文献 ≈ 13 次串行 LLM 调用，
# 实测 ~100s），但绝大多数重建只是"新增/重抽了文档"，旧文献根本没变——
# 为没变的文献重付一遍 LLM 时间和费用纯属浪费。
# 缓存键用【文献全文的 sha256】：源文件变了、或解析代码升级导致抽取内容变了，
# 哈希自然对不上，摘要自动重生成，不需要手动失效。
# ---------------------------------------------------------------------------
def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _cache_path() -> str:
    return os.path.join(config.SUMMARY_DIR, "_cache.json")


def _load_cache() -> dict:
    try:
        with open(_cache_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}  # 首次重建 / 文件损坏 → 视为无缓存


def _save_cache(cache: dict) -> None:
    os.makedirs(config.SUMMARY_DIR, exist_ok=True)
    with open(_cache_path(), "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def build_summaries(documents: list) -> dict:
    """入库用：为每篇文献生成摘要，返回 {文献名: 摘要文本}。

    两级提速：
      1. 缓存命中（全文哈希一致）直接复用，不调 LLM；
      2. 未命中的文献用线程池并行摘要——LLM 调用是 IO 密集型，
         等响应的时间里线程是闲着的，多路并行近似线性加速。

    任何一篇失败都不会中断入库（否则一次网络抖动就白等几分钟），
    只打印提示并跳过——工程上叫"优雅降级"。
    """
    texts = collect_doc_texts(documents)

    cache = _load_cache() if config.SUMMARY_CACHE_ENABLED else {}
    results, todo = {}, {}
    for name, text in texts.items():
        h = _text_hash(text)
        hit = cache.get(name)
        if hit and hit.get("hash") == h and hit.get("summary"):
            results[name] = hit["summary"]
            print(f"  [摘要] {name}（{len(text)} 字）→ 缓存命中，跳过 LLM")
        else:
            todo[name] = (text, h)

    if todo:
        workers = max(1, min(config.SUMMARY_WORKERS, len(todo)))
        print(f"  [摘要] {len(todo)} 篇需新生成，{workers} 路并行...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(summarize_text, n, t): (n, h) for n, (t, h) in todo.items()}
            for fut in as_completed(futures):
                name, h = futures[fut]
                try:
                    s = normalize(fut.result())
                    results[name] = s
                    cache[name] = {"hash": h, "summary": s}
                    print(f"  [摘要] {name} 完成")
                except Exception as e:
                    print(f"  [摘要] {name} 失败，跳过：{e}")
        _save_cache(cache)  # 只在确有新摘要时落盘，避免无谓写文件

    # 按输入顺序返回，保证入库块顺序稳定（便于复现与调试）
    return {name: results[name] for name in texts if name in results}


def save_summary_files(summaries: dict) -> None:
    """把摘要另存为 Markdown——人工可读，方便检查摘要质量（也便于写进简历附录）。"""
    os.makedirs(config.SUMMARY_DIR, exist_ok=True)
    for name, summary in summaries.items():
        stem = os.path.splitext(name)[0]
        path = os.path.join(config.SUMMARY_DIR, f"{stem}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {name} · 文献摘要\n\n{summary}\n")


def to_documents(summaries: dict) -> list:
    """把摘要包装成 Document，好和普通块一起入库。

    为什么要入库而不是问答时现场生成？
      现场生成意味着每次提问都要重读全部文献（几分钟），既慢又贵；
      入库后它就是普通块，走同一套召回/精排/引用链路，零额外成本。
    """
    from langchain_core.documents import Document

    docs = []
    for name, summary in summaries.items():
        docs.append(
            Document(
                page_content=f"【文献摘要 · {name}】\n{summary}",
                metadata={
                    "source": os.path.join(config.SUMMARY_DIR, os.path.splitext(name)[0] + ".md"),
                    "doc_name": name,
                    "title": "",
                    "page": 0,                 # 0 = 不适用（摘要不属于任何单页）
                    "content_type": "summary",
                    "asset": "",
                    "caption": "",
                },
            )
        )
    return docs


def load_summaries(vectordb=None) -> list:
    """从向量库里取出所有文献摘要（全库汇总模式的数据来源）。"""
    from vector_store import get_all_chunks

    chunks = get_all_chunks(vectordb)
    summaries = [c for c in chunks if c.metadata.get("content_type") == "summary"]
    return sorted(summaries, key=lambda c: c.metadata.get("doc_name", ""))


def is_global_query(question: str) -> bool:
    """判断是否"全库汇总型问题"（命中关键词即算）。

    为什么用规则而不是让 LLM 分类？——这类问题的语言特征非常明显（共性/总结/都有哪些…），
    规则零延迟、零成本、可解释；LLM 分类虽然更灵活，但每次提问都要多花一次调用和 1 秒延迟。
    （更高阶的做法是"问题路由"：LLM 输出 detail/global 标签再分流，作为可选升级。）
    """
    q = question or ""
    return any(kw in q for kw in config.GLOBAL_KEYWORDS)
