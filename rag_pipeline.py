"""
rag_pipeline.py —— RAG 主链路（编排层）

完整流程（相比最初版本，多了 ① ② ⑤）：
  用户提问
    ↓ ① 查询改写：多轮对话里把"它有什么缺点"改写成"RAG 有什么缺点"
    ↓ ② 混合召回：向量通道（语义）+ BM25 通道（关键词）→ RRF 融合
    ↓ ③ Rerank 精排：交叉编码器逐个打分
    ↓ ④ 阈值过滤：丢掉分数太低的块
    ↓ ⑤ 全库汇总模式：汇总型问题（"这些文献的共性是什么"）或检索被清空时，
        改送【全部文献摘要】进 LLM——解决 Top-K 检索答不了全局问题这一固有边界
    ↓ 生成：拼 Prompt → LLM（支持流式）
    ↓ 回答

这个文件只做"编排"——把各层的零件按顺序串起来，自己不做具体实现。
想调整流程顺序或加新步骤，改这里；想改进某一环的效果，去改对应的模块。
"""
import time

import config
from hybrid_retriever import retrieve_candidates
from llm import generate_answer, rewrite_question, stream_answer, translate_query
from summarizer import is_global_query, load_summaries
from utils import cite_label, preview, short_name
from vector_store import load_vector_store


def rerank_all(query: str, candidates: list) -> list:
    """给【全部候选】打分，返回 [(文档, 分数)]。

    单独包一层的原因：① 避免顶层循环导入；② 把异常降级逻辑收在一处——
    接口限流/超时时退回向量排序，而不是让整个问答崩掉（工程健壮性）。
    """
    from rerank import rerank

    try:
        return rerank(query, candidates, top_k=len(candidates))
    except Exception as e:
        print(f"[警告] Rerank 调用失败，已降级为向量排序：{e}")
        return [(doc, None) for doc in candidates]


def prepare(question: str, vectordb=None, history: list = None) -> dict:
    """【检索阶段】改写 → 混合召回 → 精排 → 过滤，返回一个 context 字典。

    为什么不在这里直接生成回答？把"检索"和"生成"拆开之后：
      - 界面层可以先展示检索详情，再流式输出回答（体验更好）
      - 可以单独测试检索质量，不花 LLM 的钱（评估成本更低）
    """
    t0 = time.time()
    if vectordb is None:
        vectordb = load_vector_store()

    # ① 查询改写：让"它""这个"这类指代在检索前被澄清
    search_query = rewrite_question(question, history or [])
    rewritten = search_query if search_query != question else None

    # ①+ 跨语言查询扩展：中文问题翻译出英文变体，两路查询一起检索
    #（研究生场景刚需：中文提问、英文文献，不翻译的话 BM25 完全失效）
    extra_queries = []
    if config.QUERY_TRANSLATION_ENABLED:
        try:
            en = translate_query(search_query)
            if en and en != search_query:
                extra_queries.append(en)
        except Exception as e:
            print(f"[提示] 查询翻译失败（不影响主流程）：{e}")

    # ② 混合召回：向量 + BM25，RRF 融合（多查询时查询间也做 RRF 融合）
    candidates, recall_debug = retrieve_candidates(
        search_query, vectordb, extra_queries=extra_queries
    )
    t_recall = time.time()

    # ③ Rerank 精排（注意：全部候选都打分，不能提前截断，否则阈值过滤没意义）
    if candidates and config.RERANK_ENABLED:
        scored = rerank_all(search_query, candidates)
    else:
        # 关闭 Rerank 时用 None 占位，保持数据结构一致，下游不用分支判断
        scored = [(doc, None) for doc in candidates]

    # ④ 阈值过滤：宁可少给几个块，也不让无关内容污染上下文
    if config.RERANK_ENABLED:
        kept_scored = [(doc, s) for doc, s in scored
                       if s is not None and s >= config.RERANK_THRESHOLD]
        if not kept_scored and scored and all(s is None for _, s in scored):
            # Rerank 接口挂了（上面降级成 None 分数）——退回向量顺序，
            # 保证"有答案可给"优先于"绝对精准"（可用性优先原则）
            kept_scored = list(scored)
    else:
        kept_scored = list(scored)
    # 按 Rerank 分数从高到低排再截断 TOP_K（真实踩坑：候选顺序 ≠ 分数顺序，
    # 保底配额补进来的表格块虽然拿到 0.74 高分，却因为排在候选列表末尾
    # 被无序截断挤掉）。分数 None（未精排/摘要块）排最后。
    kept_scored.sort(key=lambda p: (p[1] is None, -(p[1] or 0.0)))
    kept_scored = kept_scored[: config.TOP_K]
    mode, mode_reason = "retrieval", ""

    # ⑤ 全库汇总模式：标准 Top-K 检索答不了"这些文献的共性是什么"这类问题
    #（答案分散在全库、不存在于任何单块），所以改用"全部文献摘要"作答。
    # 两条触发路径：① 问题命中汇总型关键词；② 检索被阈值清空（自动兜底）。
    if config.GLOBAL_QA_ENABLED:
        by_intent = is_global_query(question)
        by_fallback = config.GLOBAL_AUTO_FALLBACK and not kept_scored
        if by_intent or by_fallback:
            summaries = load_summaries(vectordb)[: config.GLOBAL_MAX_DOCS]
            if summaries:
                mode = "global"
                mode_reason = (
                    "问题属于全库汇总型（命中关键词）" if by_intent
                    else "检索结果全部低于阈值，自动降级为全库摘要模式"
                )
                # 摘要块没有检索分数，分数位用 None 占位（界面显示"—"）
                kept_scored = [(doc, None) for doc in summaries]
            elif by_fallback:
                mode_reason = "检索无命中，且文献库还没有摘要（可开启 SUMMARY_ENABLED 后重建）"

    kept = [doc for doc, _ in kept_scored]

    return {
        "mode": mode,
        "mode_reason": mode_reason,
        "question": question,
        "search_query": search_query,
        "rewritten": rewritten,
        "search_queries": [search_query] + extra_queries,  # 全部查询变体（界面展示用）
        "candidates": candidates,
        "recall": recall_debug,
        "scored": scored,
        "chunks": kept,
        # 保留块 + 各自的 Rerank 分数——界面据此渲染"引用证据卡"
        "kept_scored": kept_scored,
        "timings": {
            "召回(秒)": round(t_recall - t0, 2),
            "精排(秒)": round(time.time() - t_recall, 2),
        },
    }


def _no_answer_text() -> str:
    return (
        "未能从知识库中检索到与问题相关的内容。\n"
        f"可能原因：① 知识库为空；② 所有候选的 Rerank 分数都低于阈值 "
        f"{config.RERANK_THRESHOLD}（可调低阈值或换个问法）；"
        "③ 你问的是全库汇总型问题，但文献库里还没有摘要块"
        "（把 config.SUMMARY_ENABLED 设为 True 后重建知识库即可解决）。"
    )


def answer(context: dict) -> str:
    """【生成阶段·非流式】"""
    if not context["chunks"]:
        return _no_answer_text()
    return generate_answer(context["question"], context["chunks"], mode=context.get("mode", "retrieval"))


def answer_stream(context: dict):
    """【生成阶段·流式】逐字产出回答，供界面层 for 循环消费。"""
    if not context["chunks"]:
        yield _no_answer_text()
        return
    yield from stream_answer(
        context["question"], context["chunks"], mode=context.get("mode", "retrieval")
    )


def ask(question: str, vectordb=None, history: list = None) -> str:
    """一站式入口（命令行模式用）。界面层请用 prepare + answer_stream。"""
    context = prepare(question, vectordb, history)
    print_report(context)
    return answer(context)


def print_report(context: dict) -> None:
    """把检索全过程打印出来——调参和排查问题时最有用的观测工具。"""
    print(f"\n问题: {context['question']}")
    if context.get("mode") == "global":
        print(f"模式: 全库汇总模式（{context.get('mode_reason')}）→ 送入 {len(context['chunks'])} 篇文献摘要")
    elif context.get("mode_reason"):
        print(f"模式: 常规检索（{context['mode_reason']}）")
    if context["rewritten"]:
        print(f"查询改写: {context['rewritten']}")  # 多轮追问时能看到改写效果
    if len(context.get("search_queries", [])) > 1:
        print(f"查询变体（多语言检索）: {context['search_queries']}")

    recall = context["recall"]
    print(f"\n召回阶段（混合检索={'开' if recall['hybrid'] else '关'}，耗时 {context['timings']}）:")
    print("  [向量通道 Top5]")
    for hit in recall["vector"][:5]:
        print(f"    距离 {hit['score']:.3f} | {hit['name']} | {hit['preview']}...")
    if recall["bm25"]:
        print("  [BM25 通道 Top5]")
        for hit in recall["bm25"][:5]:
            print(f"    分数 {hit['score']:.2f} | {hit['name']} | {hit['preview']}...")

    if context["scored"]:
        print(f"\nRerank 打分（阈值 {config.RERANK_THRESHOLD}）:")
        for i, (doc, score) in enumerate(context["scored"], 1):
            if score is None:
                print(f"  [{i}] 未精排   | {cite_label(doc)} | {preview(doc, 30)}...")
                continue
            mark = "[保留]" if score >= config.RERANK_THRESHOLD else "[丢弃]"
            print(f"  [{i}] {score:.3f} {mark} | {cite_label(doc)} | {preview(doc, 30)}...")

    print(f"\n最终送入 LLM 的 {len(context['chunks'])} 个文本块:")
    for i, doc in enumerate(context["chunks"], 1):
        print(f"  [{i}] {cite_label(doc)} | {preview(doc, 60)}...")


if __name__ == "__main__":
    # 命令行模式：python rag_pipeline.py
    # 与之前不同的是维护了 history，所以支持多轮追问
    print("RAG 问答已启动（输入 q 退出）")
    db = load_vector_store()
    history = []
    while True:
        q = input("\n你的问题: ").strip()
        if q.lower() in ("q", "quit", "exit"):
            break
        if not q:
            continue
        ans = ask(q, db, history)
        history.append((q, ans))  # 记住这一轮，下轮追问时用于查询改写
        print("\n回答:", ans)
