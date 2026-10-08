"""
llm.py —— 大模型层：生成回答 / 流式输出 / 多轮查询改写

要点：
1. 为什么 RAG 能缓解幻觉？—— 因为 Prompt 里塞的是"检索到的真实资料"，
   并明确要求模型"只根据资料回答"，把开放式生成变成受限阅读理解。
   （注意是"缓解"不是"消除"：检索错了照样答错，垃圾进垃圾出。）
2. 温度(temperature)设为 0：问答场景要稳定、不要发散。
3. 流式输出(stream=True)：模型是一个字一个字生成的，逐字返回能大幅
   改善等待体验——商用产品几乎都是流式的。
"""
import config

from openai import OpenAI

# 复用 openai SDK 连接 DeepSeek（两者接口兼容）。
# 放在模块顶层：连接只需建立一次，后续所有请求复用，不必每次重新握手。
client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)


def build_context(context_chunks: list) -> str:
    """把检索到的块拼成"带编号 + 带来源"的参考资料文本。

    编号（[1]、[2]…）是引用溯源的基础——模型照着编号在回答里标注来源。

    【文献场景升级】来源标注从"文件路径"改成「文献名 第X页 类型」：
    研究生需要精确到页码的引用，这样回答里的每个论断都能回溯到原文位置。
    """
    from utils import TYPE_LABEL

    blocks = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk.metadata
        name = meta.get("doc_name") or meta.get("source", "未知")
        page = meta.get("page") or 0
        kind = TYPE_LABEL.get(meta.get("content_type", "text"), "正文")
        # 摘要块没有页码（它不属于任何一页），标成「文献名 摘要」更准确
        ref = f"{name} 第{page}页 {kind}" if page else f"{name} {kind}"
        blocks.append(f"[{i}] (来源: {ref})\n{chunk.page_content}")
    return "\n\n".join(blocks)


def pick_prompt(mode: str) -> str:
    """按检索模式挑 Prompt：明细问题用"阅读理解"模板，汇总问题用"综述归纳"模板。"""
    return config.SYSTEM_PROMPT_GLOBAL if mode == "global" else config.SYSTEM_PROMPT


def generate_answer(question: str, context_chunks: list, mode: str = "retrieval") -> str:
    """【非流式】一次性拿到完整回答。

    context_chunks: 检索层返回的 Document 列表
    mode: "retrieval"（正常检索）| "global"（全库摘要模式）
    """
    prompt = pick_prompt(mode).format(context=build_context(context_chunks))
    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0,  # 问答场景要事实性，不要创造性
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


def stream_answer(question: str, context_chunks: list, mode: str = "retrieval"):
    """【流式】逐字返回回答，用 yield 变成一个"生成器"。

    调用方写法：for piece in stream_answer(q, chunks): 打印(piece)
    效果就是打字机式的逐字输出。原理：模型本来就是逐 token 生成的，
    stream=True 让它生成一个字就推一个字，不用等全部生成完。
    """
    prompt = pick_prompt(mode).format(context=build_context(context_chunks))
    stream = client.chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0,
        stream=True,  # 关键：开启流式
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )
    for chunk in stream:
        # 流式的每个片段结构是 chunk.choices[0].delta.content
        # 有些片段是空的（比如结束标记），要判断一下再吐出去
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


REWRITE_PROMPT = """你是一个查询改写助手。下面是一段对话历史和用户的最新提问。

最新提问可能包含"它""这个""那货"等指代，脱离历史就无法理解，也无法用于检索。

请把最新提问改写成一个不依赖对话历史、可以独立用于文档检索的完整问题。

要求：
1. 只输出改写后的问题本身，不要任何解释、不要引号、不要前缀。
2. 如果最新提问本身已经完整独立，原样输出。
3. 保持原语言（中文输入输出中文）。
4. 不要添加历史中不存在的信息。

对话历史：
{history}

最新提问：{question}

改写结果："""


def rewrite_question(question: str, history: list) -> str:
    """【多轮对话的关键一步】把指代性追问改写成独立问题。

    例：历史是"什么是 RAG"，用户追问"它有什么缺点？"
        → 改写为"RAG 有哪些缺点？"  ← 这样检索器才能查到东西

    history 格式：[("用户问过的话", "助手答过的话"), ...]
    """
    if not history or not config.QUERY_REWRITE_ENABLED:
        return question

    # 只带最近几轮，太长的历史既浪费 token 又容易让模型抓错重点
    recent = history[-config.MAX_HISTORY_TURNS :]
    history_text = "\n".join(f"用户: {q}\n助手: {a[:200]}" for q, a in recent)

    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": REWRITE_PROMPT.format(history=history_text, question=question),
            }
        ],
    )
    rewritten = response.choices[0].message.content.strip()
    # 防御式处理：模型偶尔会加引号或"改写结果："前缀，清洗掉
    return rewritten.strip('"“”').removeprefix("改写结果：").strip()


TRANSLATE_PROMPT = """把下面的问题翻译成英文，用于检索英文文献。

要求：
1. 只输出英文翻译本身，不要任何解释、不要引号。
2. 专有名词、模型名、数据集名保持原样。
3. 如果问题本身已经是英文，原样输出。

问题：{question}

英文翻译："""


def translate_query(question: str) -> str:
    """【跨语言检索的关键一步】把中文问题翻译出英文变体。

    场景：研究生用中文提问，文献却是英文。向量模型跨语言能力有限，
    BM25 更是完全失效（中文词匹配不到英文文本）。
    把问题翻译成英文后一起检索，两条查询的结果再做 RRF 融合。

    问题本身没有中文时不翻译（省一次 LLM 调用）。
    """
    if not config.QUERY_TRANSLATION_ENABLED:
        return question
    import re

    if not re.search(r"[\u4e00-\u9fff]", question):
        return question

    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "user", "content": TRANSLATE_PROMPT.format(question=question)}
        ],
    )
    translated = response.choices[0].message.content.strip()
    # 防御式清洗：模型偶尔会加引号或"英文翻译："前缀
    return translated.strip('"“”').removeprefix("英文翻译：").strip()
