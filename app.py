"""
app.py —— 文献检索助手 · 可视化界面

界面分三个 Tab：
  💬 智能问答   流式多轮问答 + 「文献名+页码」引用 + 命中表格/插图原貌展示
  📚 文献库     库里所有文献的清单；点开任意一篇，看它被抽取出的全部表格与插图
  📤 知识库管理 上传文献 + 一键重建向量库

设计原则（也是工程亮点）：
  - 检索过程"白盒化"：右栏能看到每个候选的分数和去留判定，可信、可解释；
  - 证据可核验：回答引用到页码，命中的表格/插图直接给原图，研究生可点开核对；
  - 参数实时可调：演示时能"当场做实验"，同一问题开关 Rerank 对比效果。
"""
import os
import shutil

import gradio as gr

# ---------------------------------------------------------------------------
# 兼容性补丁：修 gradio_client 1.3.0 与新版 pydantic 的冲突（上游 bug）
#
# 现象：python app.py 启动后满屏
#         TypeError: argument of type 'bool' is not iterable
#       最后抛
#         ValueError: When localhost is not accessible, ...
#       （后者只是前者的连锁反应，不是真正的病根）
#
# 原因：gr.Chatbot 组件生成的 api_info 里含有
#         "constructor_args": {"additionalProperties": true, ...}
#       这个 additionalProperties 是【布尔值】。而 gradio_client 1.3.0 的
#       _json_schema_to_python_type() 没有做类型判断，会把 True 当成一个 schema
#       继续递归解析，执行到 `"const" in True` 时崩溃。
#
# 修法：给该函数加一层 isinstance 守卫——不是字典的 schema 一律当 Any 处理。
# 备注：gradio 4.44.1 是 2024 年的版本，解析不了新版 pydantic 生成的 schema；
#       等以后升级到 gradio 5.x（需要 Python 3.10+）即可删除本补丁。
# ---------------------------------------------------------------------------
import gradio_client.utils as _gc_utils

_orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type


def _safe_json_schema_to_python_type(schema, defs):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_json_schema_to_python_type(schema, defs)


_gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 兼容性补丁 2：修「开着代理时启动崩溃」（上游设计缺陷）
#
# 现象：python app.py 打印 "Running on local URL: http://127.0.0.1:7860" 之后
#       立刻抛
#         ValueError: When localhost is not accessible, a shareable link must
#         be created. Please set share=True or check your proxy settings...
#       上一行往往还有一条
#         ConnectionResetError: [WinError 10054] 远程主机强迫关闭了一个现有的连接
#
# 原因：gradio 启动最后会做一次「本地连通性自检」——用 httpx 发一个 HEAD 请求
#      到自己刚监听的 http://127.0.0.1:7860，只有状态码是 200/401/302 才算通过
#      （源码：gradio/blocks.py 的 launch() → networking.url_ok()）。
#       而 httpx 默认 trust_env=True，会读环境变量和 Windows 系统代理（注册表
#       Internet Settings）。国内常见的代理软件（Clash / v2rayN 等）一旦开启
#       "系统代理"，这个发往 127.0.0.1 的请求也会被塞进代理：代理既可能直接
#       重置连接（WinError 10054），也可能回一个 502 —— 两者都不在
#       gradio 的白名单里，于是自检误判"本机服务不可达"，启动失败。
#       实测：被代理接管时 httpx.head 拿到 502，而 502 ∉ {200,401,302} → 报错。
#
# 修法（双保险）：
#   ① 让 loopback 地址绕过代理（NO_PROXY/no_proxy，httpx 与 requests 都认）；
#   ② 覆写 url_ok：目标主机是本机时直接判可达（本地服务本来就不需要探测），
#      非本机 URL（如 share 链接）仍走原逻辑，不改变原语义。
# ---------------------------------------------------------------------------
import urllib.parse

import gradio.networking as _gr_net

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
_LOOPBACK_LIST = "127.0.0.1,localhost,::1"

# ① 把 loopback 追加进代理绕过名单（保留用户已有的配置，不覆盖）
for _var in ("NO_PROXY", "no_proxy"):
    _cur = os.environ.get(_var, "").strip()
    if _LOOPBACK_LIST not in _cur:
        os.environ[_var] = f"{_cur},{_LOOPBACK_LIST}" if _cur else _LOOPBACK_LIST

# ② 本机地址不自检，直接判可达
_orig_url_ok = _gr_net.url_ok


def _url_ok(url: str) -> bool:
    try:
        if urllib.parse.urlparse(url).hostname in _LOOPBACK_HOSTS:
            return True
    except Exception:
        pass
    return _orig_url_ok(url)


_gr_net.url_ok = _url_ok
# ---------------------------------------------------------------------------

import config
from library import doc_detail, doc_names, library_overview, library_stats
from pdf_extractor import backend
from rag_pipeline import answer_stream, prepare
from utils import cite_label, doc_type, esc, preview
from vector_store import build_vector_store

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, config.DOCS_DIR.lstrip("./"))
ASSETS_DIR_ABS = os.path.abspath(config.ASSETS_DIR)
os.makedirs(DOCS_DIR, exist_ok=True)

# ============================== 视觉样式 ==============================

CUSTOM_CSS = """
.stat-card { display:flex; gap:12px; flex-wrap:wrap; margin:2px 0 8px 0; }
.stat { flex:1; min-width:110px; background:#f6f8fc; border:1px solid #e4e9f2;
        border-radius:12px; padding:10px 14px; text-align:center; }
.stat .num { font-size:24px; font-weight:700; color:#2b53c8; }
.stat .lbl { font-size:12px; color:#5a6472; margin-top:2px; }

.ev-card { border:1px solid #e4e9f2; border-left:4px solid #4f7cff; border-radius:10px;
           padding:10px 12px; margin-bottom:10px; background:#fbfcff; }
.ev-head { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:6px; }
.ev-name { font-weight:600; color:#1f2d3d; font-size:14px; }
/* 修复：超长文件名（无空格长串）默认不会在中间断行，会溢出压到相邻内容上。
   overflow-wrap:anywhere + word-break 允许在任意字符处折行。 */
.ev-name { word-break:break-all; overflow-wrap:anywhere; white-space:normal; min-width:0; }
.prose table td, .prose table th, table td, table th {
  word-break:break-word; overflow-wrap:anywhere; }
.prose table, .markdown table { display:table; width:100%; }
.prose, .markdown, .chatbot { overflow-wrap:anywhere; }
.pill { font-size:12px; padding:1px 9px; border-radius:999px; background:#e8efff;
        color:#2b53c8; white-space:nowrap; display:inline-block; }
.pill.table { background:#fff3e0; color:#b26a00; }
.pill.image { background:#e8f6ef; color:#1b7a4b; }
.pill.page { background:#f0f2f5; color:#5a6472; }
.pill.summary { background:#f3edff; color:#6b3fbf; }
.pill.score { background:#fdeef0; color:#c0392b; }
.ev-body { font-size:13px; color:#3d4756; line-height:1.65; white-space:pre-wrap;
           word-break:break-word; max-height:170px; overflow:auto; background:#fff;
           border:1px dashed #e4e9f2; border-radius:8px; padding:8px 10px; }
.ev-asset { margin-top:8px; text-align:center; background:#fff;
            border:1px solid #e4e9f2; border-radius:8px; padding:6px; }
.ev-asset img { max-width:100%; max-height:230px; }
.ev-cap { font-size:12px; color:#5a6472; margin-top:4px; font-style:italic; }
.ev-empty { color:#8a93a2; font-size:13px; padding:12px; }
.sec-title { font-weight:700; color:#1f2d3d; margin:10px 0 6px 0; }
.md-table { font-size:12px; color:#3d4756; background:#fff; border:1px dashed #e4e9f2;
            border-radius:8px; padding:8px 10px; overflow:auto; max-height:260px;
            white-space:pre; font-family:Consolas, monospace; }
/* Markdown → HTML 表格（文献库详情 / 证据卡里的表格块） */
.tbl-cap { font-size:12px; color:#5a6472; font-style:italic; margin:2px 0 6px 0; }
.md2tbl { border-collapse:collapse; width:100%; font-size:12px; color:#3d4756;
          background:#fff; margin:4px 0; }
.md2tbl th { background:#f6f8fc; color:#1f2d3d; font-weight:600; }
.md2tbl th, .md2tbl td { border:1px solid #e4e9f2; padding:3px 8px;
                         text-align:center; white-space:nowrap; }
.md2tbl tr:hover td { background:#f8faff; }
"""

EMPTY_EVIDENCE = '<div class="ev-empty">暂无证据——提问后，这里会列出回答所依据的文献块（含页码与相关度）。</div>'

BANNER_MD = (
    "# 📚 文献检索助手\n"
    "面向研究生论文场景的检索系统：**解析 PDF 正文 / 表格 / 插图** → "
    "**混合检索（向量+BM25+RRF）→ Rerank 精排 → 带页码引用生成**。"
    "命中的图表会在下方同步展示原图，可逐条核对。\n\n"
    f"- 解析后端：`{backend()}`（未装 PyMuPDF 时表格/插图不可用）\n"
    "- 使用入口：左侧直接提问；想浏览文献全貌去「📚 文献库」\n"
    "- 两类问题都支持：**明细型**（「这篇论文的方法是什么」）走检索；"
    "**汇总型**（「这些文献的共性是什么」）自动切到全库摘要模式"
)


# ============================== 渲染工具 ==============================

def md_table_html(md: str, max_len: int = 4000) -> str:
    """把表格的 Markdown 文本转成真正的 HTML <table>。

    为什么：直接把 Markdown 原样贴出来满屏都是 "|" 竖线，研究生核对数据
    很费劲；渲染成带边框的表格后，行列一目了然。
    结构：表格行以外的行（如 "TABLE I ..." 表题）渲染成灰色小标题。
    """
    md = md[:max_len]
    caption_lines, rows = [], []
    for line in md.split("\n"):
        s = line.strip()
        if s.startswith("|") and s.endswith("|") and s.count("|") >= 2:
            cells = [c.strip() for c in s.strip("|").split("|")]
            if cells and all(set(c) <= set("-: ") for c in cells):
                continue  # 跳过 |---|---| 分隔行
            rows.append(cells)
        elif s:
            caption_lines.append(s)

    if not rows:
        return f'<div class="ev-body">{esc(md)}</div>'

    # 跨列合并的表头常会产生"几乎全空"的残行（如只有 Model 一个词的那行），
    # 把开头的稀疏行归并进表题区，让表格主体只剩有效数据行
    while rows and sum(1 for c in rows[0] if c) <= 1:
        caption_lines.extend(c for c in rows.pop(0) if c)

    width = max(len(r) for r in rows)
    html = "".join(f'<div class="tbl-cap">{esc(l)}</div>' for l in caption_lines)
    head = rows[0] + [""] * (width - len(rows[0]))
    html += "<table class='md2tbl'><tr>" + "".join(
        f"<th>{esc(c)}</th>" for c in head) + "</tr>"
    for r in rows[1:]:
        r = r + [""] * (width - len(r))
        html += "<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>"
    return html + "</table>"


def render_stats() -> str:
    """顶部指标卡：文献数 / 总块数 / 表格数 / 插图数。"""
    try:
        stats = library_stats()
    except Exception:
        return ""
    items = "".join(
        f'<div class="stat"><div class="num">{v}</div><div class="lbl">{k}</div></div>'
        for k, v in stats.items()
    )
    return f'<div class="stat-card">{items}</div>'


def _type_pill(ctype: str) -> str:
    label = {"text": "正文", "table": "表格", "image": "插图", "summary": "摘要"}.get(ctype, "正文")
    cls = "" if ctype == "text" else ctype
    return f'<span class="pill {cls}">{label}</span>'


def render_evidence(ctx: dict) -> str:
    """把本轮"最终送入 LLM 的块"渲染成证据卡片（文献名 + 页码 + 分数 + 原文/原图）。"""
    pairs = ctx.get("kept_scored") or []
    if not pairs:
        return EMPTY_EVIDENCE

    cards = []
    for i, (doc, score) in enumerate(pairs, 1):
        ctype = doc_type(doc)
        meta = doc.metadata
        asset = meta.get("asset", "")

        score_html = (
            f'<span class="pill score">相关度 {score:.2f}</span>'
            if score is not None else ""
        )
        head = (
            f'<div class="ev-head"><span class="pill">[{i}]</span>'
            f'<span class="ev-name">{esc(meta.get("doc_name", "未知"))}</span>'
            f'{_type_pill(ctype)}'
            f'<span class="pill page">第 {meta.get("page") or "-"} 页</span>'
            f'{score_html}</div>'
        )

        # 图片块的正文只有一句"位于第几页"，展示图注更有意义
        body = meta.get("caption") or doc.page_content if ctype == "image" else doc.page_content
        if ctype == "table":
            # 表格块渲染成真正的 HTML 表格（满屏竖线的 Markdown 不好读）
            body_html = f'<div style="overflow:auto; max-height:260px">{md_table_html(body)}</div>'
        else:
            body_html = f'<div class="ev-body">{esc(body[:600])}</div>'

        asset_html = ""
        if asset and os.path.exists(asset):
            src = f"/file={os.path.abspath(asset).replace(chr(92), '/')}"
            cap = (meta.get("caption") or "").replace("\n", " ")
            asset_html = (
                f'<div class="ev-asset"><img src="{src}" alt="asset"/>'
                f'<div class="ev-cap">{esc(cap[:160])}</div></div>'
            )

        cards.append(f'<div class="ev-card">{head}{body_html}{asset_html}</div>')
    return "\n".join(cards)


def evidence_images(ctx: dict) -> list:
    """本轮命中的表格/插图 → gr.Gallery 的 (图片, 说明) 列表。"""
    out = []
    for doc, _s in ctx.get("kept_scored", []):
        asset = doc.metadata.get("asset", "")
        if asset and os.path.exists(asset):
            cap = f"第{doc.metadata.get('page')}页 · {doc.metadata.get('caption') or doc_type(doc)}"
            out.append((asset, cap.replace("\n", " ")[:120]))
    return out


def render_detail(ctx: dict) -> str:
    """检索过程详情（白盒化：每一步都可解释）。"""
    lines = []
    if ctx.get("mode") == "global":
        lines.append(
            f"🔭 **全库汇总模式**（{ctx.get('mode_reason')}）"
            f"：绕过 Top-K 检索，直接把 **{len(ctx['chunks'])} 篇文献的摘要**送进 LLM"
        )
    elif ctx.get("mode_reason"):
        lines.append(f"⚠️ {ctx['mode_reason']}")
    if ctx.get("rewritten"):
        lines.append(f"**查询改写**：`{ctx['question']}` → `{ctx['rewritten']}`")
    if len(ctx.get("search_queries", [])) > 1:
        lines.append(
            "**查询变体**（多语言检索）：" + " ｜ ".join(f"`{q}`" for q in ctx["search_queries"])
        )
    lines.append(f"**耗时**：{ctx['timings']} ｜ 解析后端：`{backend()}`")
    if ctx["recall"].get("quota"):
        # 类型保底配额补入的块（否则 Rerank 根本看不到它们，见 hybrid_retriever._type_quota）
        lines.append("**类型保底补入**：" + "、".join(ctx["recall"]["quota"]))

    recall = ctx["recall"]
    lines.append(f"\n**召回通道**（混合检索={'开启' if recall['hybrid'] else '关闭'}，各展示前 5）")

    if recall["vector"]:
        lines.append("\n*向量通道（语义）*")
        lines.append("| # | 距离 | 来源 | 内容预览 |")
        lines.append("|---|---|---|---|")
        for i, hit in enumerate(recall["vector"][:5], 1):
            lines.append(f"| {i} | {hit['score']} | {hit['name']} | {hit['preview']} |")

    if recall["bm25"]:
        lines.append("\n*BM25 通道（关键词）*")
        lines.append("| # | 分数 | 来源 | 内容预览 |")
        lines.append("|---|---|---|---|")
        for i, hit in enumerate(recall["bm25"][:5], 1):
            lines.append(f"| {i} | {hit['score']} | {hit['name']} | {hit['preview']} |")

    if ctx["scored"]:
        lines.append(f"\n**Rerank 精排**（阈值 {config.RERANK_THRESHOLD}）")
        lines.append("| # | 分数 | 判定 | 来源 | 内容预览 |")
        lines.append("|---|---|---|---|---|")
        for i, (doc, score) in enumerate(ctx["scored"][:10], 1):
            name = cite_label(doc)
            if score is None:
                lines.append(f"| {i} | — | 未精排 | {name} | {preview(doc, 24)} |")
            else:
                flag = "保留" if score >= config.RERANK_THRESHOLD else "丢弃"
                lines.append(f"| {i} | {score:.3f} | {flag} | {name} | {preview(doc, 24)} |")

    lines.append(f"\n**最终送入 LLM：{len(ctx['chunks'])} 个文本块**")
    return "\n".join(lines)


def render_doc_detail(name: str) -> str:
    """文献库 · 单篇详情 HTML。"""
    try:
        d = doc_detail(name)
    except Exception as e:
        return f'<div class="ev-empty">读取失败：{esc(e)}</div>'

    parts = [
        f'<div class="sec-title">📄 {esc(d["doc_name"])}</div>',
        f'<span class="pill page">正文块 {d["n_text"]}</span> '
        f'<span class="pill table">表格 {d["n_table"]}</span> '
        f'<span class="pill image">插图 {d["n_image"]}</span>',
    ]
    if d.get("summary"):
        parts.append('<div class="sec-title">🧾 文献摘要（入库时自动生成，用于"全库汇总"类问题）</div>')
        parts.append(f'<div class="ev-body" style="max-height:none">{esc(d["summary"])}</div>')
    if d["preview"]:
        parts.append(f'<div class="ev-body" style="margin-top:8px">{esc(d["preview"])}</div>')
    for i, t in enumerate(d["tables"], 1):
        parts.append(f'<div class="sec-title">📊 表格 {i}（第 {t["page"]} 页）</div>')
        parts.append(f'<div style="overflow:auto; max-height:320px">{md_table_html(t["markdown"])}</div>')
    if not d["tables"] and not d["images"]:
        parts.append('<div class="ev-empty">该文献未抽取到表格/插图（可能为纯文本型 PDF）。</div>')
    return "\n".join(parts)


def doc_gallery_images(name: str) -> list:
    """文献库 · 单篇的表格/插图画廊。"""
    try:
        d = doc_detail(name)
    except Exception:
        return []
    out = []
    for t in d["tables"]:
        if t["asset"] and os.path.exists(t["asset"]):
            out.append((t["asset"], f"第{t['page']}页 · 表格"))
    for im in d["images"]:
        if im["asset"] and os.path.exists(im["asset"]):
            out.append((im["asset"], f"第{im['page']}页 · {im['caption'][:80]}"))
    return out


# ============================== 交互逻辑 ==============================

def apply_settings(enable_rerank, threshold, top_k, recall_k, enable_hybrid, enable_rewrite,
                   enable_global):
    """把界面上的参数写回全局配置。

    config 是模块级变量，各模块在【调用时】才读取它的值，
    所以运行时修改立刻生效、无需重启（这就是"配置与代码分离"的好处）。
    """
    config.RERANK_ENABLED = bool(enable_rerank)
    config.RERANK_THRESHOLD = float(threshold)
    config.TOP_K = int(top_k)
    config.RECALL_K = int(recall_k)
    config.HYBRID_ENABLED = bool(enable_hybrid)
    config.QUERY_REWRITE_ENABLED = bool(enable_rewrite)
    config.GLOBAL_QA_ENABLED = bool(enable_global)


def respond(message, history, enable_rerank, threshold, top_k, recall_k, enable_hybrid,
            enable_rewrite, enable_global):
    """聊天主流程（生成器：分多次 yield，界面逐步刷新）。

    返回四个组件的值：聊天记录 / 检索详情 / 证据卡 / 命中图表画廊
    """
    if not message.strip():
        yield history, "请输入问题", EMPTY_EVIDENCE, []
        return

    apply_settings(enable_rerank, threshold, top_k, recall_k, enable_hybrid, enable_rewrite,
                   enable_global)
    # Gradio 的历史是 [[问, 答], ...]，转成 [(问, 答)] 供查询改写使用
    past = [(h[0], h[1]) for h in history if len(h) == 2]

    history = history + [[message, ""]]
    yield history, "🔍 检索中…", EMPTY_EVIDENCE, []

    try:
        ctx = prepare(message, history=past)
    except Exception as e:
        history[-1][1] = f"检索失败：{e}"
        yield history, f"检索阶段出错：{e}", EMPTY_EVIDENCE, []
        return

    detail = render_detail(ctx)
    evidence = render_evidence(ctx)
    gallery = evidence_images(ctx)
    yield history, detail, evidence, gallery

    try:
        acc = ""
        for piece in answer_stream(ctx):  # 流式：模型吐一个字，界面就更新一次
            acc += piece
            history[-1][1] = acc
            yield history, detail, evidence, gallery
    except Exception as e:
        history[-1][1] = f"生成失败：{e}"
        yield history, detail, evidence, gallery


def _library_names() -> list:
    """当前文献库里的全部文件名（下拉框选项 / 表格行点击共用）。"""
    try:
        return [r["文献"] for r in library_overview()]
    except Exception:
        return []


def refresh_library():
    """刷新文献清单 + 下拉框选项。

    注意：这里只更新 choices，不连同 value 一起更新——gradio 4.44 在 Tab 内
    对 Dropdown 做「choices+value」组合更新时前端会渲染异常（列表只剩一项、
    无法选择其他文献），拆开更新可绕开这个坑。
    """
    try:
        rows = library_overview()
        df = [[r["文献"], r["页数"], r["正文"], r["表格"], r["摘要"], r["插图"]] for r in rows]
        names = [r["文献"] for r in rows]
        stats_html = render_stats()
        return df, gr.update(choices=names), stats_html
    except Exception as e:
        return [], gr.update(choices=[]), f"读取失败：{e}"


def show_doc(name):
    if not name:
        return "请在左侧选择一篇文献", []
    try:
        return render_doc_detail(name), doc_gallery_images(name)
    except Exception as e:
        # 兜底：单篇解析失败不能让整个 change 事件报错（报错会导致前端
        # 把下拉框的选择弹回旧值，看起来就像"选择不了其他文献"）
        return f'<div class="ev-empty">解析这篇文献时出错：{esc(e)}</div>', []


def on_row_pick(evt: gr.SelectData):
    """点文献表格的某一行 → 直接展示该文献的解析详情（并同步下拉框）。"""
    names = _library_names()
    row = evt.index[0] if evt.index else None
    if row is None or row >= len(names):
        return "请选择一篇文献", [], gr.update()
    name = names[row]
    html, imgs = show_doc(name)
    return html, imgs, gr.update(value=name)


def upload_docs(files) -> str:
    """把上传的文献保存到 docs/ 目录（还没入库，需要点重建）。"""
    if not files:
        return "没有选择文件"
    saved = []
    for f in files:
        path = f if isinstance(f, str) else getattr(f, "name", str(f))
        name = os.path.basename(path)
        shutil.copy(path, os.path.join(DOCS_DIR, name))
        saved.append(name)
    return f"已接收 {len(saved)} 个文件：{', '.join(saved)}\n点击下方「重建知识库」后生效。"


def rebuild() -> str:
    try:
        build_vector_store()
        return ("✅ 知识库已重建完成（旧库已清空，新文献已入库；未改动的文献摘要直接复用缓存）。\n"
                "请到「📚 文献库」查看解析结果与摘要。")
    except Exception as e:
        return f"❌ 重建失败：{e}"


# ============================== 界面搭建 ==============================

def build_demo() -> gr.Blocks:
    """构建界面（单独抽出，方便不启动服务就做自动化测试）。"""
    with gr.Blocks(title="文献检索助手", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
        gr.Markdown(BANNER_MD)
        stats_bar = gr.HTML(render_stats())

        with gr.Tabs():
            # ---------------- Tab 1：智能问答 ----------------
            with gr.Tab("💬 智能问答"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="问答（支持多轮追问，回答自动标注文献与页码）",
                            height=420,
                        )
                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="例如：这篇论文提出的模型整体框架是什么？（问图表相关的问题会展示原图）",
                                show_label=False,
                                scale=4,
                                container=False,
                            )
                            send = gr.Button("发送", variant="primary", scale=1)
                        clear = gr.Button("清空对话")

                        gr.Markdown("### 📎 本轮引用证据")
                        evidence = gr.HTML(EMPTY_EVIDENCE)
                        gallery = gr.Gallery(
                            label="命中的表格 / 插图（原图）",
                            columns=3,
                            height=280,
                            object_fit="contain",
                        )

                    with gr.Column(scale=2):
                        with gr.Accordion("⚙️ 参数面板（实时生效，可现场做实验）", open=True):
                            enable_rerank = gr.Checkbox(label="启用 Rerank 精排", value=config.RERANK_ENABLED)
                            threshold = gr.Slider(
                                0.0, 1.0, value=config.RERANK_THRESHOLD, step=0.05,
                                label="相关性阈值（低于此分数的块丢弃）",
                            )
                            top_k = gr.Slider(1, 8, value=config.TOP_K, step=1, label="TOP_K（送进 LLM 的块数）")
                            recall_k = gr.Slider(4, 30, value=config.RECALL_K, step=1, label="RECALL_K（召回候选数）")
                            enable_hybrid = gr.Checkbox(label="启用混合检索（向量 + BM25 + RRF）", value=config.HYBRID_ENABLED)
                            enable_rewrite = gr.Checkbox(label="启用查询改写（多轮追问）", value=config.QUERY_REWRITE_ENABLED)
                            enable_global = gr.Checkbox(
                                label="启用全库汇总模式（汇总型问题自动改走文献摘要）",
                                value=config.GLOBAL_QA_ENABLED,
                            )

                        detail = gr.Markdown("检索详情会显示在这里")

            # ---------------- Tab 2：文献库 ----------------
            with gr.Tab("📚 文献库"):
                with gr.Row():
                    with gr.Column(scale=2):
                        refresh_btn = gr.Button("🔄 刷新文献列表", variant="primary")
                        lib_table = gr.Dataframe(
                            headers=["文献", "页数", "正文块", "表格", "摘要", "插图"],
                            interactive=False,
                            wrap=True,
                        )
                        # 构建时就填好选项，不依赖 demo.load 动态更新
                        #（gradio 4.44 在 Tab 内动态更新 Dropdown 的 choices 有渲染 bug）
                        doc_pick = gr.Dropdown(
                            label="选择文献查看解析详情",
                            choices=_library_names(),
                            filterable=True,
                        )
                    with gr.Column(scale=3):
                        doc_html = gr.HTML('<div class="ev-empty">选择左侧文献后，这里展示它的解析结果。</div>')
                        doc_gallery = gr.Gallery(
                            label="该文献的表格 / 插图", columns=3, height=420, object_fit="contain"
                        )

            # ---------------- Tab 3：知识库管理 ----------------
            with gr.Tab("📤 知识库管理"):
                gr.Markdown("把新文献传到 `docs/` 后点「重建知识库」。重建会**清空旧库重新入库**（防止重复块堆积），"
                            "并**为每篇文献自动生成摘要**（供「全库汇总」类问题使用）。"
                            "文献没改过时，解析与摘要都走缓存：解析不重复抽 PDF、摘要不重复调 LLM，重建通常只需十几秒。")
                files = gr.File(
                    label="上传文献（pdf / txt / md / docx，可多选）",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".md", ".docx"],
                )
                upload_btn = gr.Button("保存到 docs/")
                rebuild_btn = gr.Button("重建知识库（清空旧库后重新入库）", variant="secondary")
                status = gr.Textbox(label="状态", interactive=False, lines=3)

        settings = [enable_rerank, threshold, top_k, recall_k, enable_hybrid, enable_rewrite,
                    enable_global]

        # ---- 问答事件 ----
        msg.submit(respond, [msg, chatbot] + settings, [chatbot, detail, evidence, gallery]).then(
            lambda: "", None, msg
        )
        send.click(respond, [msg, chatbot] + settings, [chatbot, detail, evidence, gallery]).then(
            lambda: "", None, msg
        )
        clear.click(
            lambda: ([], "检索详情会显示在这里", EMPTY_EVIDENCE, []),
            None, [chatbot, detail, evidence, gallery],
        )

        # ---- 文献库事件 ----
        refresh_btn.click(refresh_library, None, [lib_table, doc_pick, stats_bar])
        doc_pick.change(show_doc, doc_pick, [doc_html, doc_gallery])
        # 点表格某一行 → 直接看该文献详情（同步下拉框选中项）
        lib_table.select(on_row_pick, None, [doc_html, doc_gallery, doc_pick])
        demo.load(refresh_library, None, [lib_table, doc_pick, stats_bar])

        # ---- 知识库事件 ----
        upload_btn.click(upload_docs, files, status)
        rebuild_btn.click(rebuild, None, status)

    return demo


def launch():
    build_demo().launch(allowed_paths=[ASSETS_DIR_ABS])


if __name__ == "__main__":
    launch()
