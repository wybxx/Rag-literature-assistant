"""
pdf_extractor.py —— 文献结构化抽取：正文 / 表格 / 插图

【为什么不用 PyPDFLoader】
pypdf 只能按页抽出一串纯文本：表格会塌成错行文字、插图直接丢失。
对"文献检索助手"来说这是硬伤——研究生问"表 3 的指标是多少""图 2 的方法结构"，
光有正文根本答不了。

【本文件用 PyMuPDF（import fitz）做三件事】
1. 正文：按"文本块"抽取，并【剔除表格区域】，避免同一段内容被抽两遍；
2. 表格：识别 → 转成 Markdown 文本（可直接进向量库做语义检索），
        同时把表格区域渲染成 PNG 存到 assets/（供界面展示原表）；
3. 插图：把图片区域渲染成 PNG 存盘，并抓取最近的图注（"Figure 3: ..."）
        作为它的检索文本——这样"图注检索"才有效。

【一个真实的坑（已在本项目实测）】
PyMuPDF 的 find_tables 有三种策略：
  - lines / lines_strict：依据表格边框线，可靠，但无边框的三线表会漏
    （实测：4 篇 IEEE 论文 14 页全部 0 命中——学术表格大多只有横线没有竖框）；
  - text：依据文本对齐，能抓无框表，但极易把普通正文误判成表格
    （实测：双栏论文每页被判成一张 69 行 × 8 列、覆盖 77% 页面的"表"）。
所以本文件的策略是逐页三级回退：
  ① lines 法（有框线时最可靠）；
  ② mixed 法（横线定行 + 文本对齐定列，专治三线表），并且**两段式提取**：
     mixed 只负责"定位"表格区域，再在区域内用 text 法重新精提——
     text 法在全页范围误报严重，但在已知的小区域内列对齐非常准
     （实测把 "83.60 84.80" 挤在一个格子里的列全部拆干净）；
     另按"数字单元格占比 ≥ 0.3"过滤公式区误判（真实验表 0.62~0.97，公式区 0.10）；
  ③ text 法兜底 + 严格过滤（行列上限 / 面积上限 / 非空率下限）。

抽取出的每条内容都是 LangChain 的 Document，metadata 统一包含：
  doc_name / page / content_type(text|table|image) / asset / caption
"""
import os
import re

from langchain_core.documents import Document

import config

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:  # 优雅降级：没装也能跑，只是表格/插图不可用
    _HAS_FITZ = False

# 抽取逻辑版本号：凡是改动了抽取行为（策略、过滤规则、字段结构……）都要 +1。
# loader 的抽取缓存把这个版本号算进缓存键里，版本一变旧缓存自动全部失效，
# 避免"改了代码却读出旧结果"的脏缓存问题。
EXTRACTOR_VERSION = 1


CAPTION_RE = re.compile(r"^\s*(fig(?:ure)?\.?|tab(?:le)?\.?|图|表)\s*\.?\s*\d+", re.IGNORECASE)
_SAFE_RE = re.compile(r"[^\w\u4e00-\u9fff\-]+")


def backend() -> str:
    """当前 PDF 解析后端：pymupdf（完整能力）或 pypdf（仅纯文本兜底）。"""
    return "pymupdf" if _HAS_FITZ else "pypdf"


# ------------------------------------------------------------------ 基础工具

def _safe_stem(name: str) -> str:
    return _SAFE_RE.sub("_", os.path.splitext(name)[0]) or "doc"


def _asset_path(doc_name: str, page: int, kind: str, idx: int) -> str:
    """生成资产文件路径，如 assets/DHyMamba_final/p005_table1.png。"""
    folder = os.path.join(config.ASSETS_DIR, _safe_stem(doc_name))
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"p{page:03d}_{kind}{idx}.png")


def _render(page, rect, path: str) -> None:
    """把页面上某个矩形区域渲染成 PNG 存盘（表格预览图、插图都用它）。"""
    pix = page.get_pixmap(clip=rect, dpi=config.RENDER_DPI)
    pix.save(path)


def _table_to_markdown(rows: list) -> str:
    """把二维单元格数组转成 Markdown 表格文本。

    为什么转 Markdown 而不是留二维数组？
      ① Markdown 是纯文本，能直接送进 embedding 模型做语义检索；
      ② LLM 天然读得懂 Markdown 表格，回答时能准确引用表里的数字。
    """
    clean = []
    for r in rows:
        cells = [("" if c is None else str(c)).replace("\n", " ").replace("|", "/").strip() for c in r]
        if any(cells):
            clean.append(cells)
    if len(clean) < 2:
        return ""
    width = max(len(r) for r in clean)
    clean = [r + [""] * (width - len(r)) for r in clean]
    head, body = clean[0], clean[1:]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _text_blocks(page) -> list:
    """返回 [(rect, text)]：既用于找图注，也用于剔除表格区域。"""
    out = []
    for b in page.get_text("blocks"):
        if len(b) >= 7 and b[6] != 0:
            continue  # 跳过图片块，只保留文本块
        rect = fitz.Rect(b[0], b[1], b[2], b[3])
        txt = (b[4] or "").strip()
        if txt:
            out.append((rect, txt))
    return out


def _nearest_caption(blocks: list, rect) -> str:
    """找离图表最近的图注，如 "Figure 3: The architecture of ..."。

    优先级：命中图注正则 > 距离近。只在正下方 90pt / 正上方 40pt 内找，
    且要求横向有重叠——避免把隔壁栏的文字错当图注。
    """
    cands = []
    for brect, txt in blocks:
        below = 0 <= (brect.y0 - rect.y1) <= 90
        above = 0 <= (rect.y0 - brect.y1) <= 40
        if not (below or above):
            continue
        if not (brect.x0 < rect.x1 and brect.x1 > rect.x0):  # 横向无重叠
            continue
        dist = (brect.y0 - rect.y1) if below else (rect.y0 - brect.y1)
        cands.append(((0 if CAPTION_RE.match(txt) else 1, dist), txt))
    if not cands:
        return ""
    cands.sort(key=lambda x: x[0])
    return cands[0][1].strip()[:300]


def _mk_doc(content: str, source: str, doc_name: str, page: int,
            ctype: str, asset: str, caption: str) -> Document:
    """统一构造 Document——所有抽取路径都从这里出去，保证 metadata 结构一致。"""
    return Document(
        page_content=content,
        metadata={
            "source": source,           # 完整文件路径（兼容 LangChain 习惯）
            "doc_name": doc_name,       # 文件名（界面展示、引用标注用）
            "page": int(page or 0),     # 页码，0 表示不适用（txt/md 无页码）
            "content_type": ctype,      # text / table / image
            "asset": asset or "",       # 表格/插图预览图路径（正文为空串）
            "caption": (caption or "")[:300],
        },
    )


# ------------------------------------------------------------------ 表格识别

def _digit_ratio(rows) -> float:
    """非空单元格中"含数字"的占比。

    用途：区分真实验表格（必然数字密集，实测 0.62~0.97）和
    公式区误判（几乎没有数字，实测 0.10）。
    """
    cells = [str(c).strip() for r in rows for c in r if str(c).strip()]
    if not cells:
        return 0.0
    return sum(1 for c in cells if any(ch.isdigit() for ch in c)) / len(cells)


def _locate_tables(page, strategy: str) -> list:
    """按指定策略定位表格，返回 [(rect, rows)]。

    strategy="mixed" 采用两段式提取（实测关键改进）：
      第一段 mixed（横线定行 + 文本对齐定列）只负责"定位"表格区域，
      第二段在区域内用 text 策略重提——text 策略在全页范围会把正文判成
      假表格，但在"已知是表格的小区域"内列对齐非常准，
      能把 "83.60 | 84.80" 这种被挤压的单元格拆干净。
    """
    try:
        if strategy == "mixed":
            tf = page.find_tables(horizontal_strategy="lines",
                                  vertical_strategy="text")
        else:
            tf = page.find_tables(strategy=strategy)
    except Exception:
        return []

    out = []
    for t in tf.tables:
        rect = fitz.Rect(t.bbox)
        rows = t.extract()
        if strategy == "mixed" and rows:
            # 第二段：区域内 text 精提（拿不到就沿用第一段结果）
            try:
                tf2 = page.find_tables(strategy="text", clip=rect)
                if tf2.tables:
                    best = max(tf2.tables, key=lambda x: len(x.extract()))
                    rows = best.extract()
            except Exception:
                pass
            # 数字占比过滤：公式区/排版结构被误判成表格时几乎不含数字
            if _digit_ratio(rows) < 0.3:
                continue
        out.append((rect, rows))
    return out


def _tables_of_page(page, strategy: str, strict: bool) -> list:
    """按指定策略识别本页表格，并用启发式规则过滤误报，返回 [(rect, rows)]。

    strict=True 时启用严格过滤（行列上限、面积上限、非空率下限），
    专治 text 策略"把正文判成表格"的毛病。
    """
    page_area = abs(page.rect.width * page.rect.height) or 1.0
    kept = []
    for rect, rows in _locate_tables(page, strategy):
        if not rows:
            continue
        width = max(len(r) for r in rows)
        if len(rows) < 2 or width < 2:
            continue  # 只有一行/一列，不构成表格

        if strict:
            # mixed 策略要求页面里有真实横线才可能报出表格，误报风险低，
            # 列数上限放宽一些（如 p13 的 13 列实验配置表）
            max_cols = 16 if strategy == "mixed" else 12
            if len(rows) > config.MAX_TABLE_ROWS or width > max_cols:
                continue
            if abs(rect.width * rect.height) > 0.55 * page_area:
                continue  # 占了大半页 → 必是把正文当表格
            filled = sum(1 for r in rows for c in r if c and str(c).strip())
            if filled < 0.5 * len(rows) * width:
                continue  # 空格子过多 → 不是真表格
        kept.append((rect, rows))
    return kept


# ------------------------------------------------------------------ 逐页抽取

def _dense_rule_regions(page) -> list:
    """找"线条密集区"——矢量转曲表格的定位器。

    【最阴险的一类 PDF】有些文档（Word 另存、图表工具导出）把表格整个画成
    矢量图形，连文字都"转曲"成了路径——get_text 读不到任何内容，
    find_tables 也认不出（行线被拆成上百个小线段，连不成网格）。
    实测：Analyzing...pdf 的 TABLE I，search_for("MISA") 都搜不到。

    这种表格的特征：一簇 y 间距均匀、x 范围一致的横线条带（每条由大量
    小线段拼成）。按此特征聚出候选区域，交给调用方处理。
    返回 [(rect, 条带数)]。
    """
    segs = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    for d in drawings:
        for item in d["items"]:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                # 注意不要加"线段长度"下限——实测有的 PDF 把表格行线拆成
                # 上百个 1~2pt 的小线段，任何长度过滤都会把它们全杀掉
                if abs(p1.y - p2.y) < 0.5:
                    segs.append((p1.y, min(p1.x, p2.x), max(p1.x, p2.x)))
            elif item[0] == "re":
                r = item[1]
                if r.height < 2:
                    segs.append((r.y0, r.x0, r.x1))
    if not segs:
        return []

    # 按 y 聚成条带：同一条带 = y 相近的线段集合
    segs.sort()
    bands = []  # [y列表, x0, x1, 线段数]
    for y, x0, x1 in segs:
        if bands and abs(y - bands[-1][0][-1]) < 2:
            bands[-1][0].append(y)
            bands[-1][1] = min(bands[-1][1], x0)
            bands[-1][2] = max(bands[-1][2], x1)
            bands[-1][3] += 1
        else:
            bands.append([[y], x0, x1, 1])
    # 只留"像表格行线"的条带：跨度够宽、线段够多（排除下划线/分数线等零散线条）
    bands = [(b[0][0], b[1], b[2], b[3]) for b in bands
             if (b[2] - b[1]) >= 100 and b[3] >= 20]
    if not bands:
        return []

    # 相邻条带纵向贴近且横向重叠 → 归为同一区域
    groups, cur = [], [bands[0]]
    for b in bands[1:]:
        prev = cur[-1]
        overlap = min(prev[2], b[2]) - max(prev[1], b[1])
        span = min(prev[2] - prev[1], b[2] - b[1])
        if b[0] - prev[0] <= 40 and span > 0 and overlap >= 0.6 * span:
            cur.append(b)
        else:
            groups.append(cur)
            cur = [b]
    groups.append(cur)

    out = []
    for g in groups:
        if len(g) >= 5:  # 至少 5 条行线才像一张表
            rect = fitz.Rect(min(b[1] for b in g) - 2, g[0][0] - 2,
                             max(b[2] for b in g) + 2, g[-1][0] + 2)
            out.append((rect, len(g)))
    return out


def _best_caption(blocks: list, rect) -> str:
    """为矢量图形区域挑最合适的图注。

    普通 _nearest_caption 只选"距离最近"的文本块，但表题/图题常被拆成
    多个文本块，最近的一块可能只是表题的最后一行（实测抓到
    'OWN EXPERIMENTS' 而不是 'TABLE I COMPARISON...'）。
    这里优先选以 TABLE / Fig / Figure 开头的块（图注的标志性前缀），
    找不到再退回最近距离策略。
    """
    cands = []
    for brect, txt in blocks:
        below = 0 <= (brect.y0 - rect.y1) <= 90
        above = 0 <= (rect.y0 - brect.y1) <= 60
        if (below or above) and brect.x0 < rect.x1 and brect.x1 > rect.x0:
            cands.append((brect, txt))
    for _brect, txt in cands:
        if re.match(r"^\s*(TABLE|Fig\.?|Figure)", txt, re.I):
            return txt.replace("\n", " ").strip()[:300]
    return _nearest_caption(blocks, rect)


def _outlined_table_docs(page, blocks, doc_name: str, pno: int, source_path: str,
                         existing_rects: list) -> list:
    """把"矢量转曲表格"区域渲染成插图入库（图注用表题，可被检索）。

    只处理文字几乎读不出来的区域（词数 < 8）——能读出文字的表格
    走正常的表格抽取链路，不在这里抢活。
    """
    out = []
    idx = 0
    for rect, n_bands in _dense_rule_regions(page):
        if any(rect.intersects(r) for r in existing_rects):
            continue  # 已被正常表格/插图覆盖，不重复收
        try:
            n_words = len(page.get_text("words", clip=rect))
        except Exception:
            n_words = 0
        if n_words >= 8:
            continue  # 文字能读到 → 不是转曲表格
        idx += 1
        asset = _asset_path(doc_name, pno, "tbl_img", idx)
        try:
            _render(page, rect, asset)
        except Exception:
            continue
        caption = _best_caption(blocks, rect)
        body = (f"{caption}\n" if caption else "") + \
            "[矢量图形] 该区域在 PDF 中以矢量线条绘制（文字已转曲，可能是表格或图表），" \
            "无法提取文本，请看原图核对内容。"
        out.append(_mk_doc(body, source_path, doc_name, pno, "image", asset, caption))
    return out


def _page_documents(page, doc_name: str, pno: int, tables: list, source_path: str) -> list:
    """抽取单页的正文 + 表格 + 插图。"""
    blocks = _text_blocks(page)
    docs = []

    # ① 表格：转 Markdown 存为可检索文本，同时渲染原表预览图
    table_rects = []
    if config.EXTRACT_TABLES:
        for i, (rect, rows) in enumerate(tables, 1):
            table_rects.append(rect)
            md = _table_to_markdown(rows)
            if not md:
                continue
            asset = _asset_path(doc_name, pno, "table", i)
            try:
                _render(page, rect, asset)
            except Exception:
                asset = ""
            caption = _nearest_caption(blocks, rect)
            body = (f"{caption}\n" if caption else "") + md
            docs.append(_mk_doc(body, source_path, doc_name, pno, "table", asset, caption))

    # ② 正文：剔除表格区域，避免同一段内容被抽两遍（污染检索结果）
    parts = [txt for rect, txt in blocks if not any(rect.intersects(r) for r in table_rects)]
    text = "\n".join(parts).strip()
    if text:
        docs.append(_mk_doc(text, source_path, doc_name, pno, "text", "", ""))

    # ③ 插图
    if config.EXTRACT_IMAGES:
        docs.extend(_images_of_page(page, blocks, doc_name, pno, source_path))
        # 兜底：矢量转曲表格（读不到文字的"线条密集区"）渲染成插图，
        # 虽提取不了数据，但原图可看、表题可检索——好过彻底丢失
        try:
            covered = table_rects + [fitz.Rect(i["bbox"]) for i in page.get_image_info()]
        except Exception:
            covered = table_rects
        docs.extend(_outlined_table_docs(page, blocks, doc_name, pno, source_path, covered))

    return docs


def _images_of_page(page, blocks: list, doc_name: str, pno: int, source_path: str) -> list:
    """抽插图：渲染存盘 + 用图注当检索文本。"""
    try:
        infos = page.get_image_info()
    except Exception:
        return []

    out, idx = [], 0
    for info in infos:
        try:
            rect = fitz.Rect(info["bbox"])
        except Exception:
            continue
        if rect.width * rect.height < config.MIN_IMAGE_AREA:
            continue  # 太小 → 当作噪声（页眉 logo、分隔线、公式碎片）
        idx += 1
        asset = _asset_path(doc_name, pno, "img", idx)
        try:
            _render(page, rect, asset)
        except Exception:
            continue
        caption = _nearest_caption(blocks, rect)
        # 图片本身没法做文本检索，于是拿"图注 + 位置"当它的检索文本：
        # 用户问"图 3 的对比结果"时，就能靠图注里的关键词把这张图召回。
        body = (caption + "\n" if caption else "") + f"[插图] 位于《{doc_name}》第 {pno} 页。"
        out.append(_mk_doc(body, source_path, doc_name, pno, "image", asset, caption))
    return out


# ------------------------------------------------------------------ 对外入口

def extract_pdf(path: str) -> list:
    """抽取一个 PDF 的全部内容，返回 Document 列表（未切分）。"""
    doc_name = os.path.basename(path)
    if not _HAS_FITZ:
        return _fallback_pypdf(path, doc_name)

    doc = fitz.open(path)
    try:
        pages = list(range(doc.page_count))

        # 逐页三级策略：边框法 → 混合法（治三线表）→ 文本对齐法
        by_lines = {p: _tables_of_page(doc[p], "lines", strict=False) for p in pages}
        any_lines = any(by_lines.values())

        out = []
        for p in pages:
            tables = by_lines[p]
            if not tables:
                # 本页边框法没找到表：先试"混合法"（横线定行 + 文本对齐定列，
                # 学术文献的三线表就靠它）；再不行才回退"文本对齐法"，
                # 并启用严格过滤压掉"把整页正文判成表格"的误报
                tables = _tables_of_page(doc[p], "mixed", strict=True)
                if not tables and not any_lines:
                    tables = _tables_of_page(doc[p], "text", strict=True)
            out.extend(_page_documents(doc[p], doc_name, p + 1, tables, path))
        return out
    finally:
        doc.close()


def _fallback_pypdf(path: str, doc_name: str) -> list:
    """降级路径：没装 PyMuPDF 时只能抽纯文本。"""
    print("[提示] 未安装 PyMuPDF，PDF 只能抽纯文本（表格/插图不可用）。"
          "执行 pip install pymupdf 可启用完整抽取。")
    from langchain_community.document_loaders import PyPDFLoader

    docs = PyPDFLoader(path).load()
    for i, d in enumerate(docs):
        d.metadata.update({
            "doc_name": doc_name,
            "page": int(d.metadata.get("page", i)) + 1,  # PyPDFLoader 页码从 0 开始
            "content_type": "text",
            "asset": "",
            "caption": "",
        })
    return docs


if __name__ == "__main__":
    # 自测：python pdf_extractor.py [某篇PDF路径]
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target is None:
        pdfs = [f for f in os.listdir(config.DOCS_DIR) if f.lower().endswith(".pdf")]
        target = os.path.join(config.DOCS_DIR, pdfs[0])

    print(f"解析后端: {backend()}")
    print(f"目标文件: {target}")
    docs = extract_pdf(target)
    kinds = {}
    for d in docs:
        kinds[d.metadata["content_type"]] = kinds.get(d.metadata["content_type"], 0) + 1
    print(f"抽取结果: {len(docs)} 条 -> {kinds}")
    for d in docs:
        if d.metadata["content_type"] != "text":
            print(f"  [{d.metadata['content_type']}] p{d.metadata['page']} "
                  f"asset={os.path.basename(d.metadata['asset'])} "
                  f"caption={d.metadata['caption'][:50]}")
