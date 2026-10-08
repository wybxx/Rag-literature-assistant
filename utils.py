"""
utils.py —— 小工具函数

放在这里而不是散落在各个文件里，避免重复代码（DRY 原则：Don't Repeat Yourself）。
"""

# 内容类型 → 中文名（界面徽章、引用标注都用它）
# summary 是入库时给每篇文献生成的"摘要块"，用于回答全库汇总型问题
TYPE_LABEL = {"text": "正文", "table": "表格", "image": "插图", "summary": "摘要"}


def short_name(doc) -> str:
    """从 Document 的 metadata 里取出文件名（去掉又长又乱的完整路径）。

    Windows 路径用反斜杠，所以按 chr(92) 切分——这样写比 '\\' 更不容易出错。
    """
    # 优先用 doc_name（抽取阶段就存好了，比从路径里切更可靠）
    name = doc.metadata.get("doc_name")
    if name:
        return name
    source = doc.metadata.get("source", "未知来源")
    return source.split(chr(92))[-1].split("/")[-1]  # 兼容 Windows 和 Linux 路径


def doc_type(doc) -> str:
    """取内容类型：text / table / image。"""
    return doc.metadata.get("content_type", "text")


def type_label(doc) -> str:
    """取内容类型的中文名：正文 / 表格 / 插图。"""
    return TYPE_LABEL.get(doc_type(doc), "正文")


def page_of(doc) -> int:
    """取页码（非分页文档如 txt/md 返回 0）。"""
    p = doc.metadata.get("page") or 0
    try:
        return int(p)
    except (TypeError, ValueError):
        return 0


def preview(doc, n: int = 40) -> str:
    """取文本块开头 n 个字符做预览（把换行压成空格，避免打印时串行）。"""
    return doc.page_content[:n].replace("\n", " ")


def cite_label(doc, with_type: bool = True) -> str:
    """把一块内容转成人类可读的引用标签。

    例：'DHyMamba_final.pdf · 第 5 页 · 插图'
        'RAG入门介绍.md · 正文'      （txt/md 没有页码）
    """
    parts = [short_name(doc)]
    page = page_of(doc)
    if page:
        parts.append(f"第 {page} 页")
    if with_type:
        parts.append(type_label(doc))
    return " · ".join(parts)


def esc(text) -> str:
    """转义 HTML 特殊字符——把模型/文档内容嵌进 HTML 前必须做，防注入也好防排版崩。"""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
