"""
library.py —— 文献库视图：把向量库里的块按"文献"聚合，供界面展示

为什么不单独维护一个"文献表"，而是从向量库聚合？
  向量库里每条块都带着 doc_name / page / content_type，本身就是最全的清单——
  从它聚合可以保证"文献库页面看到的"和"检索真正在用的"永远是同一份数据，
  不会出现"界面上有这篇文献但检索不到"的两张皮问题。
"""
from utils import TYPE_LABEL, doc_type, page_of, short_name
from vector_store import get_all_chunks


def _doc_name_of(chunk) -> str:
    return chunk.metadata.get("doc_name") or short_name(chunk)


def library_overview(vectordb=None) -> list:
    """所有文献的清单：各类型块数 + 覆盖页数。返回值直接喂给 gr.Dataframe。"""
    agg = {}
    for c in get_all_chunks(vectordb):
        name = _doc_name_of(c)
        row = agg.setdefault(
            name,
            {"文献": name, "页数": 0, "正文": 0, "表格": 0, "插图": 0, "摘要": 0,
             "_pages": set()},
        )
        row[TYPE_LABEL.get(doc_type(c), "正文")] += 1
        p = page_of(c)
        if p:
            row["_pages"].add(p)

    rows = []
    for r in agg.values():
        pages = r.pop("_pages")   # 先弹出内部字段（否则 JSON 序列化会因 set 报错）
        r["页数"] = max(pages) if pages else 0
        rows.append(r)
    return sorted(rows, key=lambda r: r["文献"])


def doc_names(vectordb=None) -> list:
    """文献名列表（界面的下拉框用）。"""
    return [r["文献"] for r in library_overview(vectordb)]


def doc_detail(name: str, vectordb=None) -> dict:
    """单篇文献详情：摘要 + 抽出来的全部表格与插图（带预览图路径）。"""
    chunks = [c for c in get_all_chunks(vectordb) if _doc_name_of(c) == name]

    tables = sorted((c for c in chunks if doc_type(c) == "table"), key=page_of)
    images = sorted((c for c in chunks if doc_type(c) == "image"), key=page_of)
    texts = [c for c in chunks if doc_type(c) == "text"]
    summary = next((c.page_content for c in chunks if doc_type(c) == "summary"), "")

    return {
        "doc_name": name,
        "n_text": len(texts),
        "n_table": len(tables),
        "n_image": len(images),
        "summary": summary,
        "tables": [
            {
                "page": page_of(c),
                "markdown": c.page_content,
                "asset": c.metadata.get("asset", ""),
                "caption": c.metadata.get("caption", ""),
            }
            for c in tables
        ],
        "images": [
            {
                "page": page_of(c),
                "asset": c.metadata.get("asset", ""),
                "caption": c.metadata.get("caption") or c.page_content[:80],
            }
            for c in images
        ],
        "preview": texts[0].page_content[:500] if texts else "",
    }


def library_stats(vectordb=None) -> dict:
    """全局统计（界面顶部的指标卡用）。"""
    rows = library_overview(vectordb)
    return {
        "文献数": len(rows),
        "总块数": sum(r["正文"] + r["表格"] + r["插图"] + r["摘要"] for r in rows),
        "表格数": sum(r["表格"] for r in rows),
        "插图数": sum(r["插图"] for r in rows),
        "摘要数": sum(r["摘要"] for r in rows),
    }


if __name__ == "__main__":
    # 自测：python library.py
    import json

    print("统计:", json.dumps(library_stats(), ensure_ascii=False))
    for row in library_overview():
        print(" ", json.dumps(row, ensure_ascii=False))
