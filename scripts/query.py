"""Interactive end-to-end RAG test.
Usage:
  python scripts/query.py                      # interactive loop
  python scripts/query.py "你的问题"           # single query
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.factory import build_hybrid_rag


def _doc_line(i: int, d, max_len: int = 200) -> str:
    meta = d.metadata or {}
    tag = meta.get("doc_type") or meta.get("source") or "-"
    extra = ""
    if meta.get("community_id") is not None:
        extra = f" cid={meta['community_id']}"
    if meta.get("product_id"):
        extra += f" pid={meta['product_id']}"
    if meta.get("review_id"):
        extra += f" rid={meta['review_id']}"
    content = (d.page_content or "").replace("\n", " ")
    if len(content) > max_len:
        content = content[:max_len] + "..."
    return f"  [{i}] ({tag}{extra}) {content}"


def print_result(q: str, res: dict) -> None:
    print("\n" + "=" * 80)
    print(f"Q: {q}")
    print("-" * 80)
    print(f"[Route] {res['route']}  ({res['route_reason']})")

    v_docs = res.get("vector_docs") or []
    g_docs = res.get("graph_docs") or []
    fused = res.get("retrieved_docs") or []
    reranked = res.get("reranked_docs") or []

    print("-" * 80)
    print(f"[Vector recall] {len(v_docs)} docs (showing top 5)")
    for i, d in enumerate(v_docs[:5], 1):
        print(_doc_line(i, d))

    print("-" * 80)
    print(f"[Graph recall] {len(g_docs)} docs (showing all)")
    for i, d in enumerate(g_docs, 1):
        print(_doc_line(i, d, max_len=300))

    print("-" * 80)
    print(f"[Fused] {len(fused)} docs (showing top 10)")
    for i, d in enumerate(fused[:10], 1):
        print(_doc_line(i, d))

    print("-" * 80)
    print(f"[Reranked] {len(reranked)} docs")
    for i, d in enumerate(reranked, 1):
        print(_doc_line(i, d, max_len=300))

    print("-" * 80)
    print("A:")
    print(res["text"])
    if res.get("citations"):
        print("-" * 80)
        print("[Citations]")
        for k, v in res["citations"].items():
            preview = v if isinstance(v, str) else str(v)
            print(f"  {k}: {preview[:120]}")
    print("=" * 80)


def main() -> None:
    print("Loading HybridRAG ...")
    rag = build_hybrid_rag(str(PROJECT_ROOT / "configs" / "config.yaml"))
    print("Ready.\n")

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print_result(q, rag.query(q))
        return

    while True:
        try:
            q = input("Q > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break
        try:
            res = rag.query(q)
            print_result(q, res)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
