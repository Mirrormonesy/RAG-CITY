import re
from langchain_core.documents import Document
from src.generation.prompts import (
    ANSWER_VECTOR_PROMPT, ANSWER_GRAPH_PROMPT, ANSWER_HYBRID_PROMPT,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CIT_RE = re.compile(r"\[([VG]\d+)\]")

def parse_citations(text: str) -> list[str]:
    seen = []
    for m in _CIT_RE.finditer(text or ""):
        tag = m.group(1)
        if tag not in seen:
            seen.append(tag)
    return seen

def format_context(docs: list[Document], prefix: str = "V") -> tuple[str, dict]:
    lines, mapping = [], {}
    for i, d in enumerate(docs, start=1):
        tag = f"{prefix}{i}"
        lines.append(f"[{tag}] {d.page_content}")
        meta = dict(d.metadata or {})
        meta["content"] = d.page_content
        mapping[tag] = meta
    return "\n\n".join(lines), mapping

class Answerer:
    def __init__(self, llm):
        self.llm = llm

    def answer(self, question: str, docs: list[Document], route: str) -> dict:
        if route == "vector":
            ctx, mapping = format_context(docs, prefix="V")
            prompt = ANSWER_VECTOR_PROMPT.format(question=question, context=ctx)
        elif route == "graph":
            ctx, mapping = format_context(docs, prefix="G")
            prompt = ANSWER_GRAPH_PROMPT.format(question=question, context=ctx)
        else:  # hybrid
            graph_docs = [d for d in docs if d.metadata.get("doc_type") == "community"]
            vec_docs = [d for d in docs if d.metadata.get("doc_type") != "community"]
            g_ctx, g_map = format_context(graph_docs, prefix="G")
            v_ctx, v_map = format_context(vec_docs, prefix="V")
            mapping = {**g_map, **v_map}
            prompt = ANSWER_HYBRID_PROMPT.format(
                question=question, graph_context=g_ctx, vector_context=v_ctx,
            )
        try:
            text = self.llm.call(prompt)
        except Exception as e:
            logger.error(f"Answerer LLM failed: {e}")
            return {"text": "服务暂时不可用,请稍后重试", "citations": {}, "prompt_used": prompt}

        cited_tags = parse_citations(text)
        citations = {tag: mapping[tag] for tag in cited_tags if tag in mapping}
        return {"text": text, "citations": citations, "prompt_used": prompt}
