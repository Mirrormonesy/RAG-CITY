import pytest
import os

@pytest.mark.skipif(not os.environ.get("RUN_INTEGRATION"), reason="需要真 API 和索引")
def test_end_to_end():
    from src.factory import build_hybrid_rag
    rag = build_hybrid_rag()
    result = rag.query("iPhone 15 的电池怎么样?")
    assert result["text"]
    assert result["route"] in {"vector", "graph", "hybrid"}
