from unittest.mock import MagicMock
from src.retrieval.router import LLMRouter, RouteDecision

def test_router_vector_route():
    llm = MagicMock()
    llm.call.return_value = '{"route":"vector","reason":"单品事实"}'
    router = LLMRouter(llm)
    d = router.route("iPhone 15 电池容量多少")
    assert d.route == "vector"
    assert d.reason == "单品事实"

def test_router_graph_route():
    llm = MagicMock()
    llm.call.return_value = '{"route":"graph","reason":"跨品牌聚合"}'
    router = LLMRouter(llm)
    assert router.route("苹果小米抱怨共同点").route == "graph"

def test_router_invalid_json_fallback_hybrid():
    llm = MagicMock()
    llm.call.return_value = "not json"
    router = LLMRouter(llm)
    assert router.route("xx").route == "hybrid"

def test_router_exception_fallback_hybrid():
    llm = MagicMock()
    llm.call.side_effect = RuntimeError("api down")
    router = LLMRouter(llm)
    assert router.route("xx").route == "hybrid"
