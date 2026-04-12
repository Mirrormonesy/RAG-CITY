import json
import re
from dataclasses import dataclass
from src.generation.prompts import ROUTER_PROMPT
from src.utils.logger import get_logger

logger = get_logger(__name__)

_VALID_ROUTES = {"vector", "graph", "hybrid"}

@dataclass
class RouteDecision:
    route: str
    reason: str

class LLMRouter:
    def __init__(self, llm):
        self.llm = llm

    def route(self, query: str) -> RouteDecision:
        prompt = ROUTER_PROMPT.format(query=query)
        try:
            raw = self.llm.call(prompt)
        except Exception as e:
            logger.warning(f"Router LLM failed: {e}, fallback to hybrid")
            return RouteDecision("hybrid", "fallback:llm_error")

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return RouteDecision("hybrid", "fallback:no_json")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return RouteDecision("hybrid", "fallback:json_parse")

        route = data.get("route", "").strip().lower()
        if route not in _VALID_ROUTES:
            return RouteDecision("hybrid", "fallback:invalid_route")
        return RouteDecision(route, data.get("reason", ""))
