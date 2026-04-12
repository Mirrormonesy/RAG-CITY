import json
import re
from src.generation.prompts import EXTRACT_PROMPT

_DEFAULT = {"aspects": [], "features": [], "sentiment": "neutral"}

def parse_extraction_json(raw: str) -> dict:
    if not raw:
        return dict(_DEFAULT)
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return dict(_DEFAULT)
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return dict(_DEFAULT)
    return {
        "aspects": list(data.get("aspects", []))[:5],
        "features": list(data.get("features", []))[:5],
        "sentiment": data.get("sentiment", "neutral"),
    }

def extract_review_facts(llm, review_content: str) -> dict:
    prompt = EXTRACT_PROMPT.format(review=review_content)
    try:
        raw = llm.call(prompt)
    except Exception:
        return dict(_DEFAULT)
    return parse_extraction_json(raw)
