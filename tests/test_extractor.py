from unittest.mock import MagicMock
from src.indexing.extractor import extract_review_facts, parse_extraction_json

def test_parse_valid_json():
    raw = '{"aspects":["续航","屏幕"],"features":["电池","显示"],"sentiment":"positive"}'
    out = parse_extraction_json(raw)
    assert out["aspects"] == ["续航", "屏幕"]
    assert out["sentiment"] == "positive"

def test_parse_json_with_markdown_fence():
    raw = "```json\n{\"aspects\":[\"a\"],\"features\":[],\"sentiment\":\"neutral\"}\n```"
    out = parse_extraction_json(raw)
    assert out["aspects"] == ["a"]

def test_parse_invalid_json_returns_empty():
    out = parse_extraction_json("not json at all")
    assert out == {"aspects": [], "features": [], "sentiment": "neutral"}

def test_extract_review_facts_calls_llm():
    mock_llm = MagicMock()
    mock_llm.call.return_value = '{"aspects":["续航"],"features":["电池"],"sentiment":"negative"}'
    out = extract_review_facts(mock_llm, "电池一天两充太烂")
    assert out["aspects"] == ["续航"]
    assert out["sentiment"] == "negative"
    mock_llm.call.assert_called_once()
