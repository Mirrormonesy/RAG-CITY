from unittest.mock import MagicMock
import pandas as pd
from src.indexing.vector_builder import build_vector_index, format_product_chunk, format_review_chunk

def test_format_product_chunk():
    row = {"product_id": "P1", "title": "iPhone 15", "brand": "苹果", "category": "手机", "description": "旗舰手机"}
    text = format_product_chunk(row)
    assert "iPhone 15" in text
    assert "苹果" in text
    assert "手机" in text
    assert "旗舰手机" in text

def test_format_review_chunk():
    row = {"review_id": "R1", "content": "很好用"}
    text = format_review_chunk(row)
    assert text == "很好用"

def test_build_vector_index_writes_chroma(tmp_path):
    products = pd.DataFrame([
        {"product_id": "P1", "title": "iPhone 15", "brand": "苹果", "category": "手机", "description": "旗舰"}
    ])
    reviews = pd.DataFrame([
        {"review_id": "R1", "product_id": "P1", "user_id": "U1", "rating": 5, "content": "好手机推荐购买"},
    ])
    fake_emb = MagicMock()
    fake_emb.embed_documents = MagicMock(return_value=[[0.1]*1024, [0.2]*1024])
    fake_emb.embed_query = MagicMock(return_value=[0.1]*1024)
    persist = str(tmp_path / "chroma")
    db = build_vector_index(products, reviews, fake_emb, persist)
    assert db._collection.count() == 2
