import pandas as pd
from src.data.cleaner import clean_reviews, normalize_brand, BRAND_ALIASES

def test_filter_length():
    df = pd.DataFrame({
        "review_id": ["R1", "R2", "R3"],
        "product_id": ["P1", "P1", "P1"],
        "user_id": ["U1", "U2", "U3"],
        "rating": [5, 4, 3],
        "content": ["太短", "a"*400, "a"*700],
    })
    out = clean_reviews(df, min_len=10, max_len=500)
    assert len(out) == 1
    assert out.iloc[0]["review_id"] == "R2"

def test_dedup_same_user_same_product():
    df = pd.DataFrame({
        "review_id": ["R1", "R2"],
        "product_id": ["P1", "P1"],
        "user_id": ["U1", "U1"],
        "rating": [5, 4],
        "content": ["a"*50, "a"*60],
    })
    out = clean_reviews(df, min_len=10, max_len=500)
    assert len(out) == 1

def test_filter_emoji_only():
    df = pd.DataFrame({
        "review_id": ["R1", "R2"],
        "product_id": ["P1", "P2"],
        "user_id": ["U1", "U2"],
        "rating": [5, 4],
        "content": ["😀😀😀😀😀😀😀😀😀😀😀😀", "这个商品真的非常棒,推荐给大家"],
    })
    out = clean_reviews(df, min_len=10, max_len=500)
    assert len(out) == 1
    assert out.iloc[0]["review_id"] == "R2"

def test_normalize_brand():
    assert normalize_brand("Apple") == "苹果"
    assert normalize_brand("APPLE") == "苹果"
    assert normalize_brand("苹果") == "苹果"
    assert normalize_brand("未知品牌") == "未知品牌"
