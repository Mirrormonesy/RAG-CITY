from src.data.loader import load_products, load_reviews


def test_load_products():
    df = load_products("data/raw/sample_products.csv")
    assert len(df) == 2
    assert set(df.columns) >= {"product_id", "title", "category", "brand", "price", "description"}


def test_load_reviews():
    df = load_reviews("data/raw/sample_reviews.csv")
    assert len(df) == 3
    assert set(df.columns) >= {"review_id", "product_id", "rating", "content"}
