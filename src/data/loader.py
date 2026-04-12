import pandas as pd


def load_products(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"product_id": str, "brand": str, "category": str})
    required = {"product_id", "title", "category", "brand", "price", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in products: {missing}")
    return df


def load_reviews(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"review_id": str, "product_id": str, "user_id": str})
    required = {"review_id", "product_id", "user_id", "rating", "content"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in reviews: {missing}")
    return df
