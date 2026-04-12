import argparse
from pathlib import Path

from src.data.loader import load_products, load_reviews
from src.data.cleaner import clean_products, clean_reviews
from src.utils.config import load_config
from src.utils.logger import get_logger


def main():
    parser = argparse.ArgumentParser(description="Preprocess products and reviews data.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--products-csv", required=True, help="Path to products CSV")
    parser.add_argument("--reviews-csv", required=True, help="Path to reviews CSV")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("preprocess", log_file=cfg["paths"].get("logs", "logs/app.log"))

    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    min_len = cfg["data"]["min_review_length"]
    max_len = cfg["data"]["max_review_length"]

    logger.info(f"Loading products from {args.products_csv}")
    products = load_products(args.products_csv)
    logger.info(f"Loaded {len(products)} products")

    logger.info(f"Loading reviews from {args.reviews_csv}")
    reviews = load_reviews(args.reviews_csv)
    logger.info(f"Loaded {len(reviews)} reviews")

    products_clean = clean_products(products)
    reviews_clean = clean_reviews(reviews, min_len=min_len, max_len=max_len)
    logger.info(f"Cleaned: {len(products_clean)} products, {len(reviews_clean)} reviews")

    products_out = processed_dir / "products.parquet"
    reviews_out = processed_dir / "reviews.parquet"
    products_clean.to_parquet(products_out, index=False)
    reviews_clean.to_parquet(reviews_out, index=False)
    logger.info(f"Saved to {products_out} and {reviews_out}")


if __name__ == "__main__":
    main()
