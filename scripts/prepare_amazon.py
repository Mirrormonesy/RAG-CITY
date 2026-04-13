"""Sample Amazon 2018 dataset into products.csv + reviews.csv.

Strict quality gates per category:
- Cell Phones: require meta['brand'] AND meta['description'] non-empty
- Luxury Beauty: require brand extractable from title via " - " / " by " rule

Keep shuffling hot products until target count is reached.
"""
from __future__ import annotations

import argparse
import html
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
SPLIT_BRAND_RE = re.compile(r"\s+-\s+|\s+by\s+", re.IGNORECASE)

CATEGORIES = {
    "手机": {
        "reviews": RAW_DIR / "Cell_Phones_and_Accessories_5.json",
        "meta": RAW_DIR / "meta_Cell_Phones_and_Accessories.json",
        "strategy": "meta_brand",
    },
    "美妆": {
        "reviews": RAW_DIR / "All_Beauty_5.json",
        "meta": RAW_DIR / "meta_All_Beauty.json",
        "strategy": "meta_brand",
    },
}


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(str(s))
    s = HTML_TAG_RE.sub(" ", s)
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s


def parse_price(raw: str) -> str:
    if not raw:
        return ""
    m = re.search(r"(\d+(?:\.\d+)?)", str(raw))
    return m.group(1) if m else ""


def extract_brand_from_title(title: str) -> Optional[str]:
    """Split rule: '<brand> - <rest>' or '<rest> by <brand>'. Returns None if unclear."""
    if not title:
        return None
    parts = SPLIT_BRAND_RE.split(title, maxsplit=1)
    if len(parts) < 2:
        return None
    # If "by" used, brand is the RIGHT segment; else brand is the LEFT segment
    if re.search(r"\s+by\s+", title, re.IGNORECASE):
        brand = parts[1].strip()
    else:
        brand = parts[0].strip()
    # Quality filter: 2-40 chars, not mostly punctuation/digits
    if not (2 <= len(brand) <= 40):
        return None
    alpha = sum(c.isalpha() for c in brand)
    if alpha < 2:
        return None
    return brand


def load_meta(path: Path) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = obj.get("asin")
            if asin:
                meta[asin] = obj
    return meta


def load_reviews_grouped(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = obj.get("asin")
            text = obj.get("reviewText", "").strip()
            if asin and text:
                grouped[asin].append(obj)
    return grouped


def build_product_record(
    asin: str, m: dict, cat_label: str, strategy: str
) -> Optional[dict]:
    """Apply quality gates. Return None if product must be skipped."""
    title = clean_text(m.get("title") or "")
    if not title or len(title) < 5:
        return None

    desc_list = m.get("description") or []
    if isinstance(desc_list, list):
        description = " ".join(clean_text(d) for d in desc_list if d)
    else:
        description = clean_text(str(desc_list))

    if strategy == "meta_brand":
        brand = clean_text(m.get("brand") or "")
        if not brand:
            return None
        if not description:  # Cell Phones require description
            return None
    elif strategy == "title_split":
        brand = extract_brand_from_title(title)
        if not brand:
            return None
        # description optional for beauty; fallback to title
        if not description:
            description = title
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return {
        "product_id": asin,
        "title": title,
        "category": cat_label,
        "brand": brand,
        "price": parse_price(m.get("price", "")),
        "description": description[:1000],
    }


def sample_category(
    cat_label: str,
    paths: dict,
    n_products: int,
    n_reviews_per_product: int,
    min_reviews: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    print(f"[{cat_label}] loading meta ...")
    meta = load_meta(paths["meta"])
    print(f"[{cat_label}] loading reviews ...")
    grouped = load_reviews_grouped(paths["reviews"])

    hot = [asin for asin, rs in grouped.items() if len(rs) >= min_reviews and asin in meta]
    print(f"[{cat_label}] hot products (>= {min_reviews} reviews): {len(hot)}")

    rng = random.Random(seed)
    rng.shuffle(hot)

    products_out: list[dict] = []
    reviews_out: list[dict] = []
    skipped = 0
    strategy = paths["strategy"]

    for asin in hot:
        if len(products_out) >= n_products:
            break
        rec = build_product_record(asin, meta[asin], cat_label, strategy)
        if rec is None:
            skipped += 1
            continue
        products_out.append(rec)

        pool = grouped[asin]
        sample_n = min(n_reviews_per_product, len(pool))
        picked = rng.sample(pool, sample_n)
        for r in picked:
            content = clean_text(r.get("reviewText", ""))
            if len(content) < 10:
                continue
            reviews_out.append({
                "review_id": f"{asin}_{r.get('reviewerID', 'X')}_{r.get('unixReviewTime', 0)}",
                "product_id": asin,
                "user_id": r.get("reviewerID", ""),
                "rating": int(r.get("overall", 0)),
                "content": content,
            })

    print(
        f"[{cat_label}] kept {len(products_out)} products, "
        f"skipped {skipped} (failed quality gate), "
        f"{len(reviews_out)} reviews"
    )
    if len(products_out) < n_products:
        print(
            f"[{cat_label}] WARNING: only {len(products_out)}/{n_products} products met the gate"
        )
    return products_out, reviews_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-products", type=int, default=25, help="products per category")
    ap.add_argument("--n-reviews", type=int, default=20, help="reviews per product")
    ap.add_argument("--min-reviews", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-products", default=str(RAW_DIR / "products.csv"))
    ap.add_argument("--out-reviews", default=str(RAW_DIR / "reviews.csv"))
    args = ap.parse_args()

    all_products, all_reviews = [], []
    for cat_label, paths in CATEGORIES.items():
        ps, rs = sample_category(
            cat_label=cat_label,
            paths=paths,
            n_products=args.n_products,
            n_reviews_per_product=args.n_reviews,
            min_reviews=args.min_reviews,
            seed=args.seed,
        )
        all_products.extend(ps)
        all_reviews.extend(rs)

    pd.DataFrame(all_products).to_csv(args.out_products, index=False, encoding="utf-8-sig")
    pd.DataFrame(all_reviews).to_csv(args.out_reviews, index=False, encoding="utf-8-sig")

    print(f"\nTotal: {len(all_products)} products, {len(all_reviews)} reviews")
    print(f"Written: {args.out_products}")
    print(f"Written: {args.out_reviews}")


if __name__ == "__main__":
    main()
