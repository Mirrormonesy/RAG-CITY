"""Patch products.csv: extract brand from title when empty, fallback description."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.llm_client import QwenClient  # noqa: E402

PRODUCTS = PROJECT_ROOT / "data" / "raw" / "products.csv"
CONFIG = PROJECT_ROOT / "configs" / "config.yaml"

BRAND_PROMPT = (
    "从下面的电商商品标题中提取品牌名。只输出品牌名本身,不要任何解释、引号、标点。"
    "如果实在无法判断,输出:未知品牌"
)


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    cfg = yaml.safe_load(open(CONFIG, "r", encoding="utf-8"))
    api_key = os.getenv(cfg["qwen"]["api_key_env"])
    client = QwenClient(
        api_key=api_key,
        model=cfg["qwen"]["extract_model"],
        temperature=0.0,
        max_tokens=32,
        timeout=cfg["qwen"]["timeout"],
        max_retries=cfg["qwen"]["max_retries"],
        base_url=cfg["qwen"].get("base_url"),
    )

    df = pd.read_csv(PRODUCTS, dtype=str).fillna("")
    total = len(df)
    brand_fixed = 0
    desc_fixed = 0

    for i, row in df.iterrows():
        # Brand
        if not row["brand"].strip():
            try:
                brand = client.call(prompt=row["title"], system=BRAND_PROMPT).strip()
                # 防御:过长的返回大概率不是品牌名
                if 0 < len(brand) <= 30:
                    df.at[i, "brand"] = brand
                    brand_fixed += 1
                    print(f"[brand] {row['product_id']}: {brand}")
            except Exception as exc:  # noqa: BLE001
                print(f"[brand] FAIL {row['product_id']}: {exc}")

        # Description fallback
        if not row["description"].strip():
            df.at[i, "description"] = row["title"]
            desc_fixed += 1

    df.to_csv(PRODUCTS, index=False, encoding="utf-8-sig")
    print(f"\nTotal rows: {total}")
    print(f"Brand fixed: {brand_fixed}")
    print(f"Desc fallback: {desc_fixed}")
    print(f"Saved: {PRODUCTS}")


if __name__ == "__main__":
    main()
