import re
import pandas as pd

BRAND_ALIASES = {
    "apple": "苹果",
    "huawei": "华为",
    "xiaomi": "小米",
    "samsung": "三星",
    "oppo": "OPPO",
    "vivo": "vivo",
    "honor": "荣耀",
}

EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001F9FF\U0001FA00-\U0001FAFF\U00002600-\U000027BF\U0001F600-\U0001F64F]",
    flags=re.UNICODE,
)


def normalize_brand(brand: str) -> str:
    if brand is None:
        return ""
    key = str(brand).strip().lower()
    return BRAND_ALIASES.get(key, str(brand).strip())


def _is_meaningful(text: str) -> bool:
    if not isinstance(text, str):
        return False
    stripped = EMOJI_PATTERN.sub("", text)
    stripped = re.sub(r"\d+", "", stripped)
    stripped = re.sub(r"\s+", "", stripped)
    stripped = re.sub(r"[^\w\u4e00-\u9fff]", "", stripped)
    return len(stripped) >= 5


def clean_reviews(df: pd.DataFrame, min_len: int = 10, max_len: int = 500) -> pd.DataFrame:
    out = df.copy()
    out = out[out["content"].astype(str).str.len().between(min_len, max_len)]
    out = out[out["content"].apply(_is_meaningful)]
    out = out.drop_duplicates(subset=["user_id", "product_id"], keep="first")
    return out.reset_index(drop=True)


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["brand"] = out["brand"].apply(normalize_brand)
    out["description"] = out["description"].fillna("")
    out = out.drop_duplicates(subset=["product_id"], keep="first")
    return out.reset_index(drop=True)
