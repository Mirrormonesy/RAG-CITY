"""Translate English products.csv + reviews.csv to Chinese using Qwen.

Per-row translation with JSONL checkpoint for resume.
Run again = pick up where it left off.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.llm_client import QwenClient  # noqa: E402

SYSTEM_PROMPT = (
    "你是电商内容本地化专家。把用户给你的英文电商文本翻译成自然通顺的简体中文。"
    "要求:1) 保留品牌名和型号原文 2) 保留数字和单位 3) 符合中文电商评论/商品描述的语气"
    " 4) 只输出翻译结果,不要任何解释或引号。"
)


def load_checkpoint(path: Path) -> dict[str, str]:
    done: dict[str, str] = {}
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                done[obj["key"]] = obj["translation"]
            except Exception:
                continue
    return done


def append_checkpoint(path: Path, key: str, translation: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "translation": translation}, ensure_ascii=False) + "\n")


def translate_one(client: QwenClient, text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return client.call(prompt=text, system=SYSTEM_PROMPT).strip()


def translate_column(
    df: pd.DataFrame,
    id_col: str,
    text_col: str,
    client: QwenClient,
    ckpt_path: Path,
    tag: str,
) -> pd.DataFrame:
    done = load_checkpoint(ckpt_path)
    print(f"[{tag}] checkpoint hits: {len(done)}")

    out = df.copy()
    total = len(out)
    for i, row in out.iterrows():
        key = f"{row[id_col]}:{text_col}"
        if key in done:
            out.at[i, text_col] = done[key]
            continue
        src = row[text_col]
        try:
            zh = translate_one(client, str(src))
        except Exception as exc:  # noqa: BLE001
            print(f"[{tag}] FAILED {key}: {exc}")
            continue
        out.at[i, text_col] = zh
        append_checkpoint(ckpt_path, key, zh)
        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"[{tag}] {i + 1}/{total}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    ap.add_argument("--products", default=str(PROJECT_ROOT / "data" / "raw" / "products.csv"))
    ap.add_argument("--reviews", default=str(PROJECT_ROOT / "data" / "raw" / "reviews.csv"))
    ap.add_argument("--ckpt-dir", default=str(PROJECT_ROOT / "data" / "raw" / "_translate_ckpt"))
    args = ap.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    api_key = os.getenv(cfg["qwen"]["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Env var {cfg['qwen']['api_key_env']} not set. Fill in .env first.")

    client = QwenClient(
        api_key=api_key,
        model=cfg["qwen"]["extract_model"],
        temperature=0.2,
        max_tokens=512,
        timeout=cfg["qwen"]["timeout"],
        max_retries=cfg["qwen"]["max_retries"],
        base_url=cfg["qwen"].get("base_url"),
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Products ---
    products = pd.read_csv(args.products, dtype=str).fillna("")
    products = translate_column(
        products, "product_id", "title", client, ckpt_dir / "products_title.jsonl", "title"
    )
    products = translate_column(
        products, "product_id", "description", client, ckpt_dir / "products_desc.jsonl", "desc"
    )
    products.to_csv(args.products, index=False, encoding="utf-8-sig")
    print(f"Saved translated products -> {args.products}")

    # --- Reviews ---
    reviews = pd.read_csv(args.reviews, dtype=str).fillna("")
    reviews = translate_column(
        reviews, "review_id", "content", client, ckpt_dir / "reviews_content.jsonl", "review"
    )
    reviews.to_csv(args.reviews, index=False, encoding="utf-8-sig")
    print(f"Saved translated reviews -> {args.reviews}")


if __name__ == "__main__":
    main()
