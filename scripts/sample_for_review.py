"""从 graph_partial.jsonl 随机抽 N 条,导出 CSV 供人工标注"""
import argparse
import json
import random
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", default="indices/graph_partial.jsonl")
    parser.add_argument("--reviews", default="data/processed/reviews.parquet")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="report/figures/extraction_sample.csv")
    args = parser.parse_args()

    records = []
    with open(args.partial, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    random.seed(args.seed)
    sampled = random.sample(records, min(args.n, len(records)))

    reviews_df = pd.read_parquet(args.reviews).set_index("review_id")

    rows = []
    for rec in sampled:
        rid = rec["review_id"]
        content = reviews_df.loc[rid, "content"] if rid in reviews_df.index else ""
        rows.append({
            "review_id": rid,
            "content": content,
            "extracted_aspects": "; ".join(rec["facts"].get("aspects", [])),
            "extracted_features": "; ".join(rec["facts"].get("features", [])),
            "extracted_sentiment": rec["facts"].get("sentiment", ""),
            "human_aspects_correct": "",
            "human_sentiment_correct": "",
            "human_notes": "",
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"已导出 {len(rows)} 条至 {out},请人工标注 human_* 列")

if __name__ == "__main__":
    main()
