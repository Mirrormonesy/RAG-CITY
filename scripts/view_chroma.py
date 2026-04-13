"""Inspect a Chroma persistent DB. Usage: python scripts/view_chroma.py [path] [n]"""
import sys
from pathlib import Path
import chromadb

DEFAULT = "indices/chroma_docs"
path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

client = chromadb.PersistentClient(path=path)
print(f"DB: {path}")
print(f"Collections: {[c.name for c in client.list_collections()]}")

for coll in client.list_collections():
    c = client.get_collection(coll.name)
    total = c.count()
    print(f"\n=== Collection: {coll.name}  (total: {total}) ===")
    data = c.get(limit=n, include=["documents", "metadatas"])
    for i, (doc_id, doc, meta) in enumerate(
        zip(data["ids"], data["documents"], data["metadatas"])
    ):
        print(f"\n[{i+1}] id={doc_id}")
        print(f"    meta: {meta}")
        print(f"    text: {doc[:200]}")
