from unittest.mock import MagicMock
from src.indexing.summary_index import build_summary_index

def test_build_summary_index_creates_chroma(tmp_path):
    summaries = [
        {"community_id": 0, "size": 10, "summary": "续航抱怨社区", "core_entities": ["电池"], "sample_review_ids": ["R1"]},
        {"community_id": 1, "size": 8, "summary": "拍照好评社区", "core_entities": ["相机"], "sample_review_ids": ["R2"]},
    ]
    fake_emb = MagicMock()
    fake_emb.embed_documents.return_value = [[0.1]*1024, [0.2]*1024]
    fake_emb.embed_query.return_value = [0.1]*1024
    db = build_summary_index(summaries, fake_emb, str(tmp_path / "sum"))
    assert db._collection.count() == 2
