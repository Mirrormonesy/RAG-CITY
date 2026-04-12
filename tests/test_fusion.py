from langchain_core.documents import Document
from src.retrieval.fusion import RRFFuser, doc_key

def _d(content, meta=None):
    return Document(page_content=content, metadata=meta or {})

def test_doc_key_uses_metadata_id():
    assert doc_key(_d("x", {"review_id": "R1"})) == "review:R1"
    assert doc_key(_d("x", {"community_id": 7})) == "community:7"
    assert doc_key(_d("x", {"product_id": "P1"})) == "product:P1"
    assert doc_key(_d("content-only")) == "content:content-only"

def test_rrf_ranks_by_combined_score():
    v_list = [_d("A", {"review_id": "R1"}), _d("B", {"review_id": "R2"})]
    g_list = [_d("B", {"review_id": "R2"}), _d("C", {"review_id": "R3"})]
    fuser = RRFFuser(k_const=60)
    out = fuser.fuse(v_list, g_list)
    assert out[0].metadata["review_id"] == "R2"

def test_rrf_preserves_documents():
    v_list = [_d("only-in-v", {"review_id": "R1"})]
    fuser = RRFFuser()
    out = fuser.fuse(v_list)
    assert len(out) == 1
    assert out[0].page_content == "only-in-v"
