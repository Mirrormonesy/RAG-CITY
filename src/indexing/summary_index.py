from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

def build_summary_index(summaries: list[dict], embedding, persist_dir: str) -> Chroma:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    docs = [
        Document(
            page_content=s["summary"],
            metadata={
                "community_id": s["community_id"],
                "size": s["size"],
                "core_entities": ",".join(s["core_entities"]),
            },
        )
        for s in summaries
    ]
    db = Chroma.from_documents(docs, embedding=embedding, persist_directory=persist_dir)
    if hasattr(db, "persist"):
        db.persist()
    return db
