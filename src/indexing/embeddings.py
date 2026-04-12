from langchain_community.embeddings import HuggingFaceEmbeddings

def load_bge_embedding(model_name: str, device: str, batch_size: int) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": batch_size, "normalize_embeddings": True},
    )
