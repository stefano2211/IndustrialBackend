from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings

class Embedder:
    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},  # cambia a "cuda" si tienes GPU
            encode_kwargs={"normalize_embeddings": True}
        )

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

    def embed_query(self, text):
        return self.model.embed_query(text)