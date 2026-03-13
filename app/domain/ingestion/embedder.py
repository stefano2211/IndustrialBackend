from langchain_ollama import OllamaEmbeddings
from app.core.config import settings

class Embedder:
    def __init__(self):
        self.model = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
        )

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

    def embed_query(self, text):
        return self.model.embed_query(text)