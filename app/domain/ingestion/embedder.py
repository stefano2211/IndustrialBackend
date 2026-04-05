from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

class Embedder:
    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'trust_remote_code': True}
        )

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

    def embed_query(self, text):
        return self.model.embed_query(text)