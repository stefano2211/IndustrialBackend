from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings

class Embedder:
    def __init__(self):
        self.model = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key
        )

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

    def embed_query(self, text):
        return self.model.embed_query(text)