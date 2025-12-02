from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re

class HierarchicalSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks = []
        for doc in documents:
            # Intentar detectar secciones por títulos
            lines = doc.page_content.split("\n")
            current_section = ""
            current_content = []

            for line in lines:
                clean_line = line.strip()
                if re.match(r"^\d+\.?\s|[A-ZÁÉÍÓÚÑ\s]{3,}$", clean_line) and len(clean_line) < 100:
                    if current_content:
                        chunks.append(self._create_chunk("\n".join(current_content), current_section, doc))
                        current_content = []
                    current_section = clean_line
                else:
                    current_content.append(line)

            if current_content:
                chunks.append(self._create_chunk("\n".join(current_content), current_section, doc))

        # Asegurar que todos los chunks (vengan de secciones o no) respeten el tamaño máximo
        return self.splitter.split_documents(chunks)

    def _create_chunk(self, content, section, original_doc):
        return Document(
            page_content=content.strip(),
            metadata={
                **original_doc.metadata,
                "section": section or "No section",
                "source": original_doc.metadata.get("source", "unknown")
            }
        )