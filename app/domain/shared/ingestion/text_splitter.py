from langchain_core.documents import Document
import re

class HierarchicalSplitter:
    def __init__(self):
        pass

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks = []
        for doc in documents:
            # Intentar detectar secciones por títulos
            lines = doc.page_content.split("\n")
            current_section = ""
            current_content = []

            for line in lines:
                clean_line = line.strip()
                # Numbered section ("1. Title", "2.3 ") OR all-uppercase header (must have ≥1 cased char)
                is_section_header = (
                    bool(re.match(r"^\d+\.?\d*\.?\s+[A-Za-záéíóúñÁÉÍÓÚÑ]", clean_line))
                    or (clean_line.isupper() and len(clean_line.replace(" ", "")) >= 3)
                ) and len(clean_line) < 100
                if is_section_header:
                    if current_content:
                        chunks.append(self._create_chunk("\n".join(current_content), current_section, doc))
                        current_content = []
                    current_section = clean_line
                else:
                    current_content.append(line)

            if current_content:
                chunks.append(self._create_chunk("\n".join(current_content), current_section, doc))

        # Size enforcement is handled entirely by pipeline.py (RecursiveCharacterTextSplitter
        # with DB-configurable chunk_size/overlap). This stage only adds section metadata.
        return chunks

    def _create_chunk(self, content, section, original_doc):
        return Document(
            page_content=content.strip(),
            metadata={
                **original_doc.metadata,
                "section": section or "No section",
                "source": original_doc.metadata.get("source", "unknown")
            }
        )
