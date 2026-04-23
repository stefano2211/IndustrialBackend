# app/ingestion/document_loader.py
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from loguru import logger
from pathlib import Path

class DocumentLoader:
    @staticmethod
    def load(file_path: str) -> list[Document]:
        path = Path(file_path)
        logger.info(f"Loading document: {file_path}")

        try:
            if path.suffix.lower() == ".pdf":
                import pdfplumber
                
                documents = []
                with pdfplumber.open(file_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        # 1. Extraer texto normal
                        text = page.extract_text() or ""
                        
                        # 2. Extraer tablas y convertirlas a Markdown
                        tables = page.extract_tables()
                        tables_md = []
                        if tables:
                            for table in tables:
                                # Limpiar celdas (None -> "") y eliminar saltos de línea internos
                                clean_table = [
                                    [str(cell or "").strip().replace("\n", " ") for cell in row]
                                    for row in table
                                ]
                                
                                # Generar tabla Markdown si tiene contenido
                                if clean_table:
                                    # Asumimos primera fila como header
                                    headers = clean_table[0]
                                    rows = clean_table[1:]
                                    
                                    # Construir tabla MD
                                    md_table = f"\n| {' | '.join(headers)} |"
                                    md_table += f"\n| {' | '.join(['---'] * len(headers))} |"
                                    for row in rows:
                                        md_table += f"\n| {' | '.join(row)} |"
                                    
                                    tables_md.append(md_table)
                        
                        # 3. Combinar texto y tablas (dando prioridad a la estructura)
                        # Añadimos las tablas al final del texto de la página para contexto explícito
                        if tables_md:
                            text += "\n\n### Extracted Tables (Structured):\n" + "\n\n".join(tables_md)

                        metadata = {
                            "source": path.name,
                            "format": "pdf",
                            "page": i + 1,
                            "total_pages": len(pdf.pages)
                        }
                        
                        documents.append(Document(
                            page_content=text.strip(),
                            metadata=metadata
                        ))
                return documents

            elif path.suffix.lower() in [".docx", ".doc"]:
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(file_path)
                return loader.load()

            elif path.suffix.lower() == ".json":
                import json
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                text = json.dumps(data, ensure_ascii=False, indent=2)
                return [Document(
                    page_content=text,
                    metadata={"source": path.name, "format": "json"}
                )]

            else:
                logger.error(f"Unsupported file type: {path.suffix}")
                raise ValueError(f"Unsupported file type: {path.suffix}")

        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            raise