from gliner2 import GLiNER2
from typing import List, Dict, Optional
from loguru import logger

class GLINERExtractor:
    def __init__(
        self, 
        model_name: str = "fastino/gliner2-base-v1",
        device: str = "cpu"
    ):
        """
        Inicializa gliner2-base-v1.
        Args:
            model_name: Modelo HF (default: fastino/gliner2-base-v1).
            device: "cpu" o "cuda" (recomendado CPU para escalabilidad).
        """
        try:
            self.model = GLiNER2.from_pretrained(
                model_name, 
                device=device,
                torch_dtype="float32"  # Para CPU stability
            )
            # Schema para NER con descripciones en NL (mejora precisión)
            self.ner_schema = {
                "person": "Nombres de personas o contactos",
                "organization": "Nombres de empresas, organizaciones o entidades legales",
                "date": "Fechas, plazos o referencias temporales como '15 de enero de 2025'",
                "money": "Importes monetarios, presupuestos o pagos como '185.000 €'",
                "location": "Lugares geográficos como ciudades o países",
                "financial_metric": "Métricas financieras clave como 'Net Revenues', 'Operating Income', 'EPS', 'Organic Growth'",
                "fiscal_period": "Periodos fiscales como 'Q3 2025', 'Fiscal Year 2024', 'Nine Months Ended'",
                "growth_rate": "Porcentajes de crecimiento o variación como '+5%', 'declined 2%', 'organic revenue growth of 9%'"
            }
            # Schema para clasificación de docs (single-label, categorías mutuamente exclusivas)
            self.classification_schema = {
                "document_type": ["contract", "invoice", "report", "nda", "email_thread", "financial_statement", "policy"]
            }
            logger.info(f"GLiNER2-large-v1 loaded: {model_name} on {device}")
        except Exception as e:
            logger.error(f"Failed to load GLiNER2-large-v1: {e}")
            raise

    def extract_entities(
        self, 
        text: str
    ) -> Dict[str, List[str]]:
        """
        Extrae entidades del texto usando extract_entities (de docs GLiNER2).
        Args:
            text: Texto a procesar.
        Returns:
            Dict como {'money': ['185.000 €'], 'project_code': ['AUR-2025-007']}.
        """
        if len(text.strip()) < 50:  # Optimización: Skip chunks muy cortos
            return {}
        
        try:
            # extract_entities según docs: schema como dict posicional después de text
            result = self.model.extract_entities(
                text, 
                self.ner_schema  
            )
            
            # Parse output: {'entities': {label: [values]}} → flatten
            entities = result.get('entities', {})
            
            # Flatten a list por label (evita duplicados)
            extracted = {}
            for label, values in entities.items():
                if isinstance(values, list):
                    extracted[label] = list(set([v.strip() for v in values if v.strip()]))  
                else:
                    extracted[label] = [str(values).strip()] if values else []
            
            logger.debug(f"Extracted {len(extracted)} entity types from text (len={len(text)})")
            return extracted  # {'money': ['185.000 €'], ...}
        except Exception as e:
            logger.warning(f"NER extraction failed for text: {e}")
            return {}

    def classify_document(
        self, 
        text: str
    ) -> str:
        """
        Clasifica el documento completo usando classify_text (de docs GLiNER2).
        Args:
            text: Texto completo del documento.
        Returns:
            String con la categoría (ej: 'contract').
        """
        if len(text.strip()) < 100:  # Optimización: Skip docs muy cortos
            return "unknown"
        
        try:
            # classify_text según docs: schema como dict posicional (single-label)
            result = self.model.classify_text(
                text, 
                self.classification_schema 
            )
            
            category = result.get('document_type', 'unknown')
            logger.debug(f"Classified document as: {category}")
            return category
        except Exception as e:
            logger.warning(f"Document classification failed: {e}")
            return "unknown"


_extractor_instance = None

def get_extractor(device: str = "cpu") -> 'GLINERExtractor':
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = GLINERExtractor(device=device)
    return _extractor_instance