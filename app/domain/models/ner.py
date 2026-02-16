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
            # Schema para NER con descripciones en NL (mejora precisión) - ADAPTADO A COMPLIANCE INDUSTRIAL
            self.ner_schema = {
                "regulation": "Specific regulations, laws, or norms (e.g., 'OSHA 1910', 'ISO 14001', 'NOM-002-STPS', 'Clean Air Act')",
                "standard": "Safety, quality, or technical standards (e.g., 'ANSI Z87.1', 'NFPA 70E', 'ASTM International')",
                "date": "Compliance deadlines, audit dates, incident dates, or effective dates",
                "penalty": "Monetary fines, sanctions, or penalties for non-compliance",
                "location": "Specific facility names, zones, or areas (e.g., 'Zone A', 'Warehouse 3', 'Assembly Line 1')",
                "responsible_party": "Individuals, roles, or departments responsible for compliance (e.g., 'Safety Officer', 'Plant Manager', 'EHS Dept')",
                "equipment": "Machinery, tools, or equipment involved in compliance checks or incidents (e.g., 'Forklift', 'Boiler B')",
                "hazard": "Specific safety or environmental hazards identified (e.g., 'Chemical Spill', 'High Voltage', 'Slippery Floor')",
                "organization": "Nombres de empresas, organizaciones o entidades legales", # Mantenemos general también
                "money": "Importes monetarios" # Mantenemos general
            }
            # Schema para clasificación de docs (single-label, categorías mutuamente exclusivas) - ADAPTADO A COMPLIANCE INDUSTRIAL
            self.classification_schema = {
                "document_type": [
                    "audit_report", 
                    "permit", 
                    "incident_report", 
                    "procedure", 
                    "certification", 
                    "non_compliance_notice",
                    "contract", 
                    "invoice", 
                    "report", 
                    "policy"
                ]
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