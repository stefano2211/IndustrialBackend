import langextract
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from app.core.llm import LLMFactory
from loguru import logger
import json

# Definimos los esquemas de extracción usando Pydantic para LangExtract
class EntityExtraction(BaseModel):
    regulation: List[str] = Field(default=[], description="Specific regulations, laws, or norms (e.g., 'OSHA 1910', 'ISO 14001', 'NOM-002-STPS')")
    standard: List[str] = Field(default=[], description="Safety, quality, or technical standards (e.g., 'ANSI Z87.1', 'NFPA 70E')")
    date: List[str] = Field(default=[], description="Compliance deadlines, audit dates, incident dates, or effective dates")
    penalty: List[str] = Field(default=[], description="Monetary fines, sanctions, or penalties for non-compliance")
    location: List[str] = Field(default=[], description="Specific facility names, zones, or areas (e.g., 'Zone A', 'Warehouse 3')")
    responsible_party: List[str] = Field(default=[], description="Individuals, roles, or departments responsible for compliance")
    equipment: List[str] = Field(default=[], description="Machinery, tools, or equipment involved in compliance or incidents")
    hazard: List[str] = Field(default=[], description="Specific safety or environmental hazards identified")
    organization: List[str] = Field(default=[], description="Company names, organizations, or legal entities")
    money: List[str] = Field(default=[], description="Monetary amounts")

class DocumentClassification(BaseModel):
    document_type: str = Field(description="Categoría del documento (audit_report, permit, incident_report, procedure, certification, notice, contract, invoice, policy, unknown)")

class LangExtractExtractor:
    def __init__(self):
        """
        Inicializa extractor usando Google LangExtract.
        Utiliza el LLM configurado en LLMFactory para el rol 'extractor' (Gemini recomendado).
        """
        try:
            # Obtenemos el LLM de la factoría para el rol específico de extractor
            self.llm = LLMFactory.get_llm(role="extractor")
            # Extraemos la configuración para LangExtract (que usa su propia factoría interna)
            self.model_id = getattr(self.llm, "model_name", settings.extractor_llm_model)
            self.api_key = getattr(self.llm, "google_api_key", settings.gemini_api_key)
            
            logger.info(f"LangExtractExtractor initialized for model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize LangExtractExtractor: {e}")
            raise

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extrae entidades del texto usando LangExtract.
        """
        if len(text.strip()) < 50:
            return {}
        
        try:
            prompt = "Extract compliance and industrial safety entities from the following text."
            
            # langextract.extract es la función principal en la v1.1.1
            extraction_result = langextract.extract(
                text_or_documents=text, 
                schema_class=EntityExtraction,
                prompt_description=prompt,
                model_id=self.model_id,
                api_key=self.api_key
            )
            
            # El resultado suele ser una instancia de la clase Pydantic definida
            if isinstance(extraction_result, EntityExtraction):
                return extraction_result.model_dump()
            
            # Fallback en caso de que devuelva un dict
            return extraction_result if isinstance(extraction_result, dict) else {}
            
        except Exception as e:
            logger.warning(f"LangExtract entity extraction failed: {e}")
            return {}

    def classify_document(self, text: str) -> str:
        """
        Clasifica el documento usando LangExtract.
        """
        if len(text.strip()) < 100:
            return "unknown"
        
        try:
            prompt = "Classify this industrial document based on its content into the most appropriate category."
            
            classification_result = langextract.extract(
                text_or_documents=text,
                schema_class=DocumentClassification,
                prompt_description=prompt,
                model_id=self.model_id,
                api_key=self.api_key
            )
            
            if isinstance(classification_result, DocumentClassification):
                return classification_result.document_type
            
            if isinstance(classification_result, dict):
                return classification_result.get("document_type", "unknown")
                
            return "unknown"
        except Exception as e:
            logger.warning(f"LangExtract document classification failed: {e}")
            return "unknown"

_extractor_instance = None

def get_extractor(device: str = "cpu") -> 'LangExtractExtractor':
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = LangExtractExtractor()
    return _extractor_instance
