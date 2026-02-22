import langextract
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from app.core.llm import LLMFactory
from app.core.config import settings
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
        Inicializa extractor usando LangChain con soporte para salida estructurada.
        Utiliza el LLM configurado en LLMFactory para el rol 'extractor' (Gemini recomendado).
        """
        try:
            # Obtenemos el LLM de la factoría para el rol específico de extractor
            self.llm = LLMFactory.get_llm(role="extractor")
            
            # Preparamos los modelos con salida estructurada
            self.extraction_model = self.llm.with_structured_output(EntityExtraction)
            self.classification_model = self.llm.with_structured_output(DocumentClassification)
            
            logger.info("LangExtractExtractor (via LangChain) initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LangExtractExtractor: {e}")
            raise

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extrae entidades del texto usando salida estructurada de LangChain.
        """
        if len(text.strip()) < 50:
            return {}
        
        try:
            # Invocamos el modelo estructurado con instrucciones explícitas
            prompt = f"""Extract compliance and industrial safety entities from the following text according to the specific schema.
            
            Text:
            {text}"""
            
            result = self.extraction_model.invoke(prompt)
            
            # El resultado es una instancia de EntityExtraction
            if isinstance(result, EntityExtraction):
                return result.model_dump()
            
            return {}
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return {}

    def classify_document(self, text: str) -> str:
        """
        Clasifica el documento usando salida estructurada de LangChain.
        """
        if len(text.strip()) < 100:
            return "unknown"
        
        try:
            # Invocamos el modelo estructurado con instrucciones explícitas
            prompt = f"""Classify this industrial document based on its content into the most appropriate category.
            
            Text to classify:
            {text[:2000]}""" # Usamos los primeros 2k caracteres para clasificar
            
            result = self.classification_model.invoke(prompt)
            
            if isinstance(result, DocumentClassification):
                return result.document_type
                
            return "unknown"
        except Exception as e:
            logger.warning(f"Document classification failed: {e}")
            return "unknown"

_extractor_instance = None

def get_extractor(device: str = "cpu") -> 'LangExtractExtractor':
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = LangExtractExtractor()
    return _extractor_instance
