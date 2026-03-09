"""
NER Extractor — Entity extraction and document classification via LLM.

Uses batch processing to extract entities from multiple chunks in a single
LLM call, dramatically reducing total API calls and processing time.
Retry with exponential backoff handles rate limits automatically.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from app.core.llm import LLMFactory
from app.core.config import settings
from loguru import logger


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

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


class BatchEntityExtraction(BaseModel):
    """Wrapper for extracting entities from multiple text chunks at once."""
    results: List[EntityExtraction] = Field(
        description="One EntityExtraction per input chunk, in the SAME order as the input chunks."
    )


class DocumentClassification(BaseModel):
    document_type: str = Field(description="Categoría del documento (audit_report, permit, incident_report, procedure, certification, notice, contract, invoice, policy, unknown)")


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class LangExtractExtractor:
    def __init__(self):
        self.llm = None
        self.single_extraction_model = None
        self.batch_extraction_model = None
        self.classification_model = None

    async def _ensure_initialized(self, session: Optional[AsyncSession] = None):
        if self.llm is None:
            self.llm = await LLMFactory.get_llm(role="extractor", session=session)
            self.single_extraction_model = self.llm.with_structured_output(EntityExtraction)
            self.batch_extraction_model = self.llm.with_structured_output(BatchEntityExtraction)
            self.classification_model = self.llm.with_structured_output(DocumentClassification)
            logger.info("LangExtractExtractor initialized successfully")

    # ------------------------------------------------------------------
    # Single-chunk extraction (kept for compatibility)
    # ------------------------------------------------------------------

    async def extract_entities(
        self, text: str, session: Optional[AsyncSession] = None
    ) -> Dict[str, List[str]]:
        if len(text.strip()) < 50:
            return {}

        await self._ensure_initialized(session)
        try:
            prompt = (
                "Extract compliance and industrial safety entities from the "
                "following text according to the specific schema.\n\n"
                f"Text:\n{text}"
            )
            result = await self._invoke_with_retry(self.single_extraction_model, prompt)
            if isinstance(result, EntityExtraction):
                return result.model_dump()
            return {}
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return {}

    # ------------------------------------------------------------------
    # Batch extraction — multiple chunks in ONE LLM call
    # ------------------------------------------------------------------

    async def extract_entities_batch(
        self,
        texts: List[str],
        session: Optional[AsyncSession] = None,
    ) -> List[Dict[str, List[str]]]:
        """
        Extract entities from multiple text chunks in a single LLM call.

        Returns a list of entity dicts, one per input text, in the same order.
        Texts shorter than 50 chars get an empty dict without hitting the LLM.
        """
        await self._ensure_initialized(session)

        # Separate valid texts from too-short ones, keeping track of indices
        index_map: List[int] = []       # positions of valid texts in original list
        valid_texts: List[str] = []
        results: List[Dict[str, List[str]]] = [{} for _ in texts]

        for i, text in enumerate(texts):
            if len(text.strip()) >= 50:
                index_map.append(i)
                valid_texts.append(text)

        if not valid_texts:
            return results

        try:
            # Build a numbered prompt so the LLM can match results to chunks
            numbered_chunks = "\n\n".join(
                f"--- CHUNK {i + 1} ---\n{text}"
                for i, text in enumerate(valid_texts)
            )
            prompt = (
                "Extract compliance and industrial safety entities from EACH of the "
                f"following {len(valid_texts)} text chunks. Return one EntityExtraction "
                "per chunk in the SAME ORDER.\n\n"
                f"{numbered_chunks}"
            )

            batch_result = await self._invoke_with_retry(
                self.batch_extraction_model, prompt
            )

            if isinstance(batch_result, BatchEntityExtraction):
                for idx, extraction in zip(index_map, batch_result.results):
                    results[idx] = extraction.model_dump()
            else:
                logger.warning("Batch extraction returned unexpected type, falling back to empty")

        except Exception as e:
            logger.warning(f"Batch entity extraction failed: {e}")
            # On batch failure, fall back to single extraction per chunk
            logger.info("Falling back to single-chunk extraction")
            for idx, text in zip(index_map, valid_texts):
                results[idx] = await self.extract_entities(text, session=session)

        return results

    # ------------------------------------------------------------------
    # Document classification
    # ------------------------------------------------------------------

    async def classify_document(
        self, text: str, session: Optional[AsyncSession] = None
    ) -> str:
        if len(text.strip()) < 100:
            return "unknown"

        await self._ensure_initialized(session)
        try:
            prompt = (
                "Classify this industrial document based on its content into "
                "the most appropriate category.\n\n"
                f"Text to classify:\n{text[:2000]}"
            )
            result = await self._invoke_with_retry(self.classification_model, prompt)
            if isinstance(result, DocumentClassification):
                return result.document_type
            return "unknown"
        except Exception as e:
            logger.warning(f"Document classification failed: {e}")
            return "unknown"

    # ------------------------------------------------------------------
    # Retry wrapper — exponential backoff on rate-limit errors
    # ------------------------------------------------------------------

    @staticmethod
    async def _invoke_with_retry(model, prompt: str):
        """
        Invoke a structured-output model with automatic retry on rate-limit
        errors (HTTP 429, ResourceExhausted, etc.).

        Uses tenacity for exponential backoff: 2s → 4s → 8s → 16s → 32s
        """
        max_attempts = settings.ner_retry_max_attempts

        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            retry=retry_if_exception_type((Exception,)),
            before_sleep=lambda rs: logger.info(
                f"NER rate-limited, retrying in {rs.next_action.sleep:.1f}s "
                f"(attempt {rs.attempt_number}/{max_attempts})"
            ),
            reraise=True,
        )
        async def _call():
            return await model.ainvoke(prompt)

        return await _call()


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_extractor_instance = None


def get_extractor(device: str = "cpu") -> "LangExtractExtractor":
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = LangExtractExtractor()
    return _extractor_instance
