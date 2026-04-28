"""
Reactive Knowledge Base API endpoints.

System-scoped endpoints corresponding to SOPs, operations manuals, and emergency procedures.
Documents ingested here are grouped in the ReactiveQdrantManager and automatically
searchable by the ask_reactive_knowledge tool.
"""

import uuid
from typing import List
from tempfile import NamedTemporaryFile
import os

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlmodel.ext.asyncio.session import AsyncSession

from app.persistence.db import get_session
from app.domain.reactiva.schemas.reactive_knowledge import (
    ReactiveKnowledgeBaseRead,
    ReactiveKnowledgeBaseCreate,
    ReactiveKnowledgeBaseDetailRead,
    ReactiveKnowledgeDocument,
    ReactiveKnowledgeBase
)
from app.persistence.reactiva.repositories.reactive_knowledge_repository import ReactiveKnowledgeRepository
from app.domain.reactiva.ingestion.reactive_pipeline import ReactiveDocumentProcessor

router = APIRouter()


@router.post("/", response_model=ReactiveKnowledgeBaseRead)
async def create_reactive_kb(
    kb_in: ReactiveKnowledgeBaseCreate,
    tenant_id: str = "default",
    session: AsyncSession = Depends(get_session)
):
    repo = ReactiveKnowledgeRepository(session)
    kb = ReactiveKnowledgeBase(
        name=kb_in.name,
        description=kb_in.description,
        tenant_id=tenant_id
    )
    return await repo.create_kb(kb)


@router.get("/", response_model=List[ReactiveKnowledgeBaseRead])
async def list_reactive_kbs(
    tenant_id: str = "default",
    session: AsyncSession = Depends(get_session)
):
    repo = ReactiveKnowledgeRepository(session)
    return await repo.list_kbs(tenant_id)


@router.get("/{kb_id}", response_model=ReactiveKnowledgeBaseDetailRead)
async def get_reactive_kb(
    kb_id: uuid.UUID,
    session: AsyncSession = Depends(get_session)
):
    repo = ReactiveKnowledgeRepository(session)
    kb = await repo.get_kb_by_id(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Reactive KB not found")
    
    docs = await repo.list_documents(kb_id)
    kb_dict = kb.model_dump()
    kb_dict["documents"] = docs
    return kb_dict


@router.post("/{kb_id}/documents")
async def upload_reactive_document(
    kb_id: uuid.UUID,
    tenant_id: str = "default",
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session)
):
    """
    Upload a document securely via tempfile, process using the reactive pipeline,
    and persist in the reactive Qdrant collection.
    """
    repo = ReactiveKnowledgeRepository(session)
    kb = await repo.get_kb_by_id(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Reactive KB not found")

    # 1. Save file to temporary location
    _, ext = os.path.splitext(file.filename)
    tmp_file = NamedTemporaryFile(delete=False, suffix=ext)
    try:
        content = await file.read()
        tmp_file.write(content)
        tmp_file.flush()
        tmp_path = tmp_file.name
    finally:
        tmp_file.close()

    try:
        # 2. Process via Reactive Pipeline (Qdrant `reactive_documents`)
        processor = ReactiveDocumentProcessor()
        process_result = await processor.process_file(
            file_path=tmp_path,
            tenant_id=tenant_id,
            knowledge_base_id=str(kb_id),
            session=session
        )

        doc_id = process_result["doc_id"]

        # 3. Create document record
        new_doc = ReactiveKnowledgeDocument(
            knowledge_base_id=kb_id,
            filename=file.filename,
            file_id=doc_id,
        )
        return await repo.add_document(new_doc)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 4. Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.delete("/{kb_id}")
async def delete_reactive_kb(
    kb_id: uuid.UUID,
    session: AsyncSession = Depends(get_session)
):
    repo = ReactiveKnowledgeRepository(session)
    success = await repo.delete_kb(kb_id)
    if not success:
        raise HTTPException(status_code=404, detail="Reactive KB not found")
    # Note: Actual vector deletion in Qdrant should be handled 
    # either by a celery task or directly before returning here.
    return {"status": "ok"}
