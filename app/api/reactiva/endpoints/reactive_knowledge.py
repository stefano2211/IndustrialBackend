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

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api.deps import get_current_user
from app.persistence.db import get_session
from app.domain.shared.schemas.user import User
from app.domain.reactiva.schemas.reactive_knowledge import (
    ReactiveKnowledgeBaseRead,
    ReactiveKnowledgeBaseCreate,
    ReactiveKnowledgeBaseUpdate,
    ReactiveKnowledgeBaseDetailRead,
    ReactiveKnowledgeDocumentRead,
    ReactiveKnowledgeDocument,
    ReactiveKnowledgeBase
)
from app.persistence.reactiva.repositories.reactive_knowledge_repository import ReactiveKnowledgeRepository
from app.domain.reactiva.ingestion.reactive_pipeline import ReactiveDocumentProcessor

router = APIRouter()


@router.post("/", response_model=ReactiveKnowledgeBaseRead)
async def create_reactive_kb(
    kb_in: ReactiveKnowledgeBaseCreate,
    session: AsyncSession = Depends(get_session),
    current_user: Annotated[User, Depends(get_current_user)] = None,
):
    repo = ReactiveKnowledgeRepository(session)
    tenant_id = current_user.tenant_id if current_user else "default"
    kb = ReactiveKnowledgeBase(
        name=kb_in.name,
        description=kb_in.description,
        tenant_id=tenant_id
    )
    return await repo.create_kb(kb)


@router.get("/", response_model=List[ReactiveKnowledgeBaseRead])
async def list_reactive_kbs(
    session: AsyncSession = Depends(get_session),
    current_user: Annotated[User, Depends(get_current_user)] = None,
):
    repo = ReactiveKnowledgeRepository(session)
    tenant_filter = None if (current_user and current_user.is_superuser) else (current_user.tenant_id if current_user else "default")
    if tenant_filter is not None:
        return await repo.list_kbs(tenant_filter)
    return await repo.list_kbs()


@router.get("/{kb_id}", response_model=ReactiveKnowledgeBaseDetailRead)
async def get_reactive_kb(
    kb_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    current_user: Annotated[User, Depends(get_current_user)] = None,
):
    repo = ReactiveKnowledgeRepository(session)
    kb = await repo.get_kb_by_id(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Reactive KB not found")
    if current_user and not current_user.is_superuser and kb.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this knowledge base")

    docs = await repo.list_documents(kb_id)
    kb_dict = kb.model_dump()
    kb_dict["documents"] = docs
    return kb_dict


@router.post("/{kb_id}/documents")
async def upload_reactive_document(
    kb_id: uuid.UUID,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    current_user: Annotated[User, Depends(get_current_user)] = None,
):
    """
    Upload a document securely via tempfile, process using the reactive pipeline,
    and persist in the reactive Qdrant collection.
    """
    repo = ReactiveKnowledgeRepository(session)
    kb = await repo.get_kb_by_id(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Reactive KB not found")
    if current_user and not current_user.is_superuser and kb.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this knowledge base")

    tenant_id = current_user.tenant_id if current_user else kb.tenant_id

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


@router.put("/{kb_id}", response_model=ReactiveKnowledgeBaseRead)
async def update_reactive_kb(
    kb_id: uuid.UUID,
    kb_in: ReactiveKnowledgeBaseUpdate,
    session: AsyncSession = Depends(get_session),
    current_user: Annotated[User, Depends(get_current_user)] = None,
):
    repo = ReactiveKnowledgeRepository(session)
    kb = await repo.get_kb_by_id(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Reactive KB not found")
    if current_user and not current_user.is_superuser and kb.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to update this knowledge base")

    updated = await repo.update_kb(kb_id, kb_in.model_dump(exclude_unset=True))
    if not updated:
        raise HTTPException(status_code=500, detail="Update failed")
    return updated


@router.get("/{kb_id}/documents", response_model=List[ReactiveKnowledgeDocumentRead])
async def list_reactive_kb_documents(
    kb_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    current_user: Annotated[User, Depends(get_current_user)] = None,
):
    repo = ReactiveKnowledgeRepository(session)
    kb = await repo.get_kb_by_id(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Reactive KB not found")
    if current_user and not current_user.is_superuser and kb.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this knowledge base")

    return await repo.list_documents(kb_id)


@router.delete("/{kb_id}/documents/{doc_id}")
async def delete_reactive_document(
    kb_id: uuid.UUID,
    doc_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    current_user: Annotated[User, Depends(get_current_user)] = None,
):
    repo = ReactiveKnowledgeRepository(session)
    kb = await repo.get_kb_by_id(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Reactive KB not found")
    if current_user and not current_user.is_superuser and kb.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete documents in this knowledge base")

    success = await repo.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "ok"}


@router.delete("/{kb_id}")
async def delete_reactive_kb(
    kb_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    current_user: Annotated[User, Depends(get_current_user)] = None,
):
    repo = ReactiveKnowledgeRepository(session)
    kb = await repo.get_kb_by_id(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Reactive KB not found")
    if current_user and not current_user.is_superuser and kb.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this knowledge base")

    success = await repo.delete_kb(kb_id)
    # Note: Actual vector deletion in Qdrant should be handled
    # either by a celery task or directly before returning here.
    return {"status": "ok"}
