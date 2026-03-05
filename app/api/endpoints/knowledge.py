from typing import List
from fastapi import APIRouter, Depends, status
import uuid
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.persistence.db import get_session
from app.domain.schemas.user import User
from app.domain.schemas.knowledge import (
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeBaseRead,
    KnowledgeBaseDetailRead,
    KnowledgeDocumentRead
)
from app.domain.services.knowledge_service import KnowledgeService

router = APIRouter(dependencies=[Depends(deps.get_current_user)])

async def get_knowledge_service(session: AsyncSession = Depends(get_session)) -> KnowledgeService:
    return KnowledgeService(session=session)

@router.post("/", response_model=KnowledgeBaseRead, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    data: KnowledgeBaseCreate,
    current_user: User = Depends(deps.get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service)
):
    """Create a new knowledge base collection."""
    return await service.create_knowledge_base(user_id=current_user.id, data=data)

@router.get("/", response_model=List[KnowledgeBaseRead])
async def list_knowledge_bases(
    current_user: User = Depends(deps.get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service)
):
    """List all knowledge base collections for the current user."""
    return await service.list_knowledge_bases(user_id=current_user.id)

@router.get("/{kb_id}", response_model=KnowledgeBaseDetailRead)
async def get_knowledge_base(
    kb_id: uuid.UUID,
    current_user: User = Depends(deps.get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service)
):
    """Get a specific knowledge base and its documents."""
    kb = await service.get_knowledge_base(kb_id=kb_id, user_id=current_user.id)
    documents = await service.get_kb_documents(kb_id=kb_id, user_id=current_user.id)
    
    kb_dict = kb.model_dump()
    kb_dict["documents"] = documents
    return kb_dict

@router.patch("/{kb_id}", response_model=KnowledgeBaseRead)
async def update_knowledge_base(
    kb_id: uuid.UUID,
    data: KnowledgeBaseUpdate,
    current_user: User = Depends(deps.get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service)
):
    """Update a knowledge base."""
    return await service.update_knowledge_base(kb_id=kb_id, user_id=current_user.id, data=data)

@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_base(
    kb_id: uuid.UUID,
    current_user: User = Depends(deps.get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service)
):
    """Delete a knowledge base."""
    await service.delete_knowledge_base(kb_id=kb_id, user_id=current_user.id)

@router.get("/{kb_id}/documents", response_model=List[KnowledgeDocumentRead])
async def list_knowledge_base_documents(
    kb_id: uuid.UUID,
    current_user: User = Depends(deps.get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service)
):
    """List documents in a specific knowledge base."""
    return await service.get_kb_documents(kb_id=kb_id, user_id=current_user.id)
