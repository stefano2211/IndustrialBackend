from typing import List
from fastapi import APIRouter, Depends, status
import uuid
from sqlmodel import Session

from app.api import deps
from app.core.database import get_session
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

def get_knowledge_service(session: Session = Depends(get_session)) -> KnowledgeService:
    return KnowledgeService(session=session)

@router.post("/", response_model=KnowledgeBaseRead, status_code=status.HTTP_201_CREATED)
def create_knowledge_base(
    data: KnowledgeBaseCreate,
    current_user: User = Depends(deps.get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service)
):
    """Create a new knowledge base collection."""
    return service.create_knowledge_base(user_id=current_user.id, data=data)

@router.get("/", response_model=List[KnowledgeBaseRead])
def list_knowledge_bases(
    current_user: User = Depends(deps.get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service)
):
    """List all knowledge base collections for the current user."""
    return service.list_knowledge_bases(user_id=current_user.id)

@router.get("/{kb_id}", response_model=KnowledgeBaseDetailRead)
def get_knowledge_base(
    kb_id: uuid.UUID,
    current_user: User = Depends(deps.get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service)
):
    """Get a specific knowledge base and its documents."""
    kb = service.get_knowledge_base(kb_id=kb_id, user_id=current_user.id)
    documents = service.get_kb_documents(kb_id=kb_id, user_id=current_user.id)
    
    kb_dict = kb.model_dump()
    kb_dict["documents"] = documents
    return kb_dict

@router.patch("/{kb_id}", response_model=KnowledgeBaseRead)
def update_knowledge_base(
    kb_id: uuid.UUID,
    data: KnowledgeBaseUpdate,
    current_user: User = Depends(deps.get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service)
):
    """Update a knowledge base."""
    return service.update_knowledge_base(kb_id=kb_id, user_id=current_user.id, data=data)

@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_knowledge_base(
    kb_id: uuid.UUID,
    current_user: User = Depends(deps.get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service)
):
    """Delete a knowledge base."""
    service.delete_knowledge_base(kb_id=kb_id, user_id=current_user.id)

@router.get("/{kb_id}/documents", response_model=List[KnowledgeDocumentRead])
def list_knowledge_base_documents(
    kb_id: uuid.UUID,
    current_user: User = Depends(deps.get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service)
):
    """List documents in a specific knowledge base."""
    return service.get_kb_documents(kb_id=kb_id, user_id=current_user.id)
