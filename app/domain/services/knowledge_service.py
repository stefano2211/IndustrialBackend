import uuid
from typing import List, Optional
from sqlmodel import Session
from fastapi import HTTPException
from app.domain.schemas.knowledge import (
    KnowledgeBase,
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeDocument
)
from app.persistence.repositories.knowledge_repository import KnowledgeRepository

class KnowledgeService:
    def __init__(self, session: Session):
        self.repository = KnowledgeRepository(session)

    def create_knowledge_base(self, user_id: uuid.UUID, data: KnowledgeBaseCreate) -> KnowledgeBase:
        kb = KnowledgeBase(
            user_id=user_id,
            name=data.name,
            description=data.description
        )
        return self.repository.create_knowledge_base(kb)

    def get_knowledge_base(self, kb_id: uuid.UUID, user_id: uuid.UUID) -> KnowledgeBase:
        kb = self.repository.get_knowledge_base(kb_id, user_id)
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        return kb

    def list_knowledge_bases(self, user_id: uuid.UUID) -> List[KnowledgeBase]:
        return self.repository.list_knowledge_bases(user_id)

    def update_knowledge_base(
        self, kb_id: uuid.UUID, user_id: uuid.UUID, data: KnowledgeBaseUpdate
    ) -> KnowledgeBase:
        kb = self.get_knowledge_base(kb_id, user_id)
        
        if data.name is not None:
            kb.name = data.name
        if data.description is not None:
            kb.description = data.description
            
        return self.repository.update_knowledge_base(kb)

    def delete_knowledge_base(self, kb_id: uuid.UUID, user_id: uuid.UUID):
        kb = self.get_knowledge_base(kb_id, user_id)
        # TODO: In the future, we should also call document_service to delete vectors in Qdrant for all documents in this KB
        self.repository.delete_knowledge_base(kb)

    def add_document_to_kb(self, kb_id: uuid.UUID, user_id: uuid.UUID, file_id: str, filename: str) -> KnowledgeDocument:
        self.get_knowledge_base(kb_id, user_id) # Verify ownership
        doc = KnowledgeDocument(
            knowledge_base_id=kb_id,
            file_id=file_id,
            filename=filename
        )
        return self.repository.add_document(doc)

    def get_kb_documents(self, kb_id: uuid.UUID, user_id: uuid.UUID) -> List[KnowledgeDocument]:
        self.get_knowledge_base(kb_id, user_id) # Verify ownership
        return self.repository.get_documents_by_kb(kb_id)

    def remove_document_from_kb(self, kb_id: uuid.UUID, user_id: uuid.UUID, file_id: str):
        self.get_knowledge_base(kb_id, user_id) # Verify ownership
        docs = self.repository.get_documents_by_kb(kb_id)
        for doc in docs:
            if doc.file_id == file_id:
                self.repository.delete_document(doc)
                return True
        return False
