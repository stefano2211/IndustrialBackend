import uuid
from typing import List, Optional
from sqlmodel import Session, select
from app.domain.schemas.knowledge import KnowledgeBase, KnowledgeDocument

class KnowledgeRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_knowledge_base(self, knowledge_base: KnowledgeBase) -> KnowledgeBase:
        self.session.add(knowledge_base)
        self.session.commit()
        self.session.refresh(knowledge_base)
        return knowledge_base

    def get_knowledge_base(self, kb_id: uuid.UUID, user_id: uuid.UUID) -> Optional[KnowledgeBase]:
        statement = select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.user_id == user_id
        )
        return self.session.exec(statement).first()

    def list_knowledge_bases(self, user_id: uuid.UUID) -> List[KnowledgeBase]:
        statement = select(KnowledgeBase).where(KnowledgeBase.user_id == user_id)
        return list(self.session.exec(statement).all())

    def update_knowledge_base(self, knowledge_base: KnowledgeBase) -> KnowledgeBase:
        self.session.add(knowledge_base)
        self.session.commit()
        self.session.refresh(knowledge_base)
        return knowledge_base

    def delete_knowledge_base(self, knowledge_base: KnowledgeBase):
        self.session.delete(knowledge_base)
        self.session.commit()

    def add_document(self, document: KnowledgeDocument) -> KnowledgeDocument:
        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)
        return document

    def get_documents_by_kb(self, kb_id: uuid.UUID) -> List[KnowledgeDocument]:
        statement = select(KnowledgeDocument).where(KnowledgeDocument.knowledge_base_id == kb_id)
        return list(self.session.exec(statement).all())
        
    def delete_document(self, document: KnowledgeDocument):
        self.session.delete(document)
        self.session.commit()

    def delete_document_by_file_id(self, file_id: str):
        statement = select(KnowledgeDocument).where(KnowledgeDocument.file_id == file_id)
        docs = self.session.exec(statement).all()
        for doc in docs:
            self.session.delete(doc)
        self.session.commit()
