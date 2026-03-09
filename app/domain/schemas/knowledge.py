import uuid
from datetime import datetime, timezone
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship

class KnowledgeBaseBase(SQLModel):
    name: str = Field(index=True)
    description: Optional[str] = Field(default=None)

class KnowledgeBase(KnowledgeBaseBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Optional relationship if we want to load documents easily
    documents: List["KnowledgeDocument"] = Relationship(back_populates="knowledge_base", cascade_delete=True)

class KnowledgeDocumentBase(SQLModel):
    filename: str

class KnowledgeDocument(KnowledgeDocumentBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    knowledge_base_id: uuid.UUID = Field(foreign_key="knowledgebase.id", index=True)
    file_id: str = Field(index=True) # Reference to MinIO/Qdrant processing ID
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    knowledge_base: KnowledgeBase = Relationship(back_populates="documents")

# --- DTOs for API ---

class KnowledgeBaseCreate(KnowledgeBaseBase):
    pass

class KnowledgeBaseUpdate(SQLModel):
    name: Optional[str] = None
    description: Optional[str] = None

class KnowledgeDocumentRead(KnowledgeDocumentBase):
    id: uuid.UUID
    file_id: str
    created_at: datetime

class KnowledgeBaseRead(KnowledgeBaseBase):
    id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    
class KnowledgeBaseDetailRead(KnowledgeBaseRead):
    documents: List[KnowledgeDocumentRead] = []
