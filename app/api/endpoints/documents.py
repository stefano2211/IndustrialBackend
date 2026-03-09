import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from typing import Optional
from app.domain.services.document_service import DocumentService
from app.domain.services.knowledge_service import KnowledgeService
from app.domain.exceptions import NotFoundError
from app.api.endpoints.knowledge import get_knowledge_service
from app.api import deps
from app.domain.schemas.user import User

router = APIRouter(dependencies=[Depends(deps.get_current_user)])
document_service = DocumentService()


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    knowledge_base_id: Optional[str] = Form(None),
    current_user: User = Depends(deps.get_current_user),
    kb_service: KnowledgeService = Depends(get_knowledge_service),
):
    allowed_extensions = {".pdf", ".doc", ".docx", ".json"}
    filename = file.filename.lower() if file.filename else ""
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only .pdf, .doc, .docx, and .json files are supported.",
        )

    upload_result = await document_service.upload_document(
        file, user_id=str(current_user.id), knowledge_base_id=knowledge_base_id
    )

    if knowledge_base_id:
        await kb_service.add_document_to_kb(
            kb_id=uuid.UUID(knowledge_base_id),
            user_id=current_user.id,
            file_id=upload_result["file_id"],
            filename=file.filename,
        )

    return upload_result


@router.get("/status/{task_id}")
async def status(task_id: str):
    return document_service.get_task_status(task_id)


@router.get("/documents/{doc_id}")
async def get_document_details(
    doc_id: str,
    current_user: User = Depends(deps.get_current_user),
):
    """Recupera los detalles de un documento procesado."""
    details = document_service.get_document_details(doc_id, user_id=str(current_user.id))
    if not details:
        raise HTTPException(status_code=404, detail="Document not found or access denied")
    return details


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    current_user: User = Depends(deps.get_current_user),
    kb_service: KnowledgeService = Depends(get_knowledge_service),
):
    """Elimina un documento y sus vectores asociados."""
    document_service.delete_document(doc_id, user_id=str(current_user.id))

    try:
        await kb_service.delete_document_by_file_id(doc_id)
    except Exception:
        pass

    return {"status": "deleted", "doc_id": doc_id}
