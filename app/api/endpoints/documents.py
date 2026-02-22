from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.domain.services.document_service import DocumentService
from app.api import deps
from app.domain.schemas.user import User

router = APIRouter(dependencies=[Depends(deps.get_current_user)])
document_service = DocumentService()

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(deps.get_current_user)
):
    allowed_extensions = {".pdf", ".doc", ".docx", ".json"}
    filename = file.filename.lower() if file.filename else ""
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid file format. Only .pdf, .doc, .docx, and .json files are supported.")
    return await document_service.upload_document(file, user_id=str(current_user.id))

@router.get("/status/{task_id}")
async def status(task_id: str):
    return document_service.get_task_status(task_id)

@router.get("/documents/{doc_id}")
async def get_document_details(
    doc_id: str,
    current_user: User = Depends(deps.get_current_user)
):
    """Recupera los detalles de un documento procesado (categoría, entidades, chunks)."""
    details = document_service.get_document_details(doc_id, user_id=str(current_user.id))
    if not details:
        raise HTTPException(status_code=404, detail="Document not found or access denied")
    return details

@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    current_user: User = Depends(deps.get_current_user)
):
    """Elimina un documento y sus vectores asociados."""
    return document_service.delete_document(doc_id, user_id=str(current_user.id))
