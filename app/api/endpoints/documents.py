from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.document_service import DocumentService

router = APIRouter()
document_service = DocumentService()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed_extensions = {".pdf", ".doc", ".docx", ".json"}
    filename = file.filename.lower() if file.filename else ""
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid file format. Only .pdf, .doc, .docx, and .json files are supported.")
    return await document_service.upload_document(file)

@router.get("/status/{task_id}")
async def status(task_id: str):
    return document_service.get_task_status(task_id)

@router.get("/documents/{doc_id}")
async def get_document_details(doc_id: str):
    """Recupera los detalles de un documento procesado (categoría, entidades, chunks)."""
    details = document_service.get_document_details(doc_id)
    if not details:
        raise HTTPException(status_code=404, detail="Document not found")
    return details

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Elimina un documento y sus vectores asociados."""
    return document_service.delete_document(doc_id)
