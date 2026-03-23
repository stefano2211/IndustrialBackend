from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from typing import Any
from app.domain.services.mlops_service import MLOpsService
from pydantic import BaseModel
import os

router = APIRouter()

class WebhookPayload(BaseModel):
    model_tag: str

def get_mlops_service() -> MLOpsService:
    return MLOpsService()

@router.post("/export-historical")
async def export_historical_data(
    bg_tasks: BackgroundTasks,
    service: MLOpsService = Depends(get_mlops_service)
) -> Any:
    """
    Inicia la recopilación de datos de orígenes MCP mayores a 6 meses
    y genera un JSONL para Fine-Tuning (Hub and Spoke pattern).
    """
    # Ejecutamos asíncronamente porque el escaneo a múltiples fuentes MCP puede ser lento
    bg_tasks.add_task(service.export_historical_jsonl, 180)
    return {
        "status": "processing",
        "message": "Extracción histórica iniciada. El archivo JSONL será generado en background."
    }

@router.post("/webhook/model-ready")
async def ota_model_update(
    payload: WebhookPayload,
    bg_tasks: BackgroundTasks,
    service: MLOpsService = Depends(get_mlops_service)
) -> Any:
    """
    Webhook Endpoint: La Nube (Mothership) llama aquí cuando el entrenamiento de los 
    nuevos datos finaliza, para que el Edge actualice su modelo (Over The Air).
    """
    if not payload.model_tag:
        raise HTTPException(status_code=400, detail="Missing model_tag field")
        
    bg_tasks.add_task(service.process_ota_webhook, payload.model_tag)
    
    return {
        "status": "accepted",
        "message": f"Actualización OTA para modelo {payload.model_tag} agendada."
    }
