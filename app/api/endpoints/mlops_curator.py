from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from typing import Dict, Any
from loguru import logger

from app.domain.services.dataset_curator import dataset_curator_service
from app.core.mothership_client import mothership_client
from app.persistence.replay_buffer import replay_buffer
from app.api.deps import get_current_user
from app.domain.schemas.user import User

router = APIRouter()

async def execute_curation_pipeline():
    """Background task para no bloquear la petición de red"""
    logger.info("[MLOps Endpoint] Ejecutando Pipeline de Curación en Background...")
    
    # 1. Curar datos con DeepAgent (MCP)
    curation_result = await dataset_curator_service.curate_daily_data()
    if curation_result.get("status") != "success":
        logger.error(f"[MLOps Endpoint] La curación falló o no extrajo datos: {curation_result}")
        return

    # 2. Resincronizar con S3/Mothership
    upload_success = await mothership_client.upload_dataset(
        file_path=replay_buffer.get_dataset_path(),
        tenant_id="aura_tenant_01"  # Or dynamic from settings/request
    )
    
    # 3. Disparar Finetuning solo temporalmente (o mediante parámetro)
    if upload_success:
        logger.info("[MLOps Endpoint] Subida completada. Opcionalmente podrías llamar a trigger_training_job() aquí.")
        # await mothership_client.trigger_training_job(...)


@router.post("/run-daily")
async def trigger_daily_curation(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Desencadena la recolección automática de telemetría, 
    conversión a ShareGPT y subida a la nube.
    Diseñado para ser llamado por un Cron o manualmente por un Admin.
    """
    if not current_user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins can trigger dataset curation manually")
        
    # Despachar tarea en background porque el LLM curador puede tardar varios minutos
    background_tasks.add_task(execute_curation_pipeline)
    
    return {
        "status": "accepted",
        "message": "Dataset Curation pipeline has been queued. Monitoring client telemetry via MCP.",
        "triggered_by": current_user.email
    }
