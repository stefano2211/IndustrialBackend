from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from typing import Dict, Any
from loguru import logger

from app.core.config import settings
from app.domain.services.dataset_curator import dataset_curator_service
from app.core.mothership_client import mothership_client
from app.persistence.replay_buffer import replay_buffer
from app.api.deps import get_current_user
from app.domain.schemas.user import User

router = APIRouter()

async def execute_curation_pipeline(auto_train: bool = False):
    """Background task para no bloquear la petición de red"""
    logger.info("[MLOps Endpoint] Ejecutando Pipeline de Curación en Background...")
    
    # 1. Curar datos con DeepAgent (MCP)
    curation_result = await dataset_curator_service.curate_daily_data()
    if curation_result.get("status") != "success":
        logger.error(f"[MLOps Endpoint] La curación falló o no extrajo datos: {curation_result}")
        return

    # 2. Resincronizar con S3/Mothership
    tenant_id = getattr(settings, "mlops_tenant_id", "aura_tenant_01")
    upload_success = await mothership_client.upload_dataset(
        file_path=replay_buffer.get_dataset_path(),
        tenant_id=tenant_id
    )
    
    # 3. Disparar Fine-Tuning solo si auto_train=True (opt-in explícito)
    if upload_success and auto_train:
        edge_base_url = getattr(settings, "edge_public_url", "http://localhost:8000")
        webhook_url = f"{edge_base_url}/mlops/webhook/model-ready"
        triggered = await mothership_client.trigger_training_job(
            tenant_id=tenant_id,
            epochs=3,
            webhook_url=webhook_url
        )
        if triggered:
            logger.success(f"[MLOps Endpoint] Training job disparado. Webhook OTA en: {webhook_url}")
        else:
            logger.error("[MLOps Endpoint] Fallo al disparar el training job en la Mothership.")
    elif upload_success:
        logger.info("[MLOps Endpoint] Subida completada. Training no automático (auto_train=False).")


@router.post("/run-daily")
async def trigger_daily_curation(
    background_tasks: BackgroundTasks,
    auto_train: bool = False,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Desencadena la recolección automática de telemetría, 
    conversión a ShareGPT y subida a la nube.
    Diseñado para ser llamado por un Cron o manualmente por un Admin.
    
    - `auto_train=false` (default): Solo sube el dataset a MinIO. El training debe dispararse manualmente.
    - `auto_train=true`: Sube el dataset Y dispara el job de Fine-Tuning en ApiLLMOps automáticamente.
    """
    if not current_user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins can trigger dataset curation manually")
        
    # Despachar tarea en background porque el LLM curador puede tardar varios minutos
    background_tasks.add_task(execute_curation_pipeline, auto_train=auto_train)
    
    return {
        "status": "accepted",
        "message": "Dataset Curation pipeline has been queued. Monitoring client telemetry via MCP.",
        "auto_train": auto_train,
        "triggered_by": current_user.email
    }
