from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, Header
import re
from typing import Optional
from app.domain.services.mlops_service import MLOpsService
from app.domain.services.vl_mlops_service import VLMLOpsService
from app.core.config import settings
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.domain.schemas.user import User
from app.core.mothership_client import mothership_client


router = APIRouter()


class WebhookPayload(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_tag: str
    model_type: str = "text"            # "text" | "vision"
    mmproj_tag: Optional[str] = None    # Solo requerido cuando model_type="vision"


def get_mlops_service() -> MLOpsService:
    return MLOpsService()


def get_vl_mlops_service() -> VLMLOpsService:
    return VLMLOpsService()


async def verify_mothership_key(x_api_key: str = Header(...)):
    """Valida que la petición proviene de la Mothership (ApiLLMOps) usando su API key."""
    if x_api_key != settings.mothership_api_key:
        raise HTTPException(status_code=401, detail="Invalid Mothership API Key")


@router.post("/webhook/model-ready")
async def ota_model_update(
    payload: WebhookPayload,
    bg_tasks: BackgroundTasks,
    service: MLOpsService = Depends(get_mlops_service),
    vl_service: VLMLOpsService = Depends(get_vl_mlops_service),
    _: str = Depends(verify_mothership_key),
) -> dict:
    """
    Webhook OTA unificado — soporta modelos de texto y modelos VL.

    Payload para modelo de TEXTO:
      {"model_tag": "aura_tenant_01-v2", "model_type": "text"}

    Payload para modelo VL (Macrohard / Digital Optimus Local):
      {
        "model_tag": "aura_tenant_01-vl",
        "model_type": "vision",
        "mmproj_tag": "aura_tenant_01-vl-mmproj"
      }
    """
    if not payload.model_tag:
        raise HTTPException(status_code=400, detail="Missing model_tag field")

    # Sanitizar model_tag
    if not re.match(r'^[a-zA-Z0-9._:/\-]+$', payload.model_tag):
        raise HTTPException(
            status_code=400,
            detail="Invalid model_tag format. Only alphanumeric, '.', '_', ':', '/', '-' allowed.",
        )

    if payload.model_type == "vision":
        # mmproj_tag ya no es requerido — el nuevo flujo OTA descarga un único tar.gz de adaptador LoRA
        bg_tasks.add_task(
            vl_service.process_vl_ota_webhook,
            payload.model_tag,
            payload.mmproj_tag,  # Sigue aceptado por compatibilidad (se ignora internamente)
        )
        return {
            "status": "accepted",
            "model_type": "vision",
            "message": f"OTA VL agendado: {payload.model_tag}",
        }
    else:
        bg_tasks.add_task(service.process_ota_webhook, payload.model_tag)
        return {
            "status": "accepted",
            "model_type": "text",
            "message": f"OTA texto agendado: {payload.model_tag}",
        }


class TrainingLaunchRequest(BaseModel):
    tenant_id: str = "aura_tenant_01"
    epochs: int = 3
    webhook_url: Optional[str] = None


@router.post("/training/launch")
async def launch_training_on_cloud(
    req: TrainingLaunchRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Desencadena el Fine-Tuning de texto en la Mothership (pipeline existente)."""
    if not getattr(current_user, "is_superuser", False):
        raise HTTPException(status_code=403, detail="Superuser access required.")

    success = await mothership_client.trigger_training_job(
        tenant_id=req.tenant_id,
        epochs=req.epochs,
        webhook_url=req.webhook_url,
    )
    if not success:
        raise HTTPException(
            status_code=500, detail="Error al intentar lanzar el entrenamiento de texto."
        )
    return {"status": "success", "message": "Entrenamiento MLOps (texto) iniciado exitosamente."}


class VLTrainingLaunchRequest(BaseModel):
    tenant_id: str = "aura_tenant_01"
    vl_epochs: int = 2
    text_epochs: int = 1
    webhook_url: Optional[str] = None


@router.post("/training/launch-vl")
async def launch_vl_training_on_cloud(
    req: VLTrainingLaunchRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Desencadena el pipeline VL unificado (2 fases) en la Mothership.
    Requiere superuser. Dispara training de computer use + conocimiento industrial.
    """
    if not getattr(current_user, "is_superuser", False):
        raise HTTPException(status_code=403, detail="Superuser access required.")

    success = await mothership_client.trigger_vl_training_job(
        tenant_id=req.tenant_id,
        vl_epochs=req.vl_epochs,
        text_epochs=req.text_epochs,
        webhook_url=req.webhook_url,
    )
    if not success:
        raise HTTPException(
            status_code=500, detail="Error al intentar lanzar el entrenamiento VL."
        )
    return {"status": "success", "message": "Pipeline VL unificado iniciado en la nube."}

