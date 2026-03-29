from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, Header
import re
from typing import Optional
from app.domain.services.mlops_service import MLOpsService
from app.core.config import settings
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.domain.schemas.user import User
from app.core.mothership_client import mothership_client


router = APIRouter()

class WebhookPayload(BaseModel):
    model_tag: str

def get_mlops_service() -> MLOpsService:
    return MLOpsService()

async def verify_mothership_key(x_api_key: str = Header(...)):
    """Valida que la petición proviene de la Mothership (ApiLLMOps) usando su API key."""
    if x_api_key != settings.mothership_api_key:
        raise HTTPException(status_code=401, detail="Invalid Mothership API Key")




@router.post("/webhook/model-ready")
async def ota_model_update(
    payload: WebhookPayload,
    bg_tasks: BackgroundTasks,
    service: MLOpsService = Depends(get_mlops_service),
    _: str = Depends(verify_mothership_key)
) -> dict:

    """
    Webhook Endpoint: La Nube (Mothership) llama aquí cuando el entrenamiento de los 
    nuevos datos finaliza, para que el Edge actualice su modelo (Over The Air).
    Protegido por API key de Mothership para evitar actualizaciones no autorizadas.
    """
    if not payload.model_tag:
        raise HTTPException(status_code=400, detail="Missing model_tag field")

    # Sanitizar model_tag para prevenir inyección de shell
    if not re.match(r'^[a-zA-Z0-9._:/\-]+$', payload.model_tag):
        raise HTTPException(status_code=400, detail="Invalid model_tag format. Only alphanumeric, '.', '_', ':', '/', '-' allowed.")

    bg_tasks.add_task(service.process_ota_webhook, payload.model_tag)
    
    return {
        "status": "accepted",
        "message": f"Actualización OTA para modelo {payload.model_tag} agendada."
    }

class TrainingLaunchRequest(BaseModel):
    tenant_id: str = "aura_tenant_01"
    epochs: int = 3
    webhook_url: Optional[str] = None

@router.post("/training/launch")
async def launch_training_on_cloud(
    req: TrainingLaunchRequest,
    current_user: User = Depends(get_current_user)
) -> dict:
    """
    Desencadena el proceso de Fine-Tuning de MLOps en la Mothership (Nube).
    Solo accesible por superusuarios en el Edge.
    """
    if not getattr(current_user, "is_superuser", False):
        raise HTTPException(status_code=403, detail="Superuser access required.")
        
    success = await mothership_client.trigger_training_job(
        tenant_id=req.tenant_id,
        epochs=req.epochs,
        webhook_url=req.webhook_url
    )
    if not success:
        raise HTTPException(status_code=500, detail="Error al intentar lanzar el entrenamiento en la nube.")
    
    return {"status": "success", "message": "Entrenamiento MLOps iniciado en la nube exitosamente."}
