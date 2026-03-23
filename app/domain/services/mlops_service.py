import json
import logging
import uuid
import httpx
from datetime import datetime, timedelta
from loguru import logger
from typing import List, Dict, Any

from app.persistence.db import async_session_factory
from app.persistence.repositories.tool_config_repository import ToolConfigRepository
from app.domain.services.mcp_service import MCPService
from app.domain.schemas.mcp import MCPResponse

class MLOpsService:
    """
    MLOps Service para la extracción automática de datos hacia JSONL y actualizaciones OTA.
    Implementa el pipeline del Arquitectura Edge-to-Cloud.
    """

    def __init__(self):
        self.mcp_service = MCPService()

    async def export_historical_jsonl(self, days_older_than: int = 180) -> str:
        """
        Escanea todas las herramientas MCP configuradas, extrae datos históricos,
        y genera un archivo JSONL etiquetado por contexto para Fine-Tuning de Models.
        """
        target_date = datetime.now() - timedelta(days=days_older_than)
        logger.info(f"[MLOps] Starting historical export for data older than {target_date}")

        dataset = []
        
        async with async_session_factory() as session:
            tool_repo = ToolConfigRepository(session)
            # Extracción universal de clientes
            tools = await tool_repo.get_all()
            
            for tool in tools:
                logger.info(f"[MLOps] Extracting from tool: {tool.name}")
                # Mock extraction logic or real execution if endpoint allows fetching by date
                # In a real environment, we would pass date filters if supported by parameter_schema.
                
                config_data = tool.config or {}
                url = config_data.get("url") or tool.api_url
                transport = config_data.get("transport", "mcp")
                method = config_data.get("method", "GET")
                
                if not url:
                    continue
                
                # Intentamos obtener datos. Aquí se asume que proveemos fechas amplias en los argumentos
                # si las soporta
                # (Para esta versión, si la API no requiere filtros, pedimos data general)
                try:
                    res: MCPResponse = await self.mcp_service.execute_tool(
                        base_url=url,
                        tool_name=tool.name,
                        arguments={}, # O podríamos enviar un argumento de 'start_date': target_date.timestamp()
                        is_stdio=(transport == "stdio"),
                        transport_type=transport,
                        method=method
                    )
                    
                    if res.error:
                        continue
                     
                    # Transform Data to Prompt-Completion format (ChatML or JSONL pairs)
                    # We inject a specific Context tag per tool to avoid Catastrophic Forgetting
                    # and mixed-domain hallucinations.
                    context_tag = f"[Contexto: {tool.name.replace('_', ' ').title()}]"
                    
                    if res.key_figures:
                        figures_str = ", ".join([f"{hf.name}: {hf.value}" for hf in res.key_figures])
                        completion = f"Los indicadores históricos muestran: {figures_str}"
                        dataset.append({
                            "prompt": f"{context_tag} ¿Cuáles eran las métricas operativas registradas?",
                            "completion": completion
                        })

                    if res.key_values:
                        values_str = ", ".join([f"{hv.name}: {hv.value}" for hv in res.key_values])
                        dataset.append({
                            "prompt": f"{context_tag} Genera un reporte de estados guardados.",
                            "completion": f"Reporte de estado histórico: {values_str}"
                        })

                except Exception as e:
                    logger.error(f"[MLOps] Extraction error for {tool.name}: {e}")
                    
        # Escribir a sistema de archivos local para OTA update
        import os
        export_path = f"/tmp/historical_dataset_{datetime.now().strftime('%Y%m%d%H%M')}.jsonl"
        with open(export_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        # Subir historial a la Nube (ApiLLMOps / Mothership) invocando el client nativo
        import os
        from app.core.mothership_client import mothership_client
        
        tenant_id = os.getenv("TENANT_ID", "default")
        logger.info(f"[MLOps] Uploading dataset {export_path} to Mothership using MothershipClient...")
        
        upload_success = await mothership_client.upload_dataset(export_path, tenant_id=tenant_id)
        
        if upload_success:
            logger.info("[MLOps] History successfully dispatched to cloud!")
        else:
            logger.error("[MLOps] Failed to sync local history to cloud.")
                
        logger.info(f"[MLOps] Export completed. {len(dataset)} examples exported to {export_path}")
        return export_path
        
    async def process_ota_webhook(self, new_model_tag: str):
        """
        Recibe una señal del Hub Central (Servidor Nube) de que el nuevo modelo entrenado
        con este historial está listo. Descarga por Ollama los pesos actualizados OTA.
        """
        import asyncio
        logger.info(f"[MLOps OTA] Received instruction to pull new model version: {new_model_tag}")
        
        # Ejecuta ollama pull asíncronamente
        try:
            process = await asyncio.create_subprocess_shell(
                f"ollama pull {new_model_tag}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"[MLOps OTA] Successfully pulled {new_model_tag}")
            else:
                logger.error(f"[MLOps OTA] Failed to pull {new_model_tag}. Error: {stderr.decode()}")
        except Exception as e:
             logger.error(f"[MLOps OTA] Exception during model pull: {e}")
        
        return {"status": "success", "tag": new_model_tag}
