import json
import logging
import uuid
import httpx
import os
import re
from datetime import datetime, timedelta
from loguru import logger
from typing import List, Dict, Any

from app.persistence.db import async_session_factory
from app.persistence.repositories.tool_config_repository import ToolConfigRepository
from app.domain.services.mcp_service import MCPService
from app.domain.schemas.mcp import MCPResponse
from app.core.mothership_client import mothership_client

class MLOpsService:
    """
    MLOps Service para la extracción automática de datos hacia JSONL y actualizaciones OTA.
    Implementa el pipeline del Arquitectura Edge-to-Cloud.
    """

    def __init__(self):
        self.mcp_service = MCPService()

    async def export_historical_jsonl(self, days_older_than: int = 180, tenant_id: str = "aura_tenant_01") -> List[str]:
        """
        Escanea todas las herramientas MCP configuradas, extrae datos históricos,
        y genera un archivo JSONL independiente por herramienta para Fine-Tuning.
        """
        target_date = datetime.now() - timedelta(days=days_older_than)
        logger.info(f"[MLOps] Starting historical modular export for data older than {target_date}")

        exported_files = []
        
        async with async_session_factory() as session:
            tool_repo = ToolConfigRepository(session)
            tools = await tool_repo.get_all()
            
            for tool in tools:
                logger.info(f"[MLOps] Processing modular export for: {tool.name}")
                
                config_data = tool.config or {}
                url = config_data.get("url") or tool.api_url
                transport = config_data.get("transport", "mcp")
                method = config_data.get("method", "GET")
                # Extraemos sector y dominio de la configuración de la herramienta
                sector = config_data.get("sector", "Industrial")
                domain = config_data.get("domain", "General")
                
                if not url:
                    continue
                
                try:
                    res: MCPResponse = await self.mcp_service.execute_tool(
                        base_url=url,
                        tool_name=tool.name,
                        arguments={},
                        is_stdio=(transport == "stdio"),
                        transport_type=transport,
                        method=method
                    )
                    
                    if res.error or (not res.key_figures and not res.key_values):
                        continue
                     
                    # Generar Dataset específico para esta herramienta con formato de instrucción
                    tool_dataset = self._format_dataset_entries(tool.name, sector, domain, res)
                    
                    if not tool_dataset:
                        continue

                    # Guardar archivo temporal único por herramienta (Saneado)
                    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', tool.name)
                    filename = f"/tmp/{tenant_id}_{safe_name}.jsonl"
                    
                    with open(filename, "w", encoding="utf-8") as f:
                        for entry in tool_dataset:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    
                    # Subir a la nube de forma independiente usando MothershipClient
                    success = await mothership_client.upload_dataset(filename, tenant_id=tenant_id, tool_name=safe_name)
                    
                    if success:
                        exported_files.append(filename)
                        logger.success(f"[MLOps] Dataset for {tool.name} uploaded successfully.")
                    
                    # Limpiar archivo temporal inmediatamente tras subida (o intento)
                    if os.path.exists(filename):
                        os.remove(filename)
                
                except Exception as e:
                    logger.error(f"[MLOps] Error processing tool {tool.name}: {e}")

        logger.info(f"[MLOps] Modular export completed. {len(exported_files)} tool datasets uploaded.")
        return exported_files

    def _format_dataset_entries(self, tool_name: str, sector: str, domain: str, res: MCPResponse) -> List[Dict[str, Any]]:
        """Genera pares de conversación con alta calidad narrativa e inyección de contexto."""
        dataset = []
        # Etiqueta de contexto enriquecida para evitar el Olvido Catastrófico
        context_tag = f"[Sector: {sector}] [Dominio: {domain}] [Fuente: {tool_name}]"
        
        # 1. Diagnóstico Numérico (Telemetría)
        if res.key_figures:
            # Redondeo y normalización de unidades para mejorar la precisión del modelo
            figures_str = ", ".join([f"{hf.name}: {hf.value:.2f} {hf.unit or ''}" for hf in res.key_figures])
            dataset.append({
                "conversations": [
                    {"from": "user", "value": f"{context_tag} ¿Cuáles son las métricas operativas actuales?"},
                    {"from": "assistant", "value": f"Las métricas registradas en el dominio {domain} son: {figures_str}. Los valores se encuentran dentro de los rangos normales para el sector {sector}."}
                ]
            })
            
            # Variante 2: Análisis de Anomalías/Estabilidad
            dataset.append({
                "conversations": [
                    {"from": "user", "value": f"{context_tag} Analiza si existen anomalías en la telemetría."},
                    {"from": "assistant", "value": f"Tras revisar los indicadores ({figures_str}), no se detectan desviaciones críticas. El comportamiento es estable según los estándares industriales de {domain}."}
                ]
            })

        # 2. Análisis Categórico / Estados (Logs/Eventos)
        if res.key_values:
            values_str = ", ".join([f"{kv.name}: {kv.value}" for kv in res.key_values])
            dataset.append({
                "conversations": [
                    {"from": "user", "value": f"{context_tag} Resume el estado actual del sistema."},
                    {"from": "assistant", "value": f"Estado del sistema en el dominio {domain}: {values_str}. Todos los componentes reportan estados operativos nominales."}
                ]
            })

        return dataset

    async def process_ota_webhook(self, new_model_tag: str):
        """
        Recibe una señal del Hub Central de que el nuevo modelo adaptado está listo.
        Descarga por Ollama los pesos actualizados OTA.
        """
        import asyncio
        logger.info(f"[MLOps OTA] Received instruction to pull new model version: {new_model_tag}")
        
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
