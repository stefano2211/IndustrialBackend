import json
import re
from loguru import logger

from app.domain.agent.deep_agent import create_industrial_agent
from app.core.llm import LLMFactory, LLMProvider
from app.persistence.replay_buffer import replay_buffer

class DatasetCuratorService:
    """
    Servicio de Curación de Datasets (MLOps Edge).
    Utiliza el DeepAgent interno (y sus herramientas MCP) para extraer
    el contexto operativo de maquinaria automáticamente, sin Hardcoding SQL.
    Aplica técnicas ShareGPT de "Data Storytelling" para el Fine-Tuning.
    """

    def __init__(self):
        pass

    async def curate_daily_data(self) -> dict:
        """
        Inicia la rutina de extracción usando el agente y guarda los datos textualizados.
        """
        logger.info("[Dataset Curator] Iniciando recolección diaria vía DeepAgent MCP...")

        # 1. Obtener el LLM rápido local
        llm = await LLMFactory.get_llm(
            provider=LLMProvider.OLLAMA,
            temperature=0.1, # Lower temperature for strictly formatted output
            max_tokens=2048,
            num_ctx=32768,   # Solid window for telemetry data
            streaming=False  # Disable streaming for cleaner sync in background tasks
        )
        
        # 2. Instrucciones personalizadas altamente específicas para la curación
        curator_instructions = """
        ERES EL CURADOR DE DATASETS DE MLOPS. TU OBJETIVO ES EXTRAER HISTORIAS DE OPERACIÓN PARA FINE-TUNING.
        1. Usa `call_dynamic_mcp` (o las herramientas que tengas) para inspeccionar las métricas de equipos, variables de producción y posibles anomalías de las ÚLTIMAS 24 HORAS.
        2. Analiza los resultados profundos.
        3. Crea EXACTAMENTE de 3 a 5 "historias" en la que expliques un problema u observación de la planta y cómo solucionarlo/entenderlo, basado únicamente en la data que encontraste.
        4. DEVUELVE TU RESPUESTA EXACTAMENTE EN UN BLOQUE JSON COMO ESTE, SIN markdown extra ni explicación fuera del json:
        [
          {
            "conversations": [
              {"from": "system", "value": "Eres experto en diagnósticos de MAQUINARIA INDUSTRIAL."},
              {"from": "human", "value": "<pregunta sobre el estado hallado>"},
              {"from": "gpt", "value": "<explicación de la anomalía, sus datos exactos y causa raíz según la telemetría>"}
            ]
          }
        ]
        """

        # 3. Instanciar el agente Industrial
        agent = create_industrial_agent(
            model=llm,
            custom_system_prompt=curator_instructions,
            mcp_tools_context="Descubre anomalías operacionales usando MCP y conviértelas en JSON de entrenamiento."
        )

        initial_state = {"messages": [("user", (
            "1. Obtén la telemetría de los equipos usando `call_dynamic_mcp`.\n"
            "2. Analiza los datos de las últimas 24h.\n"
            "3. Basado en esos datos, genera un array JSON de 3 a 5 historias para fine-tuning.\n"
            "FORMATO REQUERIDO: Devuelve ÚNICAMENTE el array JSON, sin explicaciones ni markdown:\n"
            "[{\"conversations\": [{\"from\": \"system\", \"value\": \"...\"}, {\"from\": \"human\", \"value\": \"...\"}, {\"from\": \"gpt\", \"value\": \"...\"}]}]"
        ))]}
        
        # 4. Invocar el agente (Orquestación Autónoma)
        logger.info("[Dataset Curator] Compilando grafo y despachando al Agente...")
        try:
            result = await agent.ainvoke(initial_state)
            final_message = result['messages'][-1].content
            logger.debug(f"[Dataset Curator] Respuesta bruta del agente (len={len(final_message)}): {final_message[:200]}...")
        except Exception as e:
            logger.error(f"[Dataset Curator] Falló el agente durante la recolección MCP: {e}")
            return {"status": "error", "detail": str(e)}

        # 5. Parsear la respuesta y limpiar la salida 
        extracted_events = self._parse_json_array(final_message)
        
        if not extracted_events:
            logger.warning("[Dataset Curator] El agente no devolvió eventos válidos o no hubo telemetría.")
            return {"status": "no_data", "details": final_message[:100]}
            
        # 6. Almacenamiento Cíclico en el Replay Buffer
        logger.info(f"[Dataset Curator] Se generaron {len(extracted_events)} anomalías/muestras de texto. Inyectando al ReplayBuffer...")
        replay_buffer.append_events(extracted_events)

        return {"status": "success", "curated_events": len(extracted_events)}

    def _parse_json_array(self, text: str) -> list:
        """Intenta extraer de forma segura el array JSON de la respuesta libre del LLM"""
        # 1. Strip thought blocks (DeepSeek/Qwen style)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        
        # 2. Strip markdown code fences (```json ... ``` or ``` ... ```)
        text_stripped = re.sub(r'```(?:json)?\s*', '', text).replace('```', '').strip()
        
        # 3. Try direct parse on stripped text
        try:
            data = json.loads(text_stripped)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        
        # 4. Try to find array brackets in original or stripped text
        for candidate in [text_stripped, text]:
            match = re.search(r'\[\s*\{.*\}\s*\]', candidate, flags=re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    if isinstance(data, list):
                        return data
                except json.JSONDecodeError:
                    pass
        
        # 5. Log what we got so it's diagnosable
        logger.debug(f"[Dataset Curator] No se pudo extraer un array JSON válido. Texto recibido (len={len(text)}): {text[:500]}")
        return []

dataset_curator_service = DatasetCuratorService()
