import asyncio
import sys
import uuid
from loguru import logger
from sqlmodel import select

from app.domain.schemas.tool_config import ToolConfig  # noqa
from app.persistence.db import async_session_factory
from app.domain.schemas.db_source import DbSource
from app.domain.proactiva.db_collector.collector_service import collector_service
from app.core.mothership_client import mothership_client


async def run_e2e_test():
    source_id = "bcfbcb22-555c-4bfc-9e36-3d6ad3aed45f"
    logger.info(f"--- Iniciando Prueba E2E MLOps ---")
    logger.info(f"Paso 1: Buscando DbSource ID: {source_id}")
    
    async with async_session_factory() as session:
        result = await session.execute(
            select(DbSource).where(DbSource.id == uuid.UUID(source_id))
        )
        source = result.scalars().first()
        
        if not source:
            logger.error("¡No se encontró el DbSource en la base de datos!")
            sys.exit(1)
            
        logger.info(f"Fuente encontrada: {source.name} ({source.db_type})")
        
        # PASO 2: Extracción de datos
        logger.info("\n--- Paso 2: Ejecutando Extractor de Datos ---")
        try:
            collection_result = await collector_service.run_source(source)
            logger.info(f"Resultado iteración: {collection_result.status}")
            logger.info(f"Filas procesadas: {collection_result.rows_fetched}")
            if collection_result.status.value == "error":
                logger.warning(f"Error de Extracción: {collection_result.error_detail}")
                logger.info("Fallo en extracción real. Generando Mock Dataset para continuar prueba de ApiLLMOps!")
                
        except Exception as e:
            logger.warning(f"Falla en la extracción: {e}")

        async def _subir():
            mock_path = f"/tmp/{source.id}.jsonl"
            with open(mock_path, "w") as f:
                f.write('{"messages": [{"role": "user", "content": "¿Cuál es la última lectura del sensor de temperatura 104?"}, {"role": "assistant", "content": "La última lectura válida del sensor T-104 en el horno principal fue de 810 grados centígrados, sin alertas."}]}\n')
                f.write('{"messages": [{"role": "user", "content": "Explica la tendencia del motor de refrigeración."}, {"role": "assistant", "content": "El motor de refrigeración presenta vibraciones anómalas en el eje de rotación, aumentando un 5% diario."}]}\n')
            await mothership_client.upload_dataset(mock_path, tenant_id=source.tenant_id, tool_name=None)

        if 'collection_result' not in locals() or collection_result.status.value == "error":
            await _subir()
            
        # PASO 3: Disparar el Entrenamiento
        logger.info("\n--- Paso 3: Disparando Entrenamiento en ApiLLMOps ---")
        try:
            success = await mothership_client.trigger_training_job(
                tenant_id=source.tenant_id,
                epochs=3, # Pruebas cortas
                webhook_url=None  # Utilizará el endpoint web local por defecto
            )
            if success:
                logger.success("¡Pipeline de Entrenamiento lanzado exitosamente en la Mothership!")
            else:
                logger.error("La Mothership rechazó la petición de entrenamiento.")
        except Exception as e:
            logger.error(f"Falla de red contactando a la Mothership: {e}")


if __name__ == "__main__":
    asyncio.run(run_e2e_test())
