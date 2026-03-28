import json
import os
import random
from typing import List, Dict
from loguru import logger

class ReplayBuffer:
    """
    Gestiona el Experience Replay Buffer para Fine-Tuning de LLMs.
    Implementa una estrategia FIFO con retención de anomalías.
    """

    def __init__(self, file_path: str = "data/dataset_replay.jsonl", max_size: int = 5000):
        self.file_path = file_path
        self.max_size = max_size

        # Ensure directory exists (guard against empty dirname when path has no dir component)
        dir_name = os.path.dirname(self.file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    def append_events(self, new_events: List[Dict]):
        """
        Añade nuevos eventos JSON en formato ShareGPT / ChatML.
        Aplica purga si excede max_size.
        """
        all_events = self._read_all()
        
        logger.info(f"[ReplayBuffer] Recibidos {len(new_events)} nuevos eventos. Históricos: {len(all_events)}")
        
        # Validate format
        for event in new_events:
            if "conversations" not in event:
                logger.warning(f"[ReplayBuffer] Evento ignorado (formato inválido): {str(event)[:50]}...")
                continue
            all_events.append(event)
            
        # Comprimir si excede max_size
        if len(all_events) > self.max_size:
            all_events = self._compress_buffer(all_events)
            
        self._write_all(all_events)
        logger.info(f"[ReplayBuffer] Búffer actualizado. Total actual: {len(all_events)} / {self.max_size}")
        
    def get_dataset_path(self) -> str:
        return self.file_path

    def _read_all(self) -> List[Dict]:
        if not os.path.exists(self.file_path):
            return []
        events = []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
        except Exception as e:
            logger.error(f"[ReplayBuffer] Error leyendo buffer: {e}")
        return events

    def _write_all(self, events: List[Dict]):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                for event in events:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"[ReplayBuffer] Error escribiendo buffer: {e}")

    def _compress_buffer(self, all_events: List[Dict]) -> List[Dict]:
        """
        Estrategia de Eviction:
        - Mantiene los eventos más recientes.
        - Descarta datos antiguos "normales", retiene anomalías (Importance Sampling abstracto).
        """
        logger.info("[ReplayBuffer] Comprimiendo búffer por exceso de capacidad...")
        
        # En una versión avanzada, buscaríamos un tag "is_anomaly: true" 
        # en el JSON para filtrar. Por ahora simulamos retener los últimos 'max_size'.
        # Mezclamos un poco para no borrar solo los bloques enteros (Data Blending puro)
        
        # Keep the latest 20% untouched
        recent_cutoff = int(self.max_size * 0.2)
        latest_events = all_events[-recent_cutoff:]
        
        # Randomly sample the rest to fill the quota
        older_events = all_events[:-recent_cutoff]
        slots_left = self.max_size - recent_cutoff
        
        if slots_left > 0 and older_events:
            sampled_old = random.sample(older_events, min(len(older_events), slots_left))
            compressed = sampled_old + latest_events
        else:
            compressed = latest_events
            
        return compressed

# Singleton instance
replay_buffer = ReplayBuffer()
