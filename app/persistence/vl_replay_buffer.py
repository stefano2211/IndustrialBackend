"""
VL Replay Buffer
================
Almacena trayectorias de Computer Use (screenshots + acciones JSON) en formato
JSONL compatible con el pipeline de training VL (FastVisionModel de Unsloth).

Cada entrada = UN STEP del loop Observe-Think-Act:
  screenshot_base64 + instruction + action_json → entry JSONL

El buffer usa la misma estrategia FIFO + retención de muestras recientes
que el ReplayBuffer de texto existente.
"""

import asyncio
import json
import os
import random
from datetime import datetime
from typing import List, Dict
from loguru import logger


class VLReplayBuffer:
    """
    Buffer de experiencias Vision-Language para fine-tuning de Computer Use.

    Formato de cada entry (compatible con FastVisionModel):
    {
      "messages": [
        {"role": "user", "content": [
          {"type": "image"},
          {"type": "text", "text": "<instrucción de alto nivel>"}
        ]},
        {"role": "assistant", "content": [
          {"type": "text", "text": "{\"type\":\"click\",\"x\":450,\"y\":280}"}
        ]}
      ],
      "images": ["<base64_png_string>"],
      "metadata": {
        "tool": "computer_use",
        "timestamp": "2026-04-03T22:30:00",
        "task_id": "optional"
      }
    }
    """

    def __init__(self, file_path: str = "data/vl_replay.jsonl", max_size: int = 2000):
        self.file_path = file_path
        self.max_size = max_size

        dir_name = os.path.dirname(self.file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    async def append_trajectory_step(
        self,
        instruction: str,
        screenshot_b64: str,
        action_json: str,
        tool_name: str = "computer_use",
        task_id: str = "",
    ):
        """
        Guarda UN STEP de la trayectoria de computer use.

        Args:
            instruction: Instrucción de alto nivel que se está ejecutando
                         (recibida del Orchestrator, ej: "Open MB51 in SAP")
            screenshot_b64: Screenshot de la pantalla como string base64 PNG
            action_json: Acción ejecutada como JSON string
                         (ej: '{"type":"click","x":1450,"y":280}')
            tool_name: Nombre de la herramienta (default: "computer_use")
            task_id: ID opcional de la tarea para agrupar steps relacionados
        """
        entry = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": action_json}],
                },
            ],
            "images": [screenshot_b64],
            "metadata": {
                "tool": tool_name,
                "timestamp": datetime.utcnow().isoformat(),
                "task_id": task_id,
            },
        }
        await self._append_events([entry])

    async def append_trajectory_batch(self, steps: List[Dict]):
        """
        Guarda múltiples steps de una trayectoria completa de una sola vez.
        Cada step debe tener: instruction, screenshot_b64, action_json.
        """
        entries = []
        for step in steps:
            entries.append({
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": step["instruction"]},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": step["action_json"]}],
                    },
                ],
                "images": [step["screenshot_b64"]],
                "metadata": {
                    "tool": step.get("tool_name", "computer_use"),
                    "timestamp": datetime.utcnow().isoformat(),
                    "task_id": step.get("task_id", ""),
                },
            })
        await self._append_events(entries)

    def get_dataset_path(self) -> str:
        return self.file_path

    def count(self) -> int:
        """Retorna el número de steps almacenados."""
        if not os.path.exists(self.file_path):
            return 0
        count = 0
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except Exception as e:
            logger.error(f"[VLReplayBuffer] Error counting buffer: {e}")
        return count

    async def _append_events(self, new_events: List[Dict]):
        all_events = await asyncio.to_thread(self._read_all)
        logger.info(
            f"[VLReplayBuffer] Recibidos {len(new_events)} steps nuevos. "
            f"Total histórico: {len(all_events)}"
        )
        all_events.extend(new_events)

        if len(all_events) > self.max_size:
            all_events = self._compress_buffer(all_events)

        await asyncio.to_thread(self._write_all, all_events)
        logger.info(
            f"[VLReplayBuffer] Buffer actualizado: {len(all_events)} / {self.max_size} steps."
        )

    def _read_all(self) -> List[Dict]:
        if not os.path.exists(self.file_path):
            return []
        events = []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.error(f"[VLReplayBuffer] Error leyendo buffer: {e}")
        return events

    def _write_all(self, events: List[Dict]):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                for event in events:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"[VLReplayBuffer] Error escribiendo buffer: {e}")

    def _compress_buffer(self, all_events: List[Dict]) -> List[Dict]:
        """
        Estrategia FIFO con retención de muestras recientes.
        Mantiene el 20% más reciente intacto + samplea el resto.
        """
        logger.info("[VLReplayBuffer] Comprimiendo buffer por exceso de capacidad...")
        recent_cutoff = int(self.max_size * 0.2)
        latest = all_events[-recent_cutoff:]
        older = all_events[:-recent_cutoff]
        slots_left = self.max_size - recent_cutoff

        if slots_left > 0 and older:
            sampled = random.sample(older, min(len(older), slots_left))
            return sampled + latest
        return latest


# Singleton — se importa directamente donde se necesite
vl_replay_buffer = VLReplayBuffer()
