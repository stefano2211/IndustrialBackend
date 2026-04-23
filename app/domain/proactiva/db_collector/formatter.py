"""
Formatter: transforms raw database rows into ShareGPT-formatted
conversation pairs suitable for LLM fine-tuning.
"""

from itertools import islice
from typing import Any, Dict, List


def rows_to_sharegpt(
    rows: List[Dict[str, Any]],
    source_name: str,
    sector: str,
    domain: str,
) -> List[Dict[str, Any]]:
    """
    Converts a list of database rows into ShareGPT conversation pairs.

    Strategy:
      - Every individual row becomes one Q&A pair (precise, traceable).
      - A summary pair is generated if more than one row is returned (holistic view).

    Args:
        rows: List of dicts mapping column names to values.
        source_name: Human-readable DB source label for the context tag.
        sector: Industrial sector (e.g. "Manufactura").
        domain: Technical domain (e.g. "Sensores de Temperatura").

    Returns:
        List of ShareGPT-format dicts with a "conversations" key.
    """
    if not rows:
        return []

    context_tag = f"[Sector: {sector}] [Dominio: {domain}] [Fuente: {source_name}]"
    dataset: List[Dict[str, Any]] = []

    # --- Individual row entries ---
    for row in rows:
        row_str = ", ".join(
            f"{col}: {val}" for col, val in row.items() if val is not None
        )
        if not row_str:
            continue

        dataset.append(
            {
                "conversations": [
                    {
                        "from": "user",
                        "value": (
                            f"{context_tag} "
                            f"Reporta el registro operativo más reciente de {source_name}."
                        ),
                    },
                    {
                        "from": "assistant",
                        "value": (
                            f"Registro del sistema {source_name} en el dominio {domain}: "
                            f"{row_str}. "
                            f"Los valores se encuentran dentro de los parámetros operativos "
                            f"estándar del sector {sector}."
                        ),
                    },
                ]
            }
        )

    # --- Aggregate summary entry (only when multiple rows) ---
    if len(rows) > 1:
        # Use first row's columns as headers
        columns = list(rows[0].keys())
        # Build a compact tabular text
        summary_lines = [
            ", ".join(f"{col}: {row.get(col, 'N/A')}" for col in columns)
            for row in list(islice(rows, 10))  # Cap at 10 rows to keep tokens reasonable
        ]
        summary_text = " | ".join(summary_lines)

        dataset.append(
            {
                "conversations": [
                    {
                        "from": "user",
                        "value": (
                            f"{context_tag} "
                            f"Resume los últimos {len(rows)} registros de {source_name}."
                        ),
                    },
                    {
                        "from": "assistant",
                        "value": (
                            f"Resumen de {len(rows)} registros de {source_name} "
                            f"({domain}): {summary_text}. "
                            f"El sistema opera de forma nominal en el sector {sector}."
                        ),
                    },
                ]
            }
        )

    return dataset
