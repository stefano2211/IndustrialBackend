"""
CollectorService — Orchestrates on-demand and scheduled data collection
from external databases for MLOps fine-tuning.
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from app.domain.proactiva.db_collector.connectors.registry import get_connector
from app.domain.proactiva.db_collector.encryption import decrypt
from app.domain.proactiva.db_collector.formatter import rows_to_sharegpt
from app.domain.schemas.db_source import DbSource, DbSourceStatus
from app.persistence.db import async_session_factory
from app.core.mothership_client import mothership_client
from app.core.config import settings


@dataclass
class CollectionResult:
    source_id: str
    source_name: str
    status: DbSourceStatus
    rows_fetched: int = 0
    entries_generated: int = 0
    error_detail: Optional[str] = None


class CollectorService:
    """
    Runs data collection jobs for DbSource entries.

    Responsibilities:
      - Decrypt connection strings.
      - Route to the correct connector (via registry).
      - Format rows → ShareGPT pairs.
      - Write to a temp JSONL and upload to the Mothership.
      - Update run metadata in DB.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_source(self, source: DbSource) -> CollectionResult:
        """Execute a single DbSource collection job."""
        result = CollectionResult(
            source_id=str(source.id),
            source_name=source.name,
            status=DbSourceStatus.PENDING,
        )

        logger.info(f"[DbCollector] Running collection for source: {source.name} ({source.db_type})")

        try:
            connector = get_connector(source.db_type)
            connection_string = decrypt(source.connection_string_enc)

            rows = await connector.fetch(connection_string, source.query)
            result.rows_fetched = len(rows)

            # --- SMART DELTA SLICING (in memory) ---
            # Previene reenviar el histórico completo a la Mothership
            if source.accumulated_rows > 0:
                if len(rows) > source.accumulated_rows:
                    rows = rows[source.accumulated_rows:]
                    logger.info(f"[DbCollector] Smart Slicing: Only sending {len(rows)} new rows.")
                elif len(rows) < source.accumulated_rows:
                    logger.warning(f"[DbCollector] DB source rows ({len(rows)}) < accumulated ({source.accumulated_rows}). Source DB was likely reset. Resetting counter and uploading all data.")
                    source.accumulated_rows = 0
                else:
                    logger.info("[DbCollector] No new rows detected since last run. Skipping upload.")
                    result.status = DbSourceStatus.NO_DATA
                    await self._update_metadata(source, result)
                    return result

            if not rows:
                logger.warning(f"[DbCollector] No rows returned for source: {source.name}")
                result.status = DbSourceStatus.NO_DATA
                await self._update_metadata(source, result)
                return result

            # Save the count of new rows BEFORE formatting (formatter adds a summary entry
            # for multi-row results, so entries_generated != new_rows_count).
            new_rows_count = len(rows)

            entries = rows_to_sharegpt(rows, source.name, source.sector, source.domain)
            result.entries_generated = len(entries)

            # Write to temp file and upload
            success = await self._upload_entries(entries, source)
            result.status = DbSourceStatus.SUCCESS if success else DbSourceStatus.ERROR

            if success:
                # Accumulate DB rows processed, NOT training entries (which include the summary).
                source.accumulated_rows += new_rows_count
                # Hybrid: detect anomalies in collected rows and emit reactive events
                await self._detect_anomalies(rows, source)

        except Exception as exc:
            error_msg = (str(exc) + "")[:900]
            logger.error(f"[DbCollector] Error on source '{source.name}': {error_msg}")
            result.status = DbSourceStatus.ERROR
            result.error_detail = error_msg

        await self._update_metadata(source, result)
        return result

    async def run_all_enabled(self) -> List[CollectionResult]:
        """Run collection for every enabled DbSource. Called by the scheduler."""
        from app.persistence.proactiva.repositories.db_source_repository import DbSourceRepository

        results = []
        async with async_session_factory() as session:
            repo = DbSourceRepository(session)
            sources = await repo.get_all_enabled()

        logger.info(f"[DbCollector] Scheduled run — {len(sources)} enabled sources.")

        for source in sources:
            result = await self.run_source(source)
            results.append(result)

        return results

    async def run_source_by_id(self, source_id) -> Optional[CollectionResult]:
        """Manual on-demand run for a specific source (used by API endpoint)."""
        from app.persistence.proactiva.repositories.db_source_repository import DbSourceRepository

        async with async_session_factory() as session:
            repo = DbSourceRepository(session)
            source = await repo.get_by_id(source_id)

        if not source:
            return None

        return await self.run_source(source)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _upload_entries(
        self, entries: List[Dict[str, Any]], source: DbSource
    ) -> bool:
        """Serialize entries to a temp JSONL and upload to Mothership."""
        safe_name = re.sub(r"[^a-zA-Z0-9]", "_", source.name)
        tmp_path = f"/tmp/{source.tenant_id}_{safe_name}.jsonl"

        def _write_jsonl():
            with open(tmp_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        try:
            await asyncio.to_thread(_write_jsonl)

            logger.info(
                f"[DbCollector] Uploading {len(entries)} entries for '{source.name}' → Mothership"
            )
            success = await mothership_client.upload_dataset(
                tmp_path,
                tenant_id=source.tenant_id,
                tool_name=safe_name,
            )
            return success

        except Exception as exc:
            logger.error(f"[DbCollector] Upload failed for '{source.name}': {exc}")
            return False
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    async def _detect_anomalies(
        self, rows: List[Dict[str, Any]], source: DbSource
    ) -> None:
        """
        Hybrid DB Collector: scan collected rows for anomaly indicators and
        emit reactive events when patterns are detected.

        Simple heuristics (extensible):
          - Any row with a key containing 'error', 'alarm', 'fault', 'critical',
            'anomaly', 'fail' in its value triggers a reactive event.
        """
        try:
            ANOMALY_KEYWORDS = {"error", "alarm", "fault", "critical", "anomaly", "fail", "warning", "alert"}
            anomaly_rows = []
            for row in rows:
                row_text = json.dumps(row, default=str).lower()
                if any(kw in row_text for kw in ANOMALY_KEYWORDS):
                    anomaly_rows.append(row)

            if not anomaly_rows:
                return

            severity = "high" if len(anomaly_rows) > 5 else "medium"
            title = f"[DB Collector] Anomaly detected in '{source.name}'"
            description = (
                f"{len(anomaly_rows)} out of {len(rows)} rows from source '{source.name}' "
                f"({source.db_type}) contain anomaly indicators."
            )
            payload = {
                "source_name": source.name,
                "db_type": source.db_type,
                "total_rows": len(rows),
                "anomaly_rows": len(anomaly_rows),
                "sample": anomaly_rows[:3],
            }

            from app.domain.reactiva.events.event_service import EventProcessorService
            svc = EventProcessorService()
            await svc.enqueue_event(
                source_type="db_collector",
                severity=severity,
                title=title,
                description=description,
                raw_payload=payload,
                tenant_id=source.tenant_id,
            )
            logger.warning(
                f"[DbCollector] Anomaly event emitted for '{source.name}': "
                f"{len(anomaly_rows)} anomalous rows (severity={severity})"
            )
        except Exception as exc:
            logger.error(f"[DbCollector] Anomaly detection failed for '{source.name}': {exc}")

    async def _update_metadata(self, source: DbSource, result: CollectionResult):
        """Persist last_run_at, last_run_status, last_run_rows, last_error_detail."""
        from app.persistence.proactiva.repositories.db_source_repository import DbSourceRepository

        try:
            async with async_session_factory() as session:
                repo = DbSourceRepository(session)
                db_source = await repo.get_by_id(source.id)
                if db_source:
                    db_source.last_run_at = datetime.now(timezone.utc).replace(tzinfo=None)
                    db_source.last_run_status = result.status
                    db_source.last_run_rows = result.rows_fetched
                    db_source.last_error_detail = result.error_detail
                    
                    # Also update auto-trigger counters modified during run
                    db_source.accumulated_rows = source.accumulated_rows
                    
                    await repo.save(db_source)
        except Exception as exc:
            logger.error(f"[DbCollector] Failed to update metadata for '{source.name}': {exc}")


# Singleton
collector_service = CollectorService()
