"""
DbCollector Scheduler — APScheduler-based cron job manager.

Lifecycle:
  - start():    Load all enabled sources from DB, register cron jobs, start scheduler.
  - reload():   Recompute all jobs (call after CRUD operations via the API).
  - shutdown(): Gracefully stop the scheduler.
"""

from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger


class DbCollectorScheduler:
    """
    Manages APScheduler jobs for all active DbSource entries.
    Each source gets its own independently-scheduled job based on its cron_expression.
    """

    def __init__(self):
        self._scheduler = AsyncIOScheduler(timezone="UTC")

    async def start(self):
        """Start the scheduler and load initial jobs. Called from app lifespan."""
        self._scheduler.start()
        await self._load_all_jobs()
        logger.info("[DbCollector Scheduler] Started.")

    async def shutdown(self):
        """Shutdown gracefully. Called from app lifespan cleanup."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        logger.info("[DbCollector Scheduler] Shut down.")

    async def reload(self):
        """
        Remove all existing collector jobs and re-register from DB.
        Call this after any CREATE / UPDATE / DELETE on DbSource via the API.
        """
        logger.info("[DbCollector Scheduler] Reloading jobs...")
        self._remove_all_collector_jobs()
        await self._load_all_jobs()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _load_all_jobs(self):
        """Query DB and schedule one job per enabled source."""
        from app.persistence.repositories.db_source_repository import DbSourceRepository
        from app.persistence.db import async_session_factory
        from app.domain.db_collector.collector_service import collector_service

        try:
            async with async_session_factory() as session:
                repo = DbSourceRepository(session)
                sources = await repo.get_all_enabled()

            for source in sources:
                self._add_job(source, collector_service)

            logger.info(f"[DbCollector Scheduler] {len(sources)} jobs registered.")
        except Exception as exc:
            logger.error(f"[DbCollector Scheduler] Failed to load jobs: {exc}")

    def _add_job(self, source, collector_service):
        """Register one APScheduler cron job for the given source."""
        job_id = f"dbcollector_{source.id}"

        try:
            trigger = CronTrigger.from_crontab(source.cron_expression, timezone="UTC")
        except Exception:
            logger.warning(
                f"[DbCollector Scheduler] Invalid cron '{source.cron_expression}' "
                f"for source '{source.name}'. Defaulting to every 6 hours."
            )
            trigger = CronTrigger(hour="*/6", timezone="UTC")

        # Capture only the ID so the job always loads a fresh source object from DB.
        # Passing the SQLModel instance directly would cause the job to use the stale
        # state captured at scheduler startup, ignoring any config changes made via API.
        source_id = source.id

        async def _run_fresh():
            from app.persistence.db import async_session_factory
            from app.persistence.repositories.db_source_repository import DbSourceRepository
            async with async_session_factory() as session:
                repo = DbSourceRepository(session)
                fresh_source = await repo.get_by_id(source_id)
            if fresh_source:
                await collector_service.run_source(fresh_source)
            else:
                logger.warning(f"[DbCollector Scheduler] Source {source_id} not found at job time, skipping.")

        self._scheduler.add_job(
            func=_run_fresh,
            trigger=trigger,
            id=job_id,
            name=f"DbCollect:{source.name}",
            replace_existing=True,
            misfire_grace_time=300,  # 5-min grace window if scheduler was down
        )
        logger.debug(
            f"[DbCollector Scheduler] Job registered — {source.name} | cron: {source.cron_expression}"
        )

    def _remove_all_collector_jobs(self):
        """Remove all jobs whose ID starts with 'dbcollector_'."""
        for job in self._scheduler.get_jobs():
            if job.id.startswith("dbcollector_"):
                job.remove()


# Singleton — imported in main.py lifespan
collector_scheduler = DbCollectorScheduler()
