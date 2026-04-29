"""
Storage backends for the Deep Agent virtual file system.

Re-exported from app.domain.shared.agent.memory_backends so both
proactive and reactive domains share the same memory infrastructure
without cross-domain imports.
"""

from app.domain.shared.agent.memory_backends import (
    UserScopedStoreBackend,
    create_composite_backend,
)

__all__ = ["UserScopedStoreBackend", "create_composite_backend"]
