"""
Storage backends for the Deep Agent virtual file system.

Shared between proactive and reactive agent domains.
"""

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend


class UserScopedStoreBackend(StoreBackend):
    """
    A StoreBackend that uses `user_id` (not `thread_id`) as namespace.

    This means files saved to /memories/ are shared across ALL conversations
    of the same user — enabling cross-conversation learning and preference
    retention.
    """

    def _get_namespace(self):
        """Override namespace to use user_id from the config."""
        user_id = self.runtime.config.get("configurable", {}).get("user_id", "default")
        return ("memories", user_id)


def create_composite_backend(rt):
    """
    CompositeBackend that routes:
      - /memories/* → UserScopedStoreBackend (persistent per USER, shared across threads)
      - everything else → StateBackend (ephemeral, per-thread scratch pad)
    """
    return CompositeBackend(
        default=StateBackend(rt),
        routes={
            "/memories/": UserScopedStoreBackend(rt),
        },
    )
