"""MinIO client for the reactive domain — isolated bucket.

Inherits all methods from MinIOClient but points to the
`reactive-bucket` bucket. Same MinIO server and credentials,
different namespace.
"""

from app.persistence.blob import MinIOClient
from app.core.config import settings


class ReactiveMinIOClient(MinIOClient):
    """MinIO wrapper for the reactive domain — isolated bucket."""

    def __init__(self):
        super().__init__()
        self.bucket = settings.reactive_minio_bucket
        self._ensure_bucket()


reactive_minio_client = ReactiveMinIOClient()
