"""
S3 snapshot sync for the LMDB graph store.

On each significant write the LMDB data file can be snapshotted to S3 (and/or
GCP) for durability and cross-region availability. Safe no-op if boto3 or
credentials are absent, so the demo runs without any cloud setup.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class S3Connector:
    def __init__(self, bucket: Optional[str] = None):
        self.bucket = bucket or os.environ.get("IMEMORY_S3_BUCKET")
        self._client = self._make_client()

    def _make_client(self):
        if not self.bucket or not os.environ.get("AWS_ACCESS_KEY_ID"):
            return None
        try:
            import boto3  # local import keeps boto3 optional

            return boto3.client("s3")
        except Exception:  # noqa: BLE001
            return None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def snapshot(self, local_path: str, key_prefix: str = "lmdb-snapshots/") -> Optional[str]:
        """Upload the LMDB data file to S3. Returns the object key, or None if disabled."""
        if not self.enabled:
            return None
        path = Path(local_path)
        key = f"{key_prefix}{path.name}"
        self._client.upload_file(str(path), self.bucket, key)
        return key
