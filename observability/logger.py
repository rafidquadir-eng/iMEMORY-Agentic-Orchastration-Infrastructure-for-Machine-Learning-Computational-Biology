"""Minimal structured stdout logger shared across modules."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone


def log(event: str, **fields) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    print(json.dumps(record), file=sys.stderr)
