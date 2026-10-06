"""Exact-key response cache with explicit evidence eligibility flags.

The cache is intentionally transport-agnostic. Callers provide the exact serialized request bytes,
model/backend pins, suite pin, and prompt-template bytes. Cached responses are usable for quality
replay, but cache hits are never eligible for latency or availability evidence. Errors, timeouts,
and truncated responses are never stored.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = 1


def _part(value: bytes | str) -> bytes:
    return value if isinstance(value, bytes) else value.encode("utf-8")


def _frame(value: bytes | str) -> bytes:
    raw = _part(value)
    return len(raw).to_bytes(8, "big") + raw


def response_key(*, request_bytes: bytes, model_digest: str, backend_version: str,
                 suite_pin: str, repeat_namespace: str, template_bytes: bytes) -> str:
    """Hash every semantic input with length framing; repeat namespaces cannot cross-replay."""
    material = b"".join(_frame(v) for v in (
        request_bytes, model_digest, backend_version, suite_pin, repeat_namespace, template_bytes))
    return hashlib.sha256(material).hexdigest()


@dataclass(frozen=True)
class EvidenceFlags:
    """Machine-readable evidence policy attached to every replay result."""

    cache_hit: bool
    latency_eligible: bool
    availability_eligible: bool

    def as_dict(self) -> dict[str, bool]:
        return {"cache_hit": self.cache_hit,
                "latency_eligible": self.latency_eligible,
                "availability_eligible": self.availability_eligible}


@dataclass(frozen=True)
class CachedResponse:
    body: bytes
    flags: EvidenceFlags


class ResponseCache:
    """Small content-addressed cache. Corrupt/missing entries are misses, never errors."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("cache key must be a lowercase SHA-256 hex digest")
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> CachedResponse | None:
        try:
            doc = json.loads(self._path(key).read_text(encoding="utf-8"))
            body = base64.b64decode(doc["body"], validate=True)
            if doc.get("schema") != SCHEMA or doc.get("key") != key:
                return None
            if doc.get("cacheable") is not True:
                return None
            return CachedResponse(body, EvidenceFlags(cache_hit=True, latency_eligible=False,
                                                      availability_eligible=False))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def put(self, key: str, body: bytes, *, error_kind: str | None = None,
            timed_out: bool = False, truncated: bool = False) -> bool:
        """Store only complete successful responses; return whether the response was stored."""
        if error_kind is not None or timed_out or truncated:
            return False
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc: dict[str, Any] = {"schema": SCHEMA, "key": key, "cacheable": True,
                               "body": base64.b64encode(body).decode("ascii")}
        temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temp.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
            os.replace(temp, path)
        except OSError:
            try:
                temp.unlink()
            except OSError:
                pass
            return False
        return True

    @staticmethod
    def live_flags() -> EvidenceFlags:
        """Flags for a network response; callers may count latency/availability evidence."""
        return EvidenceFlags(cache_hit=False, latency_eligible=True, availability_eligible=True)
