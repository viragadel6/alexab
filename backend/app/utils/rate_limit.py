from __future__ import annotations

import time
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import JSONResponse

from app.errors import RateLimitFault

RATE_LIMITS: dict[str, tuple[int, int]] = {
    "list": (60, 60_000),
    "status": (120, 60_000),
    "frame": (180, 60_000),
    "input": (1200, 60_000),
    "mode": (20, 60_000),
    "chat": (30, 60_000),
    "conversations": (60, 60_000),
    "operator": (60, 60_000),
    "images": (20, 60_000),
    "public_worker": (120, 60_000),
    "system": (120, 60_000),
    "heal": (12, 60_000),
}


@dataclass
class Bucket:
    count: int
    reset_at: int


class RateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, Bucket] = {}

    def check(self, name: str, key: str) -> None:
        limit, window_ms = RATE_LIMITS[name]
        bucket_key = f"{name}:{key}"
        now = int(time.time() * 1000)
        bucket = self._buckets.get(bucket_key)
        if bucket is None or bucket.reset_at <= now:
            self._buckets[bucket_key] = Bucket(count=1, reset_at=now + window_ms)
            self._evict(now)
            return
        if bucket.count >= limit:
            retry_after = max(1, -(-(bucket.reset_at - now) // 1000))
            raise RateLimitFault(retry_after=retry_after)
        bucket.count += 1

    def _evict(self, now: int) -> None:
        if len(self._buckets) <= 50_000:
            return
        expired = [key for key, bucket in self._buckets.items() if bucket.reset_at <= now]
        for key in expired:
            self._buckets.pop(key, None)


rate_limiter = RateLimiter()


def client_key(request: Request, user_id: str | None = None) -> str:
    if user_id:
        return user_id
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client is not None:
        return request.client.host
    return "unknown"


def rate_limit_response(error: RateLimitFault) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"error": error.message, "code": error.code},
        headers={"Retry-After": str(error.retry_after), "Cache-Control": "no-store"},
    )
