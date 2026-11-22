"""API-key protection and a small in-memory rate limiter for outbound notifications."""

from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from fastapi import Header, HTTPException, Request, status

from .config import get_settings


class RateLimiter:
    """Sliding-window limiter: at most `max_calls` per `window_sec` for each key.

    In-memory and per-process: good enough for a single-instance deployment.
    """

    def __init__(self, window_sec: float = 60.0) -> None:
        self.window_sec = window_sec
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, max_calls: int) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window_sec:
                hits.popleft()
            if len(hits) >= max_calls:
                return False
            hits.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


notify_limiter = RateLimiter()


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Require `X-API-Key` when API_KEY is configured. Mandatory in production."""
    settings = get_settings()
    if not settings.api_key:
        if settings.app_env == "production":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="API_KEY is not configured on the server.",
            )
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")


def rate_limit_notifications(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    limit = get_settings().notify_rate_limit_per_min
    if not notify_limiter.allow(f"ip:{client_ip}", limit):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many notification requests. Try again in a minute.",
        )
