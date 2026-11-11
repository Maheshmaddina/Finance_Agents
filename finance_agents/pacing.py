"""Keep free-tier API calls under the provider's requests-per-minute limit.

- A sliding-window limiter spaces requests so we stay under LLM_RPM per minute.
- If the provider answers 429 "per minute" or 5xx "overloaded", we wait and resend (up to 3 times).
- A 429 "per day" is returned at once: waiting minutes won't help, so the app stops with a message.

Plugged into the OpenAI SDK as a custom httpx transport, so every lab benefits automatically.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque

# The OpenAI SDK ships its own fork of httpx (`httpx2`); older SDKs use plain httpx.
try:
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

PER_MINUTE_WAIT = 20.0  # seconds to wait after a per-minute 429 (Gemini sends no Retry-After)
MAX_RESENDS = 3


class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.calls: deque[float] = deque()
        self.lock = threading.Lock()

    def _delay(self) -> float:
        """Reserve a slot; return how long to sleep before using it."""
        with self.lock:
            now = time.monotonic()
            while self.calls and now - self.calls[0] >= 60:
                self.calls.popleft()
            if len(self.calls) < self.rpm:
                self.calls.append(now)
                return 0.0
            start = self.calls[-self.rpm] + 60  # when the oldest of the last `rpm` calls expires
            self.calls.append(start)
            return max(0.0, start - now)

    def wait(self) -> None:
        if self.rpm > 0 and (d := self._delay()):
            print(f"[pacing] waiting {d:.0f}s to stay under {self.rpm} requests/min", flush=True)
            time.sleep(d)

    async def wait_async(self) -> None:
        if self.rpm > 0 and (d := self._delay()):
            print(f"[pacing] waiting {d:.0f}s to stay under {self.rpm} requests/min", flush=True)
            await asyncio.sleep(d)


def _is_per_minute(body: bytes) -> bool:
    text = body.decode(errors="ignore").lower()
    return "per minute" in text or ("per day" not in text and "quota" not in text)


def _retry_wait(status: int, body: bytes, attempt: int) -> float | None:
    """Seconds to wait before resending, or None to give the response back as-is."""
    if status == 429:
        return PER_MINUTE_WAIT if _is_per_minute(body) else None  # daily cap: don't wait
    if status in (500, 502, 503, 504):  # "model overloaded" — temporary
        return 5.0 * 2 ** attempt       # 5s, 10s, 20s
    return None


class PacedTransport(httpx.HTTPTransport):
    def __init__(self, limiter: RateLimiter, **kw):
        super().__init__(**kw)
        self.limiter = limiter

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        for attempt in range(MAX_RESENDS + 1):
            self.limiter.wait()
            resp = super().handle_request(request)
            if resp.status_code < 429 or attempt == MAX_RESENDS:
                return resp
            wait = _retry_wait(resp.status_code, resp.read(), attempt)
            if wait is None:
                return resp
            resp.close()
            print(f"[pacing] provider returned {resp.status_code}, retrying in {wait:.0f}s", flush=True)
            time.sleep(wait)
        return resp


class PacedAsyncTransport(httpx.AsyncHTTPTransport):
    def __init__(self, limiter: RateLimiter, **kw):
        super().__init__(**kw)
        self.limiter = limiter

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        for attempt in range(MAX_RESENDS + 1):
            await self.limiter.wait_async()
            resp = await super().handle_async_request(request)
            if resp.status_code < 429 or attempt == MAX_RESENDS:
                return resp
            wait = _retry_wait(resp.status_code, await resp.aread(), attempt)
            if wait is None:
                return resp
            await resp.aclose()
            print(f"[pacing] provider returned {resp.status_code}, retrying in {wait:.0f}s", flush=True)
            await asyncio.sleep(wait)
        return resp
