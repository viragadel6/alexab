from __future__ import annotations

import json
from typing import Any

SSE_MEDIA_TYPE = "text/event-stream; charset=utf-8"
NO_STORE_HEADERS = {
    "Cache-Control": "private, no-store, no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def sse_frame(payload: Any, event_id: int | str | None = None) -> str:
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    return f"{prefix}data: {body}\n\n"


def sse_comment(text: str) -> str:
    return f": {text}\n\n"


def sse_done() -> str:
    return "data: [DONE]\n\n"

