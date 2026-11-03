from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.errors import PayloadTooLargeFault, RateLimitFault, RequestFault
from app.services.chat_jobs import execute_chat_job
from app.utils.access import worker_error
from app.utils.rate_limit import client_key, rate_limiter
from app.utils.security import is_loose_uuid
from app.utils.validation import parse_json_body, read_request_body

router = APIRouter()

logger = logging.getLogger("app.public_worker")

MAX_BODY_BYTES = 8_192
DEFAULT_RETRY_AFTER = 60
MIN_RETRY_AFTER = 1
MAX_RETRY_AFTER = 86_400
NO_STORE = {"Cache-Control": "no-store"}


def _json(payload: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=payload, headers=NO_STORE)


def _retry_after(exc: RateLimitFault) -> str:
    raw = getattr(exc, "retry_after", None)
    try:
        seconds = int(raw) if raw is not None else DEFAULT_RETRY_AFTER
    except (TypeError, ValueError, OverflowError):
        seconds = DEFAULT_RETRY_AFTER
    return str(min(MAX_RETRY_AFTER, max(MIN_RETRY_AFTER, seconds)))


@router.post("/api/public/chat-worker")
async def post_chat_worker(request: Request) -> Any:
    error = worker_error(request)
    if error is not None:
        message, status_code = error
        return _json({"error": message}, status_code)
    try:
        rate_limiter.check("public_worker", client_key(request))
    except RateLimitFault as exc:
        return JSONResponse(
            status_code=429,
            content={"error": exc.message},
            headers={**NO_STORE, "Retry-After": _retry_after(exc)},
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("public_worker_rate_limit_failed")
        return _json({"error": "Belső kiszolgálói hiba."}, 500)
    try:
        raw = await read_request_body(request, MAX_BODY_BYTES)
        body = parse_json_body(raw)
    except PayloadTooLargeFault as exc:
        return _json({"error": exc.message}, exc.status_code)
    except RequestFault as exc:
        return _json({"error": exc.message}, exc.status_code)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("public_worker_body_failed")
        return _json({"error": "Érvénytelen kérés."}, 400)
    if not isinstance(body, dict):
        return _json({"error": "Érvénytelen kérés."}, 400)
    job_id = body.get("job_id")
    if not isinstance(job_id, str) or not is_loose_uuid(job_id):
        return _json({"error": "Érvénytelen munkaazonosító."}, 400)
    try:
        result = await execute_chat_job(job_id)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("public_worker_failed", extra={"job_id": job_id})
        return _json({"error": "A háttérmunka hibával leállt."}, 500)
    payload = result if isinstance(result, dict) else {"status": str(result)}
    try:
        content = jsonable_encoder(payload)
    except (TypeError, ValueError, RecursionError):
        logger.exception("public_worker_encode_failed", extra={"job_id": job_id})
        return _json({"error": "A háttérmunka válasza nem továbbítható."}, 500)
    return _json(content, 200)
