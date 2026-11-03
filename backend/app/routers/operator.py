from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app import database as database_module
from app.agent import cognitive as cog
from app.database import DatabaseError
from app.errors import PayloadTooLargeFault, RateLimitFault, RequestFault
from app.utils.access import Access, job_access, load_access
from app.utils.rate_limit import client_key, rate_limiter
from app.utils.security import is_loose_uuid, verify_origin
from app.utils.validation import parse_json_body, read_request_body

router = APIRouter()

logger = logging.getLogger("app.operator")

MAX_OPERATOR_MESSAGE_LENGTH = 32_000
MAX_BODY_BYTES = 262_144
DEFAULT_RETRY_AFTER = 60
MIN_RETRY_AFTER = 1
MAX_RETRY_AFTER = 86_400


def _json(
    payload: Any, status_code: int = 200, access: Access | None = None
) -> JSONResponse:
    headers: dict[str, str] = {"Cache-Control": "no-store"}
    if access is not None:
        headers.update(access.headers())
    return JSONResponse(status_code=status_code, content=payload, headers=headers)


def _error(
    message: str, status_code: int, access: Access | None = None
) -> JSONResponse:
    return _json({"error": message}, status_code, access)


def _retry_after(exc: RateLimitFault) -> str:
    raw = getattr(exc, "retry_after", None)
    try:
        seconds = int(raw) if raw is not None else DEFAULT_RETRY_AFTER
    except (TypeError, ValueError, OverflowError):
        seconds = DEFAULT_RETRY_AFTER
    return str(min(MAX_RETRY_AFTER, max(MIN_RETRY_AFTER, seconds)))


def _rate_limited(exc: RateLimitFault, access: Access) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"error": exc.message},
        headers={
            "Cache-Control": "no-store",
            "Retry-After": _retry_after(exc),
            **access.headers(),
        },
    )


@router.post("/api/operator")
async def post_operator_message(request: Request) -> Any:
    access = load_access(request)
    try:
        origin_error = verify_origin(request)
        if origin_error is not None:
            return _error(origin_error, 403, access)
        rate_limiter.check("operator", client_key(request, access.client_id))
        raw = await read_request_body(request, MAX_BODY_BYTES)
        body = parse_json_body(raw)
    except RateLimitFault as exc:
        return _rate_limited(exc, access)
    except PayloadTooLargeFault as exc:
        return _error(exc.message, exc.status_code, access)
    except RequestFault as exc:
        return _error(exc.message, exc.status_code, access)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("operator_request_failed")
        return _error("Belső kiszolgálói hiba.", 500, access)
    if not isinstance(body, dict):
        return _error("Hibás JSON.", 400, access)
    task_id = body.get("task_id")
    content = body.get("content")
    if not isinstance(task_id, str) or not is_loose_uuid(task_id):
        return _error("Hibás task_id.", 400, access)
    if not isinstance(content, str):
        return _error("Az üzenet csak szöveg lehet.", 400, access)
    message = content.strip()
    if not message:
        return _error("Üres üzenet.", 400, access)
    if len(message) > MAX_OPERATOR_MESSAGE_LENGTH:
        return _error("Az üzenet túl hosszú.", 400, access)
    try:
        database = database_module.get_admin_database()
        job = await job_access(database, task_id)
    except asyncio.CancelledError:
        raise
    except DatabaseError as exc:
        logger.warning("operator_job_lookup_failed", extra={"error": str(exc)})
        return _error("Az adatbázis nem érhető el.", 502, access)
    except Exception:
        logger.exception("operator_job_lookup_failed")
        return _error("Belső kiszolgálói hiba.", 500, access)
    if not job.exists:
        return _error("A feladat nem található.", 404, access)
    if not job.owned_by(access.client_id):
        return _error("Tiltott művelet.", 403, access)
    try:
        await cog.post_operator_message(task_id, message)
    except asyncio.CancelledError:
        raise
    except DatabaseError as exc:
        logger.warning(
            "operator_message_failed", extra={"task_id": task_id, "error": str(exc)}
        )
        return _error("Mentési hiba.", 502, access)
    except Exception:
        logger.exception("operator_message_failed", extra={"task_id": task_id})
        return _error("Mentési hiba.", 500, access)
    return _json({"accepted": True}, 200, access)
