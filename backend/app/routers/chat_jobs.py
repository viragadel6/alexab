from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app import database as database_module
from app.agent import cognitive as cog
from app.database import DatabaseError
from app.errors import PayloadTooLargeFault, RequestFault
from app.services.chat_jobs import follow_chat_job, supervisor_origin
from app.utils.access import Access, JobAccess, job_access, load_access
from app.utils.clock import as_str
from app.utils.security import is_loose_uuid, verify_origin
from app.utils.sse import NO_STORE_HEADERS, sse_frame
from app.utils.validation import parse_json_body, read_request_body

router = APIRouter()

logger = logging.getLogger("app.chat_jobs_router")

MAX_OPERATOR_MESSAGE_LENGTH = 32_000
MAX_BODY_BYTES = 262_144
CURSOR_MAX_DIGITS = 12
ACTIVE_STATUSES = ("queued", "running")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(
    payload: Any, status_code: int = 200, access: Access | None = None
) -> JSONResponse:
    headers: dict[str, str] = {"Cache-Control": "no-store"}
    if access is not None:
        headers.update(access.headers())
    return JSONResponse(status_code=status_code, content=payload, headers=headers)


def _error(
    message: str, status_code: int = 400, access: Access | None = None
) -> JSONResponse:
    return _json({"error": message}, status_code, access)


def _trusted_request_url() -> str:
    return supervisor_origin() or ""


def _parse_cursor(raw: Any) -> int | None:
    if raw is None:
        return 0
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return 0
    if len(text) > CURSOR_MAX_DIGITS or not text.isascii() or not text.isdigit():
        return None
    return int(text)


def _resolve_cursor(request: Request) -> tuple[int | None, str | None]:
    after_raw = request.query_params.get("after")
    if after_raw is not None:
        value = _parse_cursor(after_raw)
        if value is None:
            return None, "Érvénytelen eseménykurzor."
        return value, None
    header_raw = request.headers.get("last-event-id")
    if header_raw is not None:
        value = _parse_cursor(header_raw)
        if value is None:
            return None, "Érvénytelen eseménykurzor."
        return value, None
    return 0, None


def _frame_bytes(frame: Any) -> bytes:
    if isinstance(frame, bytes):
        return frame
    if isinstance(frame, str):
        return frame.encode("utf-8")
    try:
        return sse_frame(frame).encode("utf-8")
    except (TypeError, ValueError):
        return b""


async def _load_job_access(
    job_id: str,
) -> tuple[JobAccess | None, tuple[str, int] | None]:
    try:
        database = database_module.get_admin_database()
        job = await job_access(database, job_id)
    except asyncio.CancelledError:
        raise
    except DatabaseError as exc:
        logger.warning(
            "job_access_lookup_failed", extra={"job_id": job_id, "error": str(exc)}
        )
        return None, ("Az adatbázis nem érhető el.", 502)
    except Exception:
        logger.exception("job_access_lookup_failed", extra={"job_id": job_id})
        return None, ("Belső kiszolgálói hiba.", 500)
    return job, None


def _lookup_failure(
    failure: tuple[str, int] | None, access: Access
) -> JSONResponse:
    if failure is None:
        return _error("Belső kiszolgálói hiba.", 500, access)
    message, status_code = failure
    return _error(message, status_code, access)


@router.get("/api/chat/jobs/{job_id}/events")
async def get_job_events(job_id: str, request: Request) -> Any:
    access = load_access(request)
    if not is_loose_uuid(job_id):
        return _error("Érvénytelen munkaazonosító.", 400, access)
    cursor, cursor_error = _resolve_cursor(request)
    if cursor_error is not None:
        return _error(cursor_error, 400, access)
    job, failure = await _load_job_access(job_id)
    if failure is not None or job is None:
        return _lookup_failure(failure, access)
    if not job.exists:
        return _error("A munka nem található.", 404, access)
    if not job.owned_by(access.client_id):
        return _error("Tiltott művelet.", 403, access)
    start = cursor if cursor is not None else 0
    request_url = _trusted_request_url()

    async def stream() -> AsyncIterator[bytes]:
        try:
            async for frame in follow_chat_job(job_id, start, request_url):
                yield _frame_bytes(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("job_events_stream_failed", extra={"job_id": job_id})
            yield _frame_bytes(
                sse_frame(
                    {
                        "type": "error",
                        "task_id": job_id,
                        "code": "STREAM_FAILED",
                        "message": "A feladat eseményfolyama megszakadt.",
                    }
                )
            )

    headers = {
        **NO_STORE_HEADERS,
        "Content-Type": "text/event-stream; charset=utf-8",
        **access.headers(),
    }
    return StreamingResponse(stream(), status_code=200, headers=headers)


@router.post("/api/chat/jobs/{job_id}/stop")
async def stop_job(job_id: str, request: Request) -> Any:
    access = load_access(request)
    if not is_loose_uuid(job_id):
        return _error("Érvénytelen munkaazonosító.", 400, access)
    origin_error = verify_origin(request)
    if origin_error is not None:
        return _error(origin_error, 403, access)
    job, failure = await _load_job_access(job_id)
    if failure is not None or job is None:
        return _lookup_failure(failure, access)
    if not job.exists:
        return _error("A munka nem található.", 404, access)
    if not job.owned_by(access.client_id):
        return _error("Tiltott művelet.", 403, access)
    if job.status not in ACTIVE_STATUSES:
        return _error("A munka már lezárult.", 409, access)
    stamp = _now()
    try:
        database = database_module.get_admin_database()
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("job_stop_database_failed", extra={"job_id": job_id})
        return _error("Az adatbázis nem érhető el.", 502, access)
    try:
        if job.status == "queued":
            updated = await (
                database.table("chat_jobs")
                .update(
                    {
                        "status": "cancelled",
                        "stop_requested": True,
                        "finished_at": stamp,
                        "updated_at": stamp,
                    }
                )
                .eq("id", job_id)
                .eq("status", "queued")
                .execute()
            )
        else:
            updated = await (
                database.table("chat_jobs")
                .update({"stop_requested": True, "updated_at": stamp})
                .eq("id", job_id)
                .eq("status", "running")
                .execute()
            )
    except asyncio.CancelledError:
        raise
    except DatabaseError as exc:
        logger.warning(
            "job_stop_failed", extra={"job_id": job_id, "error": str(exc)}
        )
        return _error("Az adatbázis nem érhető el.", 502, access)
    except Exception:
        logger.exception("job_stop_failed", extra={"job_id": job_id})
        return _error("Belső kiszolgálói hiba.", 500, access)
    if not updated.rows:
        return _error("A munka állapota időközben megváltozott.", 409, access)
    return _json({"ok": True}, 200, access)


@router.post("/api/chat/jobs/{job_id}/redirect")
async def redirect_job(job_id: str, request: Request) -> Any:
    access = load_access(request)
    if not is_loose_uuid(job_id):
        return _error("Érvénytelen munkaazonosító.", 400, access)
    origin_error = verify_origin(request)
    if origin_error is not None:
        return _error(origin_error, 403, access)
    try:
        raw = await read_request_body(request, MAX_BODY_BYTES)
        body = parse_json_body(raw)
    except PayloadTooLargeFault as exc:
        return _error(exc.message, exc.status_code, access)
    except RequestFault as exc:
        return _error(exc.message, exc.status_code, access)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("job_redirect_body_failed", extra={"job_id": job_id})
        return _error("Érvénytelen kérés.", 400, access)
    if not isinstance(body, dict):
        return _error("Érvénytelen kérés.", 400, access)
    message = body.get("message")
    if not isinstance(message, str):
        return _error("Az üzenet csak szöveg lehet.", 400, access)
    text = message.strip()
    if not text:
        return _error("Üres üzenet nem küldhető.", 400, access)
    if len(text) > MAX_OPERATOR_MESSAGE_LENGTH:
        return _error("Az üzenet túl hosszú.", 400, access)
    job, failure = await _load_job_access(job_id)
    if failure is not None or job is None:
        return _lookup_failure(failure, access)
    if not job.exists:
        return _error("A munka nem található.", 404, access)
    if not job.owned_by(access.client_id):
        return _error("Tiltott művelet.", 403, access)
    if job.status not in ACTIVE_STATUSES:
        return _error("A munka már lezárult, új irányt nem kaphat.", 409, access)
    try:
        database = database_module.get_admin_database()
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("job_redirect_database_failed", extra={"job_id": job_id})
        return _error("Az adatbázis nem érhető el.", 502, access)
    try:
        claimed = await (
            database.table("chat_jobs")
            .update({"updated_at": _now()})
            .eq("id", job_id)
            .in_("status", list(ACTIVE_STATUSES))
            .execute()
        )
    except asyncio.CancelledError:
        raise
    except DatabaseError as exc:
        logger.warning(
            "job_redirect_claim_failed", extra={"job_id": job_id, "error": str(exc)}
        )
        return _error("Az adatbázis nem érhető el.", 502, access)
    except Exception:
        logger.exception("job_redirect_claim_failed", extra={"job_id": job_id})
        return _error("Belső kiszolgálói hiba.", 500, access)
    if not claimed.rows:
        return _error("A munka már lezárult, új irányt nem kaphat.", 409, access)
    try:
        await cog.post_operator_message(job_id, text)
    except asyncio.CancelledError:
        raise
    except DatabaseError as exc:
        logger.warning(
            "job_redirect_failed", extra={"job_id": job_id, "error": str(exc)}
        )
        return _error("Az üzenet elküldése nem sikerült.", 502, access)
    except Exception:
        logger.exception("job_redirect_failed", extra={"job_id": job_id})
        return _error("Az üzenet elküldése nem sikerült.", 500, access)
    try:
        verification = await (
            database.table("chat_jobs")
            .select("status")
            .eq("id", job_id)
            .limit(1)
            .execute()
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("job_redirect_verify_failed", extra={"job_id": job_id})
        return _json({"ok": True}, 200, access)
    row = verification.maybe_single()
    if row is not None and as_str(row.get("status")) not in ACTIVE_STATUSES:
        return _error("A munka az üzenet küldése közben lezárult.", 409, access)
    return _json({"ok": True}, 200, access)
