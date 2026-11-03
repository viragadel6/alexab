from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app import database as database_module
from app.database import DatabaseError
from app.errors import AgentFault, RateLimitFault, RequestFault
from app.services.chat_jobs import job_queue_summary
from app.services.checkpoints import checkpoint_stats
from app.services.supervisor import RECOVERY_TABLE, heal_now, supervisor_status
from app.utils.access import admin_error
from app.utils.clock import as_int, as_list, as_str
from app.utils.rate_limit import client_key, rate_limiter
from app.utils.security import is_loose_uuid, verify_origin

router = APIRouter()

logger = logging.getLogger("app.system")

NO_STORE = {"Cache-Control": "no-store"}
ACTIVE_STATUSES = ("queued", "running")
EXTENDED_JOB_COLUMNS = (
    "id, status, created_at, updated_at, started_at, lease_expires_at, stop_requested, "
    "attempts, next_attempt_at, last_error, progress_at"
)
LEGACY_JOB_COLUMNS = (
    "id, status, created_at, updated_at, started_at, lease_expires_at, stop_requested"
)
RECOVERY_COLUMNS = "id, job_id, kind, owner_id, detail, created_at"
MISSING_COLUMN_CODES = frozenset({"42703", "PGRST204"})
MISSING_TABLE_CODES = frozenset({"42P01", "PGRST106", "PGRST205"})
MAX_JOB_ROWS = 50
MAX_RECOVERY_ROWS = 50
DEFAULT_JOB_ROWS = 25
DEFAULT_RECOVERY_ROWS = 20
DETAIL_CHARS = 400
DEFAULT_RETRY_AFTER = 60
MIN_RETRY_AFTER = 1
MAX_RETRY_AFTER = 86_400


def _json(payload: dict[str, Any], status_code: int = 200) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=payload, headers=NO_STORE)


def _failure(
    message: str, status_code: int = 400, code: str = "SYSTEM_REQUEST_FAILED"
) -> JSONResponse:
    return _json({"error": {"code": code, "message": message}}, status_code)


def _retry_after(exc: RateLimitFault) -> str:
    raw = getattr(exc, "retry_after", None)
    seconds = DEFAULT_RETRY_AFTER
    if raw is not None:
        try:
            val = float(raw)
            if val != val or val == float("inf") or val == float("-inf"):
                seconds = DEFAULT_RETRY_AFTER
            else:
                seconds = int(val)
                if val > seconds:
                    seconds += 1
        except (TypeError, ValueError, OverflowError):
            seconds = DEFAULT_RETRY_AFTER
    return str(min(MAX_RETRY_AFTER, max(MIN_RETRY_AFTER, seconds)))


def _rate_limit_failure(exc: RateLimitFault) -> JSONResponse:
    code = getattr(exc, "code", "RATE_LIMITED")
    message = getattr(exc, "message", "Túl sok kérés.")
    return JSONResponse(
        status_code=429,
        content={"error": {"code": code, "message": message}},
        headers={**NO_STORE, "Retry-After": _retry_after(exc)},
    )


def _admin_denied(request: Request) -> JSONResponse | None:
    error = admin_error(request)
    if error is None:
        return None
    if isinstance(error, tuple) and len(error) == 2:
        message, status_code = error
    else:
        message, status_code = "Hozzáférés megtagadva.", 403
    return _failure(str(message), int(status_code), "ADMIN_ACCESS_DENIED")


def _error_code(exc: DatabaseError) -> str:
    return str(getattr(exc, "code", None) or "").strip().upper()


def _error_message(exc: DatabaseError) -> str:
    msg = getattr(exc, "message", None)
    if not isinstance(msg, str) or not msg.strip():
        msg = str(exc)
    return msg.strip().lower()


def _is_missing_column_error(exc: DatabaseError) -> bool:
    if _error_code(exc) in MISSING_COLUMN_CODES:
        return True
    message = _error_message(exc)
    return "column" in message and (
        "does not exist" in message or "schema cache" in message or "not find" in message
    )


def _is_missing_table_error(exc: DatabaseError) -> bool:
    if _error_code(exc) in MISSING_TABLE_CODES:
        return True
    message = _error_message(exc)
    if "could not find the table" in message:
        return True
    return "relation" in message and "does not exist" in message


def _admin_database() -> Any:
    try:
        db = database_module.get_admin_database()
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError("Az adatbázis nem érhető el.", 502) from exc
    if db is None:
        raise DatabaseError("Az adatbázis nem érhető el.", 502)
    return db


def _database_rows(response: Any, message: str) -> list[dict[str, Any]]:
    if isinstance(response, dict):
        rows = response.get("rows")
        if rows is None:
            rows = response.get("data")
    else:
        rows = getattr(response, "rows", None)
        if rows is None:
            rows = getattr(response, "data", None)
    if rows is None or not isinstance(rows, (list, tuple)):
        raise DatabaseError(message, 502)
    sanitized: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            sanitized.append(row)
        elif hasattr(row, "model_dump") and callable(row.model_dump):
            sanitized.append(row.model_dump())
        elif hasattr(row, "dict") and callable(row.dict):
            sanitized.append(row.dict())
        elif hasattr(row, "items") and callable(row.items):
            sanitized.append(dict(row.items()))
        else:
            raise DatabaseError(message, 502)
    return sanitized


def _bool(val: Any) -> bool:
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "t", "yes", "on")
    return bool(val)


def _row_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": as_str(record.get("id")),
        "status": as_str(record.get("status")),
        "created_at": as_str(record.get("created_at")),
        "updated_at": as_str(record.get("updated_at")),
        "started_at": as_str(record.get("started_at")),
        "lease_expires_at": as_str(record.get("lease_expires_at")),
        "progress_at": as_str(record.get("progress_at")),
        "next_attempt_at": as_str(record.get("next_attempt_at")),
        "attempts": as_int(record.get("attempts")),
        "last_error": as_str(record.get("last_error"), DETAIL_CHARS),
        "stop_requested": _bool(record.get("stop_requested")),
    }


def _recovery_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": as_int(record.get("id")),
        "job_id": as_str(record.get("job_id")),
        "kind": as_str(record.get("kind")),
        "owner_id": as_str(record.get("owner_id")),
        "detail": record.get("detail"),
        "created_at": as_str(record.get("created_at")),
    }


async def _recoverable_jobs(limit: int, job_id: str | None) -> list[dict[str, Any]]:
    last_error: DatabaseError | None = None
    for columns in (EXTENDED_JOB_COLUMNS, LEGACY_JOB_COLUMNS):
        try:
            database = _admin_database()
            query = database.table("chat_jobs").select(columns).in_(
                "status", list(ACTIVE_STATUSES)
            )
            if job_id:
                query = query.eq("id", job_id)
            response = await (
                query.order("updated_at", desc=False, nullsfirst=True).limit(limit).execute()
            )
            rows = _database_rows(response, "A feladatok válasza hibás.")
            return [_row_summary(row) for row in rows]
        except DatabaseError as exc:
            if (
                columns == EXTENDED_JOB_COLUMNS
                and _is_missing_column_error(exc)
                and not _is_missing_table_error(exc)
            ):
                last_error = exc
                logger.warning("system_jobs_legacy_columns", extra={"error": str(exc)})
                continue
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise DatabaseError("A feladatok lekérdezése nem sikerült.", 502) from exc
    if last_error is not None:
        raise last_error
    raise DatabaseError("A feladatok lekérdezése nem sikerült.", 502)


async def _recovery_rows(limit: int, job_id: str | None) -> list[dict[str, Any]]:
    try:
        database = _admin_database()
        query = database.table(RECOVERY_TABLE).select(RECOVERY_COLUMNS)
        if job_id:
            query = query.eq("job_id", job_id)
        response = await query.order("id", desc=True).limit(limit).execute()
        rows = _database_rows(response, "A helyreállítási napló válasza hibás.")
        return [_recovery_summary(row) for row in rows]
    except DatabaseError as exc:
        if _is_missing_table_error(exc) or _is_missing_column_error(exc):
            logger.warning("system_recovery_log_unavailable", extra={"error": str(exc)})
            return []
        raise
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise DatabaseError("A helyreállítási napló lekérdezése nem sikerült.", 502) from exc


async def _queue_summary() -> dict[str, Any]:
    try:
        summary = await job_queue_summary()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("system_queue_failed", extra={"error": str(exc)})
        return {"available": False}
    if not isinstance(summary, dict):
        return {"available": False}
    return dict(summary)


def _checkpoint_summary() -> dict[str, Any]:
    try:
        stats = checkpoint_stats()
    except Exception as exc:
        logger.warning("system_checkpoints_failed", extra={"error": str(exc)})
        return {"available": False}
    if not isinstance(stats, dict):
        return {"available": False}
    return dict(stats)


def _positive_limit(raw: str | None, fallback: int, ceiling: int) -> int:
    return as_int(raw if raw is not None else "", fallback, 1, ceiling)


@router.get("/api/system/status")
async def get_system_status(request: Request) -> Any:
    try:
        denied = _admin_denied(request)
        if denied is not None:
            return denied
        rate_limiter.check("system", client_key(request))
        status = supervisor_status()
        payload: dict[str, Any] = dict(status) if isinstance(status, dict) else {}
        payload["queue"] = await _queue_summary()
        payload["checkpoints"] = _checkpoint_summary()
        payload["recovery_log"] = await _recovery_rows(DEFAULT_RECOVERY_ROWS, None)
        return _json(payload)
    except RateLimitFault as exc:
        return _rate_limit_failure(exc)
    except RequestFault as exc:
        return _failure(exc.message, exc.status_code, exc.code)
    except AgentFault as exc:
        return _failure(exc.message, exc.status_code, exc.code)
    except DatabaseError as exc:
        logger.warning("system_status_database_failed", extra={"error": str(exc)})
        return _failure("Az adatbázis nem érhető el.", 502, "DATABASE_ERROR")
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("system_status_failed")
        return _failure("Belső szerverhiba.", 500, "INTERNAL_ERROR")


@router.get("/api/system/jobs")
async def get_system_jobs(request: Request) -> Any:
    try:
        denied = _admin_denied(request)
        if denied is not None:
            return denied
        rate_limiter.check("system", client_key(request))
        limit = _positive_limit(
            request.query_params.get("limit"), DEFAULT_JOB_ROWS, MAX_JOB_ROWS
        )
        recovery_limit = _positive_limit(
            request.query_params.get("recovery_limit"),
            DEFAULT_RECOVERY_ROWS,
            MAX_RECOVERY_ROWS,
        )
        raw_job_id = request.query_params.get("job_id") or ""
        job_id = raw_job_id.strip()
        if job_id and not is_loose_uuid(job_id):
            raise RequestFault("Érvénytelen feladat azonosító.", code="INVALID_JOB_ID")
        normalized_job_id = job_id or None
        jobs = await _recoverable_jobs(limit, normalized_job_id)
        recovery = await _recovery_rows(recovery_limit, normalized_job_id)
        return _json({"count": len(jobs), "jobs": jobs, "recovery_log": recovery})
    except RateLimitFault as exc:
        return _rate_limit_failure(exc)
    except RequestFault as exc:
        return _failure(exc.message, exc.status_code, exc.code)
    except AgentFault as exc:
        return _failure(exc.message, exc.status_code, exc.code)
    except DatabaseError as exc:
        logger.warning("system_jobs_database_failed", extra={"error": str(exc)})
        return _failure("Az adatbázis nem érhető el.", 502, "DATABASE_ERROR")
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("system_jobs_failed")
        return _failure("Belső szerverhiba.", 500, "INTERNAL_ERROR")


@router.post("/api/system/heal")
async def post_system_heal(request: Request) -> Any:
    try:
        denied = _admin_denied(request)
        if denied is not None:
            return denied
        origin_error = verify_origin(request)
        if origin_error is not None:
            return _failure(origin_error, 403, "INVALID_ORIGIN")
        rate_limiter.check("heal", client_key(request))
        result = await heal_now()
        payload = result if isinstance(result, dict) else {}
        errors = as_list(payload.get("errors"))
        if "healed" in payload:
            healed = _bool(payload.get("healed")) and not errors
        elif "ok" in payload:
            healed = _bool(payload.get("ok")) and not errors
        else:
            healed = not errors
        return _json({"healed": healed, "result": payload})
    except RateLimitFault as exc:
        return _rate_limit_failure(exc)
    except RequestFault as exc:
        return _failure(exc.message, exc.status_code, exc.code)
    except AgentFault as exc:
        return _failure(exc.message, exc.status_code, exc.code)
    except DatabaseError as exc:
        logger.warning("system_heal_database_failed", extra={"error": str(exc)})
        return _failure("Az adatbázis nem érhető el.", 502, "DATABASE_ERROR")
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("system_heal_failed")
        return _failure("Belső szerverhiba.", 500, "INTERNAL_ERROR")