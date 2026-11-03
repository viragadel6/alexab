from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app import database as database_module
from app.database import DatabaseError, SupabaseDatabase
from app.errors import RateLimitFault
from app.utils.access import Access, load_access
from app.utils.clock import as_int, as_str
from app.utils.rate_limit import client_key, rate_limiter
from app.utils.security import is_loose_uuid

router = APIRouter()

logger = logging.getLogger("app.conversations")

MAX_CONVERSATION_IDS = 100
MAX_JOB_ROWS_PER_CONVERSATION = 20
MAX_TOTAL_JOB_ROWS = 500
MAX_PARALLEL_EVENT_QUERIES = 8
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


def _parse_ids(raw: str | None) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for value in (raw or "").split(","):
        trimmed = value.strip()
        if not trimmed or trimmed in seen:
            continue
        if not is_loose_uuid(trimmed):
            continue
        seen.add(trimmed)
        selected.append(trimmed)
        if len(selected) >= MAX_CONVERSATION_IDS:
            break
    return selected


async def _last_event_ids(
    database: SupabaseDatabase, job_ids: list[str]
) -> dict[str, int]:
    if not job_ids:
        return {}
    semaphore = asyncio.Semaphore(MAX_PARALLEL_EVENT_QUERIES)
    results: dict[str, int] = {}

    async def load(job_id: str) -> None:
        async with semaphore:
            response = await (
                database.table("chat_job_events")
                .select("id")
                .eq("job_id", job_id)
                .order("id", desc=True)
                .limit(1)
                .execute()
            )
            row = response.maybe_single()
            results[job_id] = as_int(row.get("id")) if row is not None else 0

    await asyncio.gather(*(load(job_id) for job_id in job_ids))
    return results


@router.get("/api/conversations")
async def get_conversations(request: Request) -> Any:
    access = load_access(request)
    try:
        rate_limiter.check("conversations", client_key(request, access.client_id))
    except RateLimitFault as exc:
        return _rate_limited(exc, access)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("conversations_rate_limit_failed")
        return _error("Belső kiszolgálói hiba.", 500, access)
    ids = _parse_ids(request.query_params.get("ids"))
    if not ids:
        return _json({"conversations": []}, 200, access)
    try:
        database = database_module.get_admin_database()
        conversations_response = await (
            database.table("chat_conversations")
            .select("id", "title", "messages", "created_at", "updated_at")
            .in_("id", ids)
            .eq("client_id", access.client_id)
            .order("created_at", desc=False, nullsfirst=True)
            .order("id", desc=False)
            .limit(MAX_CONVERSATION_IDS)
            .execute()
        )
        conversations = list(conversations_response.rows)
        owned_ids = [
            as_str(row.get("id")) for row in conversations if as_str(row.get("id"))
        ]
        latest_by_conversation: dict[str, dict[str, Any]] = {}
        last_event_by_job: dict[str, int] = {}
        if owned_ids:
            jobs_limit = min(
                MAX_TOTAL_JOB_ROWS, MAX_JOB_ROWS_PER_CONVERSATION * len(owned_ids)
            )
            jobs_response = await (
                database.table("chat_jobs")
                .select(
                    "id",
                    "conversation_id",
                    "status",
                    "response_content",
                    "response_reasoning",
                    "error",
                    "created_at",
                    "updated_at",
                )
                .in_("conversation_id", owned_ids)
                .order("created_at", desc=True)
                .order("id", desc=True)
                .limit(jobs_limit)
                .execute()
            )
            for job in jobs_response.rows:
                conversation_id = as_str(job.get("conversation_id"))
                if conversation_id and conversation_id not in latest_by_conversation:
                    latest_by_conversation[conversation_id] = job
            job_ids = [
                job_id
                for job_id in (
                    as_str(job.get("id")) for job in latest_by_conversation.values()
                )
                if job_id
            ]
            last_event_by_job = await _last_event_ids(database, job_ids)
        result: list[dict[str, Any]] = []
        for conversation in conversations:
            conversation_id = as_str(conversation.get("id"))
            job = latest_by_conversation.get(conversation_id)
            job_id = as_str(job.get("id")) if job is not None else ""
            result.append(
                {
                    "conversation_id": conversation_id,
                    "title": as_str(conversation.get("title")),
                    "messages": conversation.get("messages"),
                    "created_at": conversation.get("created_at"),
                    "updated_at": conversation.get("updated_at"),
                    "latest_job": (
                        {
                            "job_id": job_id,
                            "conversation_id": as_str(job.get("conversation_id")),
                            "status": as_str(job.get("status")),
                            "response_content": job.get("response_content"),
                            "response_reasoning": job.get("response_reasoning"),
                            "error": job.get("error"),
                            "last_event_id": last_event_by_job.get(job_id, 0),
                        }
                        if job is not None
                        else None
                    ),
                }
            )
        return _json(jsonable_encoder({"conversations": result}), 200, access)
    except asyncio.CancelledError:
        raise
    except DatabaseError as exc:
        logger.warning("conversations_database_failed", extra={"error": str(exc)})
        return _error("Az adatbázis nem érhető el.", 502, access)
    except Exception:
        logger.exception("conversations_failed")
        return _error("Belső kiszolgálói hiba.", 500, access)
