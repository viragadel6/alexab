from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app import database as database_module
from app.database import DatabaseError
from app.errors import PayloadTooLargeFault, RateLimitFault, RequestFault
from app.services.chat_jobs import (
    follow_chat_job,
    schedule_chat_job_dispatch,
    supervisor_origin,
)
from app.utils.access import Access, load_access
from app.utils.clock import as_str
from app.utils.rate_limit import client_key, rate_limiter
from app.utils.security import verify_origin
from app.utils.sse import NO_STORE_HEADERS, sse_frame
from app.utils.validation import (
    parse_conversation_id,
    parse_json_body,
    read_request_body,
    validate_client_history,
)

router = APIRouter()

logger = logging.getLogger("app.chat")

MAX_REQUEST_BYTES = 1_048_576
CONTENT_TYPE_PATTERN = re.compile(r"^application/json(?:\s*;|\s*$)", re.IGNORECASE)
DEFAULT_RETRY_AFTER = 60
MIN_RETRY_AFTER = 1
MAX_RETRY_AFTER = 86_400
TITLE_MAX_LENGTH = 80
DEFAULT_TITLE = "Beszélgetés"


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
    message: str,
    status_code: int,
    code: str,
    access: Access | None = None,
) -> JSONResponse:
    return _json(
        {"error": {"code": code, "message": message}}, status_code, access
    )


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
        content={"error": exc.to_payload()},
        headers={
            "Cache-Control": "no-store",
            "Retry-After": _retry_after(exc),
            **access.headers(),
        },
    )


def _trusted_request_url() -> str:
    return supervisor_origin() or ""


def _frame_bytes(frame: Any) -> bytes:
    if isinstance(frame, bytes):
        return frame
    if isinstance(frame, str):
        return frame.encode("utf-8")
    try:
        return sse_frame(frame).encode("utf-8")
    except (TypeError, ValueError):
        return b""


def _derive_title(history: list[dict[str, Any]]) -> str:
    for message in history:
        if message.get("role") != "user":
            continue
        collapsed = " ".join(str(message.get("content") or "").split())
        if collapsed:
            return collapsed[:TITLE_MAX_LENGTH]
    return DEFAULT_TITLE


def _normalize_messages(value: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if not isinstance(value, list):
        return normalized
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _is_prefix(
    prefix: list[dict[str, str]], values: list[dict[str, str]]
) -> bool:
    if len(prefix) > len(values):
        return False
    for index, expected in enumerate(prefix):
        current = values[index]
        if current["role"] != expected["role"] or current["content"] != expected["content"]:
            return False
    return True


def _merge_history(stored: Any, incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stored_messages = _normalize_messages(stored)
    incoming_messages = _normalize_messages(incoming)
    if not stored_messages:
        return incoming_messages
    if not incoming_messages:
        return stored_messages
    if _is_prefix(stored_messages, incoming_messages):
        return incoming_messages
    if _is_prefix(incoming_messages, stored_messages):
        return stored_messages
    logger.warning(
        "chat_history_diverged",
        extra={"stored": len(stored_messages), "incoming": len(incoming_messages)},
    )
    return stored_messages


@router.post("/api/chat")
async def post_chat(request: Request) -> Any:
    access = load_access(request)
    try:
        origin_error = verify_origin(request)
        if origin_error is not None:
            return _error(origin_error, 403, "INVALID_ORIGIN", access)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("chat_origin_check_failed")
        return _error("Belső kiszolgálói hiba.", 500, "INTERNAL_ERROR", access)
    try:
        rate_limiter.check("chat", client_key(request, access.client_id))
    except RateLimitFault as exc:
        return _rate_limited(exc, access)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("chat_rate_limit_failed")
        return _error("Belső kiszolgálói hiba.", 500, "INTERNAL_ERROR", access)
    try:
        content_type = request.headers.get("content-type", "")
        if not CONTENT_TYPE_PATTERN.match(content_type):
            raise RequestFault(
                "application/json szükséges.",
                code="UNSUPPORTED_MEDIA_TYPE",
                status_code=415,
            )
        raw = await read_request_body(request, MAX_REQUEST_BYTES)
        body = parse_json_body(raw)
        history = validate_client_history(body)
    except PayloadTooLargeFault as exc:
        return _error(exc.message, exc.status_code, exc.code, access)
    except RequestFault as exc:
        return _error(exc.message, exc.status_code, exc.code, access)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("chat_body_failed")
        return _error("Belső kiszolgálói hiba.", 500, "INTERNAL_ERROR", access)
    payload = body if isinstance(body, dict) else {}
    conversation_id = parse_conversation_id(payload.get("conversation_id")) or str(
        uuid.uuid4()
    )
    job_id = str(uuid.uuid4())
    try:
        database = database_module.get_admin_database()
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("chat_database_failed")
        return _error("Az adatbázis nem érhető el.", 502, "DATABASE_ERROR", access)
    try:
        existing_response = await (
            database.table("chat_conversations")
            .select("id, title, messages, client_id")
            .eq("id", conversation_id)
            .limit(1)
            .execute()
        )
    except asyncio.CancelledError:
        raise
    except DatabaseError as exc:
        logger.warning("chat_conversation_read_failed", extra={"error": str(exc)})
        return _error("Az adatbázis nem érhető el.", 502, "DATABASE_ERROR", access)
    except Exception:
        logger.exception("chat_conversation_read_failed")
        return _error("Belső kiszolgálói hiba.", 500, "INTERNAL_ERROR", access)
    existing = existing_response.maybe_single()
    stored_messages: Any = []
    stored_title = ""
    if existing is not None:
        owner = as_str(existing.get("client_id"))
        if owner and owner != access.client_id:
            return _error("Tiltott művelet.", 403, "FORBIDDEN", access)
        stored_messages = existing.get("messages")
        stored_title = as_str(existing.get("title")).strip()
    merged_history = _merge_history(stored_messages, history)
    title = stored_title or _derive_title(history)
    try:
        await database.rpc(
            "create_chat_job",
            {
                "_conversation_id": conversation_id,
                "_job_id": job_id,
                "_client_id": access.client_id,
                "_title": title,
                "_messages": merged_history,
                "_history": merged_history,
            },
        ).execute()
    except asyncio.CancelledError:
        raise
    except DatabaseError as exc:
        if exc.status_code == 403 or str(exc.code or "").strip().upper() == "42501":
            return _error("Tiltott művelet.", 403, "FORBIDDEN", access)
        logger.warning(
            "chat_job_create_failed",
            extra={"job_id": job_id, "error": str(exc)},
        )
        return _error("Az adatbázis nem érhető el.", 502, "DATABASE_ERROR", access)
    except Exception:
        logger.exception("chat_job_create_failed", extra={"job_id": job_id})
        return _error("Belső kiszolgálói hiba.", 500, "INTERNAL_ERROR", access)
    schedule_chat_job_dispatch(job_id, _trusted_request_url())

    async def stream() -> AsyncIterator[bytes]:
        meta = json.dumps(
            {"type": "meta", "conversation_id": conversation_id, "job_id": job_id},
            ensure_ascii=False,
        )
        yield f"data: {meta}\n\n".encode("utf-8")
        try:
            async for frame in follow_chat_job(job_id, 0, _trusted_request_url()):
                yield _frame_bytes(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("chat_stream_failed", extra={"job_id": job_id})
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
        "X-Chat-Job-Id": job_id,
        **access.headers(),
    }
    return StreamingResponse(stream(), status_code=200, headers=headers)
