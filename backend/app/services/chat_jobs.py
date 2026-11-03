from __future__ import annotations

import asyncio
import logging
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Awaitable
from urllib.parse import urlparse

from app import database as database_module
from app.agent.runtime import run_durable_chat
from app.config import get_settings
from app.database import DatabaseError, SupabaseDatabase
from app.errors import AgentFault
from app.models import TerminalState
from app.services import checkpoints
from app.utils.clock import age_ms, as_bool, as_int, as_list, as_str, iso, now, parse_iso
from app.utils.security import worker_origin
from app.utils.sse import sse_comment, sse_frame

EVENTS_TABLE = "chat_job_events"
JOBS_TABLE = "chat_jobs"
LEASE_MS = 90_000
LEASE_SAFE_MS = 60_000
HEARTBEAT_MS = 20_000
HEARTBEAT_FAILURE_LIMIT = 3
HEARTBEAT_RETRY_MAX = 2
HEARTBEAT_CALL_TIMEOUT_SECONDS = 10.0
FOLLOW_POLL_MS = 800
FOLLOW_POLL_MAX_MS = 3_000
FOLLOW_STATUS_CHECK_MS = 2_500
FOLLOW_HEARTBEAT_MS = 15_000
FOLLOW_MAX_MS = 21_600_000
FOLLOW_REVIVE_MS = 30_000
FOLLOW_TERMINAL_GRACE_MS = 30_000
FOLLOW_FAILURE_LIMIT = 8
FOLLOW_RETRY_BASE_SECONDS = 0.5
FOLLOW_RETRY_MAX_SECONDS = 6.0
TERMINAL_STATUSES = ("completed", "cancelled", "failed")
ACTIVE_STATUSES = ("queued", "running")
HELD_KINDS = ("terminal", "done", "error")
SINK_RETRY_BASE_SECONDS = 0.4
SINK_RETRY_MAX_SECONDS = 3.0
SINK_MAX_BLOCK_SECONDS = 6.0
SINK_FLUSH_EVERY = 10
SINK_MIN_FLUSH_CHARS = 200
SINK_PROGRESS_INTERVAL_SECONDS = 1.5
FINALIZE_RETRY_MAX = 4
ERROR_MESSAGE_CHARS = 900
EXTENDED_RECHECK_SECONDS = 300.0
DEDUPE_LOOKBACK = 50
INTERRUPT_REDISPATCH_SECONDS = 5.0
RETRY_DISPATCH_MARGIN_SECONDS = 0.25
GENERIC_FAILURE_MESSAGE = "A háttérmunka hibával leállt."
LEGACY_COLUMNS = "id, history, status, lease_expires_at, stop_requested, created_at, updated_at"
EXTENDED_COLUMNS = (
    "id, history, status, lease_expires_at, stop_requested, created_at, updated_at, "
    "attempts, next_attempt_at, last_error, progress_at"
)

logger = logging.getLogger("app.chat_jobs")

_background_tasks: set[asyncio.Task[Any]] = set()
_extended_columns: bool | None = None
_extended_checked_at: float = 0.0
_extended_lock = asyncio.Lock()


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _stamp(value: datetime) -> str:
    return _utc(value).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _lease_deadline() -> datetime:
    return datetime.now(timezone.utc) + timedelta(milliseconds=LEASE_MS)


class LeaseToken:
    def __init__(self, lease: datetime):
        self._values: list[datetime] = [_utc(lease)]

    def begin(self, lease: datetime) -> None:
        candidate = _utc(lease)
        if candidate not in self._values:
            self._values.append(candidate)

    def confirm(self, lease: datetime) -> None:
        self._values = [_utc(lease)]

    def stamps(self) -> list[str]:
        return [_stamp(value) for value in self._values]

    def matches(self, value: Any) -> bool:
        parsed = parse_iso(value)
        if parsed is None:
            return False
        normalized = _utc(parsed)
        return any(normalized == candidate for candidate in self._values)


class RunState:
    def __init__(self) -> None:
        self.stop_requested = False
        self.lost = False
        self.interrupted = False


def _retain(coroutine: Awaitable[Any]) -> asyncio.Task[Any]:
    task: asyncio.Task[Any] = asyncio.ensure_future(coroutine)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _reap(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.warning("background_job_error", extra={"error": str(error)})


def supervisor_origin() -> str | None:
    settings = get_settings()
    if not settings.worker_token:
        return None
    candidates: list[str] = []
    for raw in (settings.app_origins, settings.app_origin, settings.published_origin):
        for entry in (raw or "").split(","):
            trimmed = entry.strip()
            if not trimmed:
                continue
            parsed = urlparse(trimmed)
            if parsed.scheme in ("http", "https") and parsed.netloc:
                candidates.append(f"{parsed.scheme}://{parsed.netloc}")
    return candidates[0] if candidates else None


async def columns_available(database: SupabaseDatabase) -> bool:
    global _extended_columns, _extended_checked_at
    loop = asyncio.get_running_loop()
    if _extended_columns is True:
        return True
    if _extended_columns is False and loop.time() - _extended_checked_at < EXTENDED_RECHECK_SECONDS:
        return False
    async with _extended_lock:
        if _extended_columns is True:
            return True
        if (
            _extended_columns is False
            and loop.time() - _extended_checked_at < EXTENDED_RECHECK_SECONDS
        ):
            return False
        try:
            await (
                database.table(JOBS_TABLE)
                .select("id, attempts, next_attempt_at, last_error, progress_at")
                .limit(1)
                .execute()
            )
        except DatabaseError as exc:
            if exc.status_code in (400, 404):
                _extended_columns = False
                _extended_checked_at = loop.time()
            return False
        except Exception:
            return False
        _extended_columns = True
        return True


async def dispatch_chat_job(job_id: str, request_url: str) -> None:
    await dispatch_chat_job_origin(job_id, worker_origin(request_url))


async def _dispatch_chat_job_in_background(job_id: str, request_url: str) -> None:
    try:
        await dispatch_chat_job(job_id, request_url)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("chat_dispatch_background_failed", extra={"job_id": job_id})
        _execute_later(job_id)


def schedule_chat_job_dispatch(job_id: str, request_url: str) -> asyncio.Task[None]:
    return _retain(_dispatch_chat_job_in_background(job_id, request_url))


async def dispatch_chat_job_origin(job_id: str, origin: str | None) -> None:
    if origin is None:
        _execute_later(job_id)
        return
    database = database_module.get_admin_database()
    try:
        await database.rpc("kick_chat_worker", {"_base_url": origin, "_job_id": job_id}).execute()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "worker_kick_failed", extra={"job_id": job_id, "origin": origin, "error": str(exc)}
        )
        _execute_later(job_id)


async def _dispatch_after(job_id: str, origin: str | None, delay: float) -> None:
    try:
        await asyncio.sleep(max(0.0, delay))
        await dispatch_chat_job_origin(job_id, origin)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("delayed_dispatch_failed", extra={"job_id": job_id})


def _schedule_dispatch_after(job_id: str, origin: str | None, delay: float) -> asyncio.Task[Any]:
    task = _retain(_dispatch_after(job_id, origin, delay))
    task.add_done_callback(_reap)
    return task


def _execute_later(job_id: str) -> None:
    task = _retain(_execute_local_job(job_id))
    task.add_done_callback(_reap)


async def _execute_local_job(job_id: str) -> None:
    try:
        await execute_chat_job(job_id)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("local_job_failed", extra={"job_id": job_id, "error": str(exc)})


async def revive_stale_chat_job(job_id: str, request_url: str) -> bool:
    return await revive_stale_chat_job_origin(job_id, worker_origin(request_url))


async def revive_stale_chat_job_origin(job_id: str, origin: str | None) -> bool:
    database = database_module.get_admin_database()
    extended = await columns_available(database)
    columns = EXTENDED_COLUMNS if extended else LEGACY_COLUMNS
    response = await (
        database.table(JOBS_TABLE).select(columns).eq("id", job_id).limit(1).execute()
    )
    row = response.maybe_single()
    if row is None:
        return False
    if as_bool(row.get("stop_requested")):
        return False
    status = as_str(row.get("status"))
    if status not in ACTIVE_STATUSES:
        return False
    lease = parse_iso(row.get("lease_expires_at"))
    if extended:
        scheduled = parse_iso(row.get("next_attempt_at"))
        if scheduled is not None and scheduled > now():
            return False
    stale = status == "queued" or (status == "running" and (lease is None or lease < now()))
    if not stale:
        return False
    await dispatch_chat_job_origin(job_id, origin)
    return True


async def _event_present(
    database: SupabaseDatabase, job_id: str, run: Any, sequence: Any
) -> bool:
    if run is None or sequence is None:
        return False
    try:
        response = await (
            database.table(EVENTS_TABLE)
            .select("id", "payload")
            .eq("job_id", job_id)
            .order("id", desc=True)
            .limit(DEDUPE_LOOKBACK)
            .execute()
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        return False
    for row in response.rows:
        payload = row.get("payload")
        if (
            isinstance(payload, dict)
            and payload.get("event_run") == run
            and payload.get("event_seq") == sequence
        ):
            return True
    return False


class DurableEventSink:
    def __init__(
        self, job_id: str, database: SupabaseDatabase, lease: LeaseToken | None = None
    ):
        self.job_id = job_id
        self.database = database
        self.lease = lease
        self.content = ""
        self.reasoning = ""
        self.final_status = "failed"
        self.written = 0
        self.buffered = 0
        self.dropped = 0
        self.failures = 0
        self.closed = False
        self._buffer: list[dict[str, Any]] = []
        self._held: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._delta_counter = 0
        self._done_received = False
        self._progress_columns: bool | None = None
        self._run = uuid.uuid4().hex[:12]
        self._sequence = 0
        self._last_progress: float | None = None
        self.hold_terminal = False

    def close(self) -> None:
        self.closed = True

    async def send(self, event: dict[str, Any]) -> None:
        payload = dict(event) if isinstance(event, dict) else {"type": "unknown", "payload": event}
        kind = as_str(payload.get("type"))
        if kind == "delta":
            self.content += as_str(payload.get("content"))
            self.reasoning += as_str(payload.get("reasoning_content"))
        elif kind == "done":
            self._done_received = True
            outcome = as_str(payload.get("outcome")) or "failed"
            if outcome == "completed":
                self.final_status = "completed"
            elif outcome == "cancelled":
                self.final_status = "cancelled"
            else:
                self.final_status = "failed"
        if self.closed:
            return
        if self.hold_terminal and kind in HELD_KINDS:
            self._held.append(payload)
            return
        await self._write(payload)
        self._delta_counter += 1
        if self._progress_due(kind):
            await self.write_progress()

    def _progress_due(self, kind: str) -> bool:
        if (
            kind == "delta"
            and len(self.content) + len(self.reasoning) >= SINK_MIN_FLUSH_CHARS
            and self._delta_counter % SINK_FLUSH_EVERY != 0
        ):
            return False
        if self._last_progress is None:
            return True
        loop = asyncio.get_running_loop()
        return loop.time() - self._last_progress >= SINK_PROGRESS_INTERVAL_SECONDS

    async def enqueue(self, text: str) -> None:
        if "[DONE]" in text and self.final_status == "failed" and not self._done_received:
            self.final_status = "completed"
        await self.flush()

    async def _write(self, payload: dict[str, Any]) -> bool:
        settings = get_settings()
        async with self._lock:
            tagged = dict(payload)
            tagged["event_run"] = self._run
            tagged["event_seq"] = self._sequence
            self._sequence += 1
            self._buffer.append(tagged)
            self._trim_locked(settings.chat_job_sink_buffer_max)
            return await self._flush_locked(settings.chat_job_sink_retry_max)

    def _trim_locked(self, limit: int) -> None:
        maximum = max(16, limit)
        if len(self._buffer) <= maximum:
            return
        index = 0
        while len(self._buffer) > maximum and index < len(self._buffer):
            if as_str(self._buffer[index].get("type")) == "delta":
                self._buffer.pop(index)
                self.dropped += 1
                continue
            index += 1
        while len(self._buffer) > maximum:
            self._buffer.pop(0)
            self.dropped += 1

    async def _flush_locked(self, retries: int) -> bool:
        backoff = SINK_RETRY_BASE_SECONDS
        waited = 0.0
        attempts = max(0, retries)
        while self._buffer:
            payload = self._buffer[0]
            try:
                await self.database.table(EVENTS_TABLE).insert(
                    {"job_id": self.job_id, "payload": payload}
                ).execute()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.failures += 1
                if await _event_present(
                    self.database, self.job_id, payload.get("event_run"), payload.get("event_seq")
                ):
                    self._buffer.pop(0)
                    self.written += 1
                    self.buffered = len(self._buffer)
                    continue
                pause = min(SINK_RETRY_MAX_SECONDS, backoff * (0.8 + random.random() * 0.4))
                if attempts <= 0 or waited + pause > SINK_MAX_BLOCK_SECONDS:
                    self.buffered = len(self._buffer)
                    logger.warning(
                        "event_write_buffered",
                        extra={"job_id": self.job_id, "buffered": self.buffered, "error": str(exc)},
                    )
                    return False
                attempts -= 1
                await asyncio.sleep(pause)
                waited += pause
                backoff = min(SINK_RETRY_MAX_SECONDS, backoff * 2)
                continue
            self._buffer.pop(0)
            self.written += 1
            self.buffered = len(self._buffer)
        return True

    async def flush(self) -> bool:
        async with self._lock:
            return await self._flush_locked(HEARTBEAT_RETRY_MAX)

    async def write_progress(self) -> None:
        if self.closed:
            return
        loop = asyncio.get_running_loop()
        self._last_progress = loop.time()
        payload: dict[str, Any] = {
            "response_content": self.content,
            "response_reasoning": self.reasoning,
            "updated_at": iso(),
        }
        if await self._progress_supported():
            payload["progress_at"] = iso()
        for attempt in range(HEARTBEAT_RETRY_MAX + 1):
            try:
                query = self.database.table(JOBS_TABLE).update(payload).eq("id", self.job_id)
                if self.lease is not None:
                    query = query.eq("status", "running").in_(
                        "lease_expires_at", self.lease.stamps()
                    )
                await query.execute()
                return
            except asyncio.CancelledError:
                raise
            except DatabaseError as exc:
                if exc.status_code in (400, 404) and "progress_at" in payload:
                    self._progress_columns = False
                    payload.pop("progress_at", None)
                    continue
                self.failures += 1
                if attempt >= HEARTBEAT_RETRY_MAX:
                    logger.warning(
                        "progress_write_failed", extra={"job_id": self.job_id, "error": str(exc)}
                    )
                    return
            except Exception as exc:
                self.failures += 1
                if attempt >= HEARTBEAT_RETRY_MAX:
                    logger.warning(
                        "progress_write_failed", extra={"job_id": self.job_id, "error": str(exc)}
                    )
                    return
            await asyncio.sleep(SINK_RETRY_BASE_SECONDS * (attempt + 1))

    async def _progress_supported(self) -> bool:
        if self._progress_columns is not None:
            return self._progress_columns
        self._progress_columns = await columns_available(self.database)
        return self._progress_columns

    def release_held(self) -> list[dict[str, Any]]:
        held = list(self._held)
        self._held.clear()
        return held

    def discard_held(self) -> int:
        count = len(self._held)
        self._held.clear()
        self.hold_terminal = False
        return count

    def stats(self) -> dict[str, Any]:
        return {
            "written": self.written,
            "buffered": len(self._buffer),
            "held": len(self._held),
            "dropped": self.dropped,
            "failures": self.failures,
            "content_chars": len(self.content),
            "reasoning_chars": len(self.reasoning),
        }


async def _renew_lease(
    database: SupabaseDatabase, job_id: str, lease: LeaseToken, next_lease: datetime
) -> dict[str, Any] | None:
    response = await (
        database.table(JOBS_TABLE)
        .update(
            {
                "lease_expires_at": _stamp(next_lease),
                "updated_at": iso(),
            }
        )
        .eq("id", job_id)
        .eq("status", "running")
        .in_("lease_expires_at", lease.stamps())
        .select("stop_requested")
        .execute()
    )
    return response.maybe_single()


async def _read_ownership(
    database: SupabaseDatabase, job_id: str, lease: LeaseToken
) -> dict[str, Any] | None:
    response = await (
        database.table(JOBS_TABLE)
        .select("id, status, lease_expires_at, stop_requested")
        .eq("id", job_id)
        .limit(1)
        .execute()
    )
    row = response.maybe_single()
    if row is None:
        return None
    if as_str(row.get("status")) != "running":
        return None
    if not lease.matches(row.get("lease_expires_at")):
        return None
    return row


async def _heartbeat_loop(
    database: SupabaseDatabase,
    job_id: str,
    cancel: asyncio.Event,
    sink: DurableEventSink,
    lease: LeaseToken,
    state: RunState,
) -> None:
    loop = asyncio.get_running_loop()
    failures = 0
    last_success = loop.time()
    while not cancel.is_set():
        try:
            await asyncio.sleep(HEARTBEAT_MS / 1000)
        except asyncio.CancelledError:
            return
        if cancel.is_set():
            return
        next_lease = _lease_deadline()
        lease.begin(next_lease)
        row: dict[str, Any] | None = None
        renewed = False
        try:
            for attempt in range(HEARTBEAT_RETRY_MAX + 1):
                try:
                    row = await asyncio.wait_for(
                        _renew_lease(database, job_id, lease, next_lease),
                        timeout=HEARTBEAT_CALL_TIMEOUT_SECONDS,
                    )
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if attempt >= HEARTBEAT_RETRY_MAX:
                        raise
                    logger.warning(
                        "heartbeat_retry",
                        extra={"job_id": job_id, "attempt": attempt + 1, "error": str(exc)},
                    )
                    await asyncio.sleep(0.5 * (attempt + 1))
            if row is not None:
                renewed = True
            else:
                row = await _read_ownership(database, job_id, lease)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            failures += 1
            logger.warning(
                "heartbeat_failed",
                extra={"job_id": job_id, "failures": failures, "error": str(exc)},
            )
            silent_ms = (loop.time() - last_success) * 1000
            if failures >= HEARTBEAT_FAILURE_LIMIT or silent_ms >= LEASE_SAFE_MS:
                logger.error("heartbeat_lease_lost", extra={"job_id": job_id})
                state.interrupted = True
                cancel.set()
                try:
                    await sink.send(
                        {
                            "type": "progress",
                            "task_id": job_id,
                            "step": 0,
                            "message": (
                                "Az adatbázis-kapcsolat tartósan megszakadt, ezért a feladat "
                                "biztonságosan leáll; a felügyelő automatikusan újraindítja."
                            ),
                        }
                    )
                except Exception:
                    pass
                return
            continue
        failures = 0
        if row is None:
            logger.error("heartbeat_lease_taken_over", extra={"job_id": job_id})
            state.lost = True
            sink.close()
            cancel.set()
            return
        if renewed:
            lease.confirm(next_lease)
        else:
            actual = parse_iso(row.get("lease_expires_at"))
            if actual is not None:
                lease.confirm(actual)
        last_success = loop.time()
        if as_bool(row.get("stop_requested")):
            state.stop_requested = True
            cancel.set()
            return


def _resolve_status(sink: DurableEventSink, outcome: TerminalState) -> str:
    status = sink.final_status
    if status == "failed":
        if outcome == TerminalState.COMPLETED:
            return "completed"
        if outcome == TerminalState.CANCELLED:
            return "cancelled"
    return status


def _retryable(outcome: TerminalState, status: str) -> bool:
    if status != "failed":
        return False
    return outcome == TerminalState.FAILED


def _retry_delay_seconds(attempts: int) -> float:
    settings = get_settings()
    base = max(0.001, float(settings.chat_job_retry_base_ms) / 1000)
    cap = max(base, float(settings.chat_job_retry_max_ms) / 1000)
    delay = min(cap, base * (2 ** max(0, attempts - 1)))
    return min(cap, delay * (0.8 + random.random() * 0.4))


async def _job_status(database: SupabaseDatabase, job_id: str) -> str | None:
    response = await (
        database.table(JOBS_TABLE).select("status").eq("id", job_id).limit(1).execute()
    )
    row = response.maybe_single()
    return as_str(row.get("status")) if row else None


async def _fenced_update(
    database: SupabaseDatabase,
    job_id: str,
    payload: dict[str, Any],
    lease: LeaseToken,
    retries: int,
    label: str,
) -> bool:
    ambiguous = False
    for attempt in range(retries + 1):
        try:
            response = await (
                database.table(JOBS_TABLE)
                .update(payload)
                .eq("id", job_id)
                .eq("status", "running")
                .in_("lease_expires_at", lease.stamps())
                .execute()
            )
        except asyncio.CancelledError:
            raise
        except DatabaseError as exc:
            if exc.status_code in (400, 404) and "progress_at" in payload:
                payload.pop("progress_at", None)
                continue
            ambiguous = True
            logger.warning(
                label,
                extra={"job_id": job_id, "attempt": attempt + 1, "error": str(exc)},
            )
        except Exception as exc:
            ambiguous = True
            logger.warning(
                label,
                extra={"job_id": job_id, "attempt": attempt + 1, "error": str(exc)},
            )
        else:
            if response.rows:
                return True
            if not ambiguous:
                return False
            try:
                current = await _job_status(database, job_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                current = None
            return current is not None and current == as_str(payload.get("status"))
        if attempt < retries:
            await asyncio.sleep(0.5 * (attempt + 1))
    return False


async def _schedule_retry(
    database: SupabaseDatabase, job_id: str, reason: str, attempts: int, lease: LeaseToken
) -> tuple[bool, float]:
    settings = get_settings()
    if attempts >= settings.chat_job_max_attempts:
        return False, 0.0
    if not await columns_available(database):
        return False, 0.0
    delay = _retry_delay_seconds(attempts)
    due = datetime.now(timezone.utc) + timedelta(seconds=delay)
    payload: dict[str, Any] = {
        "status": "queued",
        "lease_expires_at": None,
        "updated_at": iso(),
        "next_attempt_at": _stamp(due),
        "last_error": reason[:ERROR_MESSAGE_CHARS],
    }
    scheduled = await _fenced_update(
        database, job_id, payload, lease, HEARTBEAT_RETRY_MAX, "retry_schedule_failed"
    )
    if not scheduled:
        return False, 0.0
    return True, delay


async def _release_job(database: SupabaseDatabase, job_id: str, lease: LeaseToken) -> bool:
    payload: dict[str, Any] = {
        "status": "queued",
        "lease_expires_at": None,
        "updated_at": iso(),
    }
    return await _fenced_update(
        database, job_id, payload, lease, HEARTBEAT_RETRY_MAX, "release_failed"
    )


async def _finalize_job(
    database: SupabaseDatabase,
    job_id: str,
    sink: DurableEventSink,
    status: str,
    error: str | None,
    lease: LeaseToken,
    include_content: bool = True,
) -> bool:
    payload: dict[str, Any] = {
        "status": status,
        "error": error,
        "finished_at": iso(),
        "lease_expires_at": None,
        "updated_at": iso(),
    }
    if include_content:
        payload["response_content"] = sink.content
        payload["response_reasoning"] = sink.reasoning
    if await columns_available(database):
        payload["progress_at"] = iso()
    return await _fenced_update(
        database, job_id, payload, lease, FINALIZE_RETRY_MAX, "finalize_failed"
    )


async def fail_chat_job(
    job_id: str, code: str, message: str, outcome: str = "failed"
) -> bool:
    database = database_module.get_admin_database()
    status = "cancelled" if outcome == "cancelled" else "failed"
    reason = message[:ERROR_MESSAGE_CHARS] or "A munka nem fejeződött be."
    takeover_at = _lease_deadline()
    takeover = LeaseToken(takeover_at)
    claimed = False
    for attempt in range(FINALIZE_RETRY_MAX + 1):
        try:
            response = await (
                database.table(JOBS_TABLE)
                .update(
                    {
                        "status": "running",
                        "lease_expires_at": _stamp(takeover_at),
                        "updated_at": iso(),
                    }
                )
                .eq("id", job_id)
                .in_("status", list(ACTIVE_STATUSES))
                .execute()
            )
            claimed = bool(response.rows)
            break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "fail_job_takeover_failed",
                extra={"job_id": job_id, "attempt": attempt + 1, "error": str(exc)},
            )
            if attempt < FINALIZE_RETRY_MAX:
                await asyncio.sleep(0.5 * (attempt + 1))
    if not claimed:
        return False
    run = uuid.uuid4().hex[:12]
    events: list[dict[str, Any]] = [
        {"type": "error", "task_id": job_id, "code": code, "message": reason},
        {
            "type": "terminal",
            "task_id": job_id,
            "outcome": status,
            "reason": reason,
            "steps": 0,
            "tool_calls": 0,
            "duration_ms": 0,
        },
        {"type": "done", "task_id": job_id, "outcome": status},
    ]
    for sequence, event in enumerate(events):
        event["event_run"] = run
        event["event_seq"] = sequence
        for attempt in range(HEARTBEAT_RETRY_MAX + 1):
            try:
                await database.table(EVENTS_TABLE).insert(
                    {"job_id": job_id, "payload": event}
                ).execute()
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if await _event_present(database, job_id, run, sequence):
                    break
                if attempt >= HEARTBEAT_RETRY_MAX:
                    logger.warning(
                        "terminal_event_failed",
                        extra={"job_id": job_id, "event": event.get("type"), "error": str(exc)},
                    )
                    break
                await asyncio.sleep(0.4 * (attempt + 1))
    payload: dict[str, Any] = {
        "status": status,
        "error": reason,
        "finished_at": iso(),
        "lease_expires_at": None,
        "updated_at": iso(),
    }
    updated = await _fenced_update(
        database, job_id, payload, takeover, FINALIZE_RETRY_MAX, "fail_job_update_failed"
    )
    await checkpoints.clear_checkpoint(job_id)
    return updated


async def job_queue_summary() -> dict[str, Any]:
    database = database_module.get_admin_database()
    extended = await columns_available(database)
    summary: dict[str, Any] = {
        "extended_columns": extended,
        "checkpointing": checkpoints.checkpoint_stats(),
        "oldest_queued_seconds": None,
        "expired_leases": 0,
        "due_retries": 0,
        "stop_requested": 0,
    }
    for status in (*ACTIVE_STATUSES, *TERMINAL_STATUSES):
        try:
            summary[status] = await (
                database.table(JOBS_TABLE)
                .select("id", count="exact", head=True)
                .eq("status", status)
                .count_exact()
            )
        except Exception as exc:
            logger.warning("summary_count_failed", extra={"status": status, "error": str(exc)})
            summary[status] = None
    try:
        expired_count = await (
            database.table(JOBS_TABLE)
            .select("id", count="exact", head=True)
            .eq("status", "running")
            .lt("lease_expires_at", iso())
            .count_exact()
        )
        summary["expired_leases"] = int(expired_count)
    except Exception:
        summary["expired_leases"] = None
    try:
        stop_count = await (
            database.table(JOBS_TABLE)
            .select("id", count="exact", head=True)
            .in_("status", list(ACTIVE_STATUSES))
            .eq("stop_requested", True)
            .count_exact()
        )
        summary["stop_requested"] = int(stop_count)
    except Exception as exc:
        logger.warning("summary_stop_failed", extra={"error": str(exc)})
        summary["stop_requested"] = None
    if extended:
        try:
            summary["due_retries"] = int(
                await database.table(JOBS_TABLE)
                .select("id", count="exact", head=True)
                .eq("status", "queued")
                .lte("next_attempt_at", iso())
                .count_exact()
            )
        except Exception:
            summary["due_retries"] = None
    try:
        oldest_response = await (
            database.table(JOBS_TABLE)
            .select("created_at")
            .eq("status", "queued")
            .order("created_at", desc=False, nullsfirst=True)
            .limit(1)
            .execute()
        )
        row = oldest_response.maybe_single()
        if row is not None:
            age = age_ms(row.get("created_at"))
            if age is not None:
                summary["oldest_queued_seconds"] = round(age / 1000, 3)
    except Exception as exc:
        logger.warning("summary_oldest_failed", extra={"error": str(exc)})
    return summary


async def _load_job(database: SupabaseDatabase, job_id: str) -> dict[str, Any] | None:
    extended = await columns_available(database)
    columns = EXTENDED_COLUMNS if extended else LEGACY_COLUMNS
    try:
        response = await (
            database.table(JOBS_TABLE).select(columns).eq("id", job_id).limit(1).execute()
        )
    except DatabaseError as exc:
        if exc.status_code not in (400, 404) or not extended:
            raise
        response = await (
            database.table(JOBS_TABLE).select(LEGACY_COLUMNS).eq("id", job_id).limit(1).execute()
        )
    return response.maybe_single()


def _claim_query(
    database: SupabaseDatabase,
    job_id: str,
    job: dict[str, Any],
    status: str,
    payload: dict[str, Any],
    use_extended: bool,
) -> Any:
    query = database.table(JOBS_TABLE).update(payload).eq("id", job_id).eq("status", status)
    lease = job.get("lease_expires_at")
    if lease is not None:
        parsed = parse_iso(lease)
        query = query.eq("lease_expires_at", _stamp(parsed) if parsed is not None else lease)
    else:
        query = query.is_("lease_expires_at", None)
    if use_extended:
        if job.get("attempts") is None:
            query = query.is_("attempts", None)
        else:
            query = query.eq("attempts", as_int(job.get("attempts")))
    return query


async def _claim_job(
    database: SupabaseDatabase, job_id: str, job: dict[str, Any], status: str
) -> LeaseToken | None:
    stamp = iso()
    lease_at = _lease_deadline()
    extended = await columns_available(database)
    payload: dict[str, Any] = {
        "status": "running",
        "started_at": stamp,
        "updated_at": stamp,
        "lease_expires_at": _stamp(lease_at),
    }
    if extended:
        payload["progress_at"] = stamp
        payload["attempts"] = as_int(job.get("attempts")) + 1
    try:
        claimed = await _claim_query(database, job_id, job, status, payload, extended).execute()
    except DatabaseError as exc:
        if exc.status_code in (400, 404) and extended:
            payload.pop("progress_at", None)
            payload.pop("attempts", None)
            claimed = await _claim_query(database, job_id, job, status, payload, False).execute()
        else:
            raise
    if not claimed.rows:
        return None
    return LeaseToken(lease_at)


async def _finish_stopped(
    database: SupabaseDatabase, job_id: str, lease: LeaseToken
) -> dict[str, str]:
    stop_sink = DurableEventSink(job_id, database, lease)
    stop_sink.hold_terminal = True
    await stop_sink.send(
        {
            "type": "terminal",
            "task_id": job_id,
            "outcome": "cancelled",
            "reason": "A feladatot a felhasználó megszakította.",
            "steps": 0,
            "tool_calls": 0,
            "duration_ms": 0,
        }
    )
    await stop_sink.send({"type": "done", "task_id": job_id, "outcome": "cancelled"})
    finalized = await _finalize_job(
        database, job_id, stop_sink, "cancelled", None, lease, include_content=False
    )
    stop_sink.hold_terminal = False
    if not finalized:
        stop_sink.discard_held()
        stop_sink.close()
        return {"status": "unknown"}
    for held in stop_sink.release_held():
        await stop_sink.send(held)
    await stop_sink.flush()
    await checkpoints.clear_checkpoint(job_id)
    return {"status": "cancelled"}


async def execute_chat_job(job_id: str) -> dict[str, str]:
    settings = get_settings()
    database = database_module.get_admin_database()
    try:
        job = await _load_job(database, job_id)
    except Exception as exc:
        logger.warning("job_load_failed", extra={"job_id": job_id, "error": str(exc)})
        return {"status": "unknown"}
    if job is None:
        return {"status": "missing"}
    status = as_str(job.get("status"))
    if status in TERMINAL_STATUSES:
        return {"status": status}
    stop_requested = as_bool(job.get("stop_requested"))
    lease_expiry = parse_iso(job.get("lease_expires_at"))
    if status == "running" and lease_expiry is not None and lease_expiry > now():
        return {"status": "running"}
    scheduled_at = parse_iso(job.get("next_attempt_at"))
    if (
        status == "queued"
        and scheduled_at is not None
        and scheduled_at > now()
        and not stop_requested
    ):
        return {"status": "queued"}
    previous_attempts = as_int(job.get("attempts"))
    if (
        not stop_requested
        and job.get("attempts") is not None
        and previous_attempts >= settings.chat_job_max_attempts
    ):
        await fail_chat_job(
            job_id,
            "ATTEMPTS_EXHAUSTED",
            "A feladat a megengedett kísérletek után sem fejeződött be.",
        )
        return {"status": "failed"}
    try:
        lease = await _claim_job(database, job_id, job, status)
    except Exception as exc:
        logger.warning("job_claim_failed", extra={"job_id": job_id, "error": str(exc)})
        return {"status": "unknown"}
    if lease is None:
        return {"status": "running"}
    if stop_requested:
        return await _finish_stopped(database, job_id, lease)

    cancel = asyncio.Event()
    state = RunState()
    sink = DurableEventSink(job_id, database, lease)
    checkpoint = await checkpoints.load_checkpoint(job_id)
    attempts = previous_attempts + 1
    if checkpoint is not None:
        await sink.send(
            {
                "type": "progress",
                "task_id": job_id,
                "step": checkpoint.steps,
                "message": (
                    f"A feladat a {attempts}. kísérletben, a mentett {checkpoint.steps}. lépéstől "
                    "folytatódik."
                ),
            }
        )
    sink.hold_terminal = True
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(database, job_id, cancel, sink, lease, state)
    )
    history = as_list(job.get("history"))
    outcome = TerminalState.FAILED
    failure_detail: str | None = None
    failure_public: str | None = None
    try:
        outcome = await run_durable_chat(job_id, history, sink.send, sink.enqueue, cancel, checkpoint)
    except asyncio.CancelledError:
        outcome = TerminalState.CANCELLED
        raise
    except AgentFault as fault:
        outcome = TerminalState.FAILED
        failure_detail = fault.message
        failure_public = fault.message
        sink.final_status = "failed"
        await sink.send(
            {
                "type": "error",
                "task_id": job_id,
                "code": fault.code,
                "message": fault.message,
            }
        )
    except Exception as exc:
        logger.exception("background_job_failed", extra={"job_id": job_id})
        outcome = TerminalState.FAILED
        failure_detail = str(exc) or GENERIC_FAILURE_MESSAGE
        failure_public = GENERIC_FAILURE_MESSAGE
        sink.final_status = "failed"
        await sink.send(
            {
                "type": "error",
                "task_id": job_id,
                "code": "BACKGROUND_FAILURE",
                "message": GENERIC_FAILURE_MESSAGE,
            }
        )
    finally:
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)

    if state.lost:
        sink.close()
        sink.discard_held()
        logger.warning("job_lease_lost", extra={"job_id": job_id})
        return {"status": "lost"}

    if state.interrupted and outcome != TerminalState.COMPLETED:
        sink.discard_held()
        sink.close()
        released = await _release_job(database, job_id, lease)
        if released:
            _schedule_dispatch_after(job_id, supervisor_origin(), INTERRUPT_REDISPATCH_SECONDS)
        logger.warning("job_interrupted", extra={"job_id": job_id, "released": released})
        return {"status": "interrupted"}

    final_status = _resolve_status(sink, outcome)
    if state.stop_requested and final_status == "failed":
        final_status = "cancelled"
    if (
        _retryable(outcome, final_status)
        and not cancel.is_set()
        and attempts < settings.chat_job_max_attempts
    ):
        reason = failure_detail or "A végrehajtás átmeneti hiba miatt megszakadt."
        scheduled, delay = await _schedule_retry(database, job_id, reason, attempts, lease)
        if scheduled:
            sink.discard_held()
            await sink.send(
                {
                    "type": "progress",
                    "task_id": job_id,
                    "step": 0,
                    "message": (
                        f"Átmeneti hiba miatt a feladat automatikusan újraindul "
                        f"({attempts}/{settings.chat_job_max_attempts}. "
                        "kísérlet); a mentett állapotból folytatom."
                    ),
                }
            )
            await sink.flush()
            _schedule_dispatch_after(
                job_id, supervisor_origin(), delay + RETRY_DISPATCH_MARGIN_SECONDS
            )
            return {"status": "queued"}
    error_message = None
    if final_status == "failed":
        error_message = (failure_public or GENERIC_FAILURE_MESSAGE)[:ERROR_MESSAGE_CHARS]
    finalized = await _finalize_job(database, job_id, sink, final_status, error_message, lease)
    if not finalized:
        sink.discard_held()
        sink.close()
        logger.warning("job_finalize_lost", extra={"job_id": job_id, "status": final_status})
        return {"status": "unknown"}
    sink.hold_terminal = False
    for held in sink.release_held():
        await sink.send(held)
    await checkpoints.clear_checkpoint(job_id)
    await sink.flush()
    logger.info(
        "job_finished",
        extra={"job_id": job_id, "status": final_status, "sink": sink.stats()},
    )
    return {"status": final_status}


async def _read_events(
    database: SupabaseDatabase, job_id: str, cursor: int
) -> tuple[list[dict[str, Any]], int]:
    response = await (
        database.table(EVENTS_TABLE)
        .select("id", "payload")
        .eq("job_id", job_id)
        .gt("id", cursor)
        .order("id", desc=False)
        .limit(200)
        .execute()
    )
    latest = cursor
    frames: list[dict[str, Any]] = []
    for row in response.rows:
        row_id = as_int(row.get("id"))
        latest = max(latest, row_id)
        payload = row.get("payload")
        if not isinstance(payload, dict):
            payload = {"type": "unknown", "payload": payload}
        frames.append({"id": row_id, "payload": payload})
    return frames, latest


async def follow_chat_job(job_id: str, after: int, request_url: str) -> AsyncIterator[str]:
    database = database_module.get_admin_database()
    origin = worker_origin(request_url)
    cursor = max(0, after)
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    last_heartbeat = started_at - FOLLOW_HEARTBEAT_MS / 1000
    last_revive = started_at
    last_status_check = started_at - FOLLOW_STATUS_CHECK_MS / 1000
    terminal_seen_at: float | None = None
    failures = 0
    backoff = FOLLOW_RETRY_BASE_SECONDS
    poll_base = FOLLOW_POLL_MS / 1000
    poll_delay = poll_base
    try:
        await revive_stale_chat_job_origin(job_id, origin)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("follow_revive_failed", extra={"job_id": job_id, "error": str(exc)})
    while True:
        if loop.time() - started_at > FOLLOW_MAX_MS / 1000:
            yield sse_frame(
                {
                    "type": "error",
                    "task_id": job_id,
                    "code": "RUNTIME_LIMIT",
                    "message": "Lejárt a teljes feladathatáridő.",
                }
            )
            return
        try:
            frames, cursor = await _read_events(database, job_id, cursor)
            last_terminal: bool | None = None
            for frame in frames:
                payload = frame["payload"]
                yield sse_frame(payload, frame["id"])
                last_terminal = as_str(payload.get("type")) in ("done", "error")
            if last_terminal is True:
                status = await _job_status(database, job_id)
                if status is None or status not in ACTIVE_STATUSES:
                    return
                if terminal_seen_at is None:
                    terminal_seen_at = loop.time()
                    yield sse_comment("retrying")
            elif last_terminal is False:
                terminal_seen_at = None
            if (
                terminal_seen_at is not None
                and loop.time() - terminal_seen_at > FOLLOW_TERMINAL_GRACE_MS / 1000
            ):
                rest, cursor = await _read_events(database, job_id, cursor)
                for frame in rest:
                    yield sse_frame(frame["payload"], frame["id"])
                return
            if not frames:
                if loop.time() - last_status_check >= FOLLOW_STATUS_CHECK_MS / 1000:
                    last_status_check = loop.time()
                    status = await _job_status(database, job_id)
                    if status is None or status not in ACTIVE_STATUSES:
                        rest, cursor = await _read_events(database, job_id, cursor)
                        for frame in rest:
                            yield sse_frame(frame["payload"], frame["id"])
                        return
                if loop.time() - last_revive >= FOLLOW_REVIVE_MS / 1000:
                    last_revive = loop.time()
                    try:
                        revived = await revive_stale_chat_job_origin(job_id, origin)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.warning(
                            "follow_revive_failed", extra={"job_id": job_id, "error": str(exc)}
                        )
                    else:
                        if revived:
                            yield sse_comment("revive")
                poll_delay = min(FOLLOW_POLL_MAX_MS / 1000, poll_delay * 1.25)
            else:
                poll_delay = poll_base
            failures = 0
            backoff = FOLLOW_RETRY_BASE_SECONDS
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failures += 1
            logger.warning(
                "follow_read_failed",
                extra={"job_id": job_id, "failures": failures, "error": str(exc)},
            )
            if failures > FOLLOW_FAILURE_LIMIT:
                yield sse_frame(
                    {
                        "type": "error",
                        "task_id": job_id,
                        "code": "FOLLOW_FAILURE",
                        "message": "Az eseményfolyam megszakadt.",
                    }
                )
                return
            yield sse_comment("retry")
            await asyncio.sleep(
                min(FOLLOW_RETRY_MAX_SECONDS, backoff * (0.8 + random.random() * 0.4))
            )
            backoff = min(FOLLOW_RETRY_MAX_SECONDS, backoff * 2)
            continue
        if loop.time() - last_heartbeat >= FOLLOW_HEARTBEAT_MS / 1000:
            last_heartbeat = loop.time()
            yield sse_comment("heartbeat")
        await asyncio.sleep(poll_delay)