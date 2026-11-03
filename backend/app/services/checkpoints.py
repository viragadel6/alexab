from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from app import database as database_module
from app.config import get_settings
from app.database import DatabaseError
from app.utils.clock import (
    as_bool,
    as_dict,
    as_int,
    as_list,
    as_str,
    iso,
    utf8_bytes,
)

JOBS_TABLE = "chat_jobs"
CHECKPOINT_VERSION = 3
PROBE_JOB_ID = "00000000-0000-4000-8000-000000000000"
MAX_TRACE_ENTRIES = 80
MAX_TRACE_CHARS = 600
MAX_LEDGER_ENTRIES = 64
MAX_LEDGER_CONTENT_CHARS = 1200
MAX_PENDING_CALLS = 16
MAX_ARGUMENT_CHARS = 20_000
MAX_REFLECTIONS = 24
MAX_PLAN_CHARS = 20_000
MAX_OBJECTIVE_CHARS = 8_000
MAX_SIGMA_STRINGS = 60
MAX_SIGMA_STRING_CHARS = 800
MEMORY_LIMIT = 64

logger = logging.getLogger("app.checkpoints")


def _clip_list(values: list[Any], limit: int) -> list[str]:
    collected: list[str] = []
    for value in values[-limit:] if limit > 0 else []:
        text = as_str(value)
        if text:
            collected.append(text)
    return collected


@dataclass
class LedgerEntry:
    call_id: str
    fingerprint: str
    tool: str
    status: str
    content: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "fingerprint": self.fingerprint,
            "tool": self.tool,
            "status": self.status,
            "content": self.content,
        }

    @staticmethod
    def from_payload(payload: Any) -> LedgerEntry | None:
        record = as_dict(payload)
        call_id = as_str(record.get("call_id"), 200)
        if not call_id:
            return None
        return LedgerEntry(
            call_id=call_id,
            fingerprint=as_str(record.get("fingerprint"), 200),
            tool=as_str(record.get("tool"), 200),
            status=as_str(record.get("status"), 40) or "completed",
            content=as_str(record.get("content"), MAX_LEDGER_CONTENT_CHARS),
        )


@dataclass
class PendingCall:
    id: str
    name: str
    arguments: str

    def to_payload(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @staticmethod
    def from_payload(payload: Any) -> PendingCall | None:
        record = as_dict(payload)
        call_id = as_str(record.get("id"), 200)
        name = as_str(record.get("name"), 200)
        if not call_id or not name:
            return None
        return PendingCall(
            id=call_id,
            name=name,
            arguments=as_str(record.get("arguments"), MAX_ARGUMENT_CHARS) or "{}",
        )


@dataclass
class Checkpoint:
    version: int = CHECKPOINT_VERSION
    task_id: str = ""
    steps: int = 0
    call_count: int = 0
    objective: str = ""
    plan: str = ""
    sigma: dict[str, Any] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)
    reflections: list[dict[str, Any]] = field(default_factory=list)
    verify_failures: int = 0
    replans: int = 0
    verified: bool = False
    message_count: int = 0
    step_appended: bool = False
    pending_calls: list[PendingCall] = field(default_factory=list)
    ledger: list[LedgerEntry] = field(default_factory=list)
    sandbox_recreations: int = 0
    step_retries: int = 0
    loop_strikes: int = 0
    uncertainty_strikes: int = 0
    saved_at: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "task_id": self.task_id,
            "steps": self.steps,
            "call_count": self.call_count,
            "objective": self.objective,
            "plan": self.plan,
            "sigma": self.sigma,
            "trace": list(self.trace),
            "reflections": list(self.reflections),
            "verify_failures": self.verify_failures,
            "replans": self.replans,
            "verified": self.verified,
            "message_count": self.message_count,
            "step_appended": self.step_appended,
            "pending_calls": [call.to_payload() for call in self.pending_calls],
            "ledger": [entry.to_payload() for entry in self.ledger],
            "sandbox_recreations": self.sandbox_recreations,
            "step_retries": self.step_retries,
            "loop_strikes": self.loop_strikes,
            "uncertainty_strikes": self.uncertainty_strikes,
            "saved_at": self.saved_at,
        }

    @staticmethod
    def from_payload(payload: Any) -> Checkpoint | None:
        record = as_dict(payload)
        if not record:
            return None
        version = as_int(record.get("version"), 0)
        if version <= 0 or version > CHECKPOINT_VERSION:
            return None
        ledger: list[LedgerEntry] = []
        for item in as_list(record.get("ledger")):
            entry = LedgerEntry.from_payload(item)
            if entry is not None:
                ledger.append(entry)
        pending: list[PendingCall] = []
        for item in as_list(record.get("pending_calls")):
            call = PendingCall.from_payload(item)
            if call is not None:
                pending.append(call)
        reflections: list[dict[str, Any]] = []
        for item in as_list(record.get("reflections")):
            record_item = as_dict(item)
            if record_item:
                reflections.append(record_item)
        return Checkpoint(
            version=version,
            task_id=as_str(record.get("task_id"), 64),
            steps=as_int(record.get("steps"), 0, 0, 100_000),
            call_count=as_int(record.get("call_count"), 0, 0, 100_000),
            objective=as_str(record.get("objective"), MAX_OBJECTIVE_CHARS),
            plan=as_str(record.get("plan"), MAX_PLAN_CHARS),
            sigma=as_dict(record.get("sigma")),
            trace=_clip_list(as_list(record.get("trace")), MAX_TRACE_ENTRIES),
            reflections=reflections[-MAX_REFLECTIONS:],
            verify_failures=as_int(record.get("verify_failures"), 0, 0, 100),
            replans=as_int(record.get("replans"), 0, 0, 100),
            verified=as_bool(record.get("verified")),
            message_count=as_int(record.get("message_count"), 0, 0, 100_000),
            step_appended=as_bool(record.get("step_appended")),
            pending_calls=pending[:MAX_PENDING_CALLS],
            ledger=ledger[-MAX_LEDGER_ENTRIES:],
            sandbox_recreations=as_int(record.get("sandbox_recreations"), 0, 0, 100),
            step_retries=as_int(record.get("step_retries"), 0, 0, 10_000),
            loop_strikes=as_int(record.get("loop_strikes"), 0, 0, 100),
            uncertainty_strikes=as_int(record.get("uncertainty_strikes"), 0, 0, 100),
            saved_at=as_str(record.get("saved_at"), 64),
        )

    def trimmed(self, max_bytes: int) -> Checkpoint:
        limit = max(4096, max_bytes)
        clone = Checkpoint(
            version=self.version,
            task_id=self.task_id,
            steps=self.steps,
            call_count=self.call_count,
            objective=self.objective,
            plan=self.plan,
            sigma=dict(self.sigma),
            trace=[entry[:MAX_TRACE_CHARS] for entry in self.trace][-MAX_TRACE_ENTRIES:],
            reflections=[dict(item) for item in self.reflections][-MAX_REFLECTIONS:],
            verify_failures=self.verify_failures,
            replans=self.replans,
            verified=self.verified,
            message_count=self.message_count,
            step_appended=self.step_appended,
            pending_calls=list(self.pending_calls),
            ledger=list(self.ledger),
            sandbox_recreations=self.sandbox_recreations,
            step_retries=self.step_retries,
            loop_strikes=self.loop_strikes,
            uncertainty_strikes=self.uncertainty_strikes,
            saved_at=self.saved_at,
        )
        if utf8_bytes(clone.to_payload()) <= limit:
            return clone
        while len(clone.trace) > 20 and utf8_bytes(clone.to_payload()) > limit:
            clone.trace = clone.trace[len(clone.trace) // 4 :]
        if utf8_bytes(clone.to_payload()) <= limit:
            return clone
        while len(clone.reflections) > 4 and utf8_bytes(clone.to_payload()) > limit:
            clone.reflections = clone.reflections[len(clone.reflections) // 2 :]
        if utf8_bytes(clone.to_payload()) <= limit:
            return clone
        clone.ledger = [
            LedgerEntry(
                call_id=entry.call_id,
                fingerprint=entry.fingerprint,
                tool=entry.tool,
                status=entry.status,
                content=entry.content[:200],
            )
            for entry in clone.ledger
        ][-MAX_LEDGER_ENTRIES:]
        if utf8_bytes(clone.to_payload()) <= limit:
            return clone
        clone.trace = clone.trace[-8:]
        clone.sigma = _shrink_sigma(clone.sigma)
        if utf8_bytes(clone.to_payload()) <= limit:
            return clone
        clone.plan = clone.plan[:4000]
        clone.objective = clone.objective[:2000]
        clone.reflections = []
        clone.ledger = [
            LedgerEntry(
                call_id=entry.call_id,
                fingerprint=entry.fingerprint,
                tool=entry.tool,
                status=entry.status,
                content="",
            )
            for entry in clone.ledger
        ][-16:]
        return clone


def _shrink_sigma(sigma: dict[str, Any]) -> dict[str, Any]:
    shrunk = dict(sigma)
    for key in ("facts", "blockers", "next_steps"):
        values = _clip_list(as_list(shrunk.get(key)), MAX_SIGMA_STRINGS)
        shrunk[key] = [value[:MAX_SIGMA_STRING_CHARS] for value in values]
    subgoals = as_list(shrunk.get("subgoals"))
    if len(subgoals) > MAX_SIGMA_STRINGS:
        shrunk["subgoals"] = subgoals[-MAX_SIGMA_STRINGS:]
    if isinstance(shrunk.get("goal"), dict):
        goal = dict(as_dict(shrunk.get("goal")))
        for key in ("required_files", "state_pointers", "file_contents"):
            values = as_list(goal.get(key))
            if len(values) > MAX_SIGMA_STRINGS:
                goal[key] = values[-MAX_SIGMA_STRINGS:]
        shrunk["goal"] = goal
    return shrunk


class CheckpointStore:
    def __init__(self) -> None:
        self._supported: bool | None = None
        self._probe_lock = asyncio.Lock()
        self._memory: dict[str, Checkpoint] = {}
        self._memory_lock = asyncio.Lock()
        self._saved = 0
        self._persisted = 0
        self._failed = 0
        self._loaded = 0
        self._cleared = 0

    async def supported(self) -> bool:
        if self._supported is not None:
            return self._supported
        async with self._probe_lock:
            if self._supported is not None:
                return self._supported
            probed = await self._probe()
            if probed is not None:
                self._supported = probed
                return probed
            return False

    async def _probe(self) -> bool | None:
        settings = get_settings()
        if not settings.agent_checkpoint_enabled:
            return False
        if not settings.supabase_url or not settings.supabase_service_role_key:
            return False
        try:
            database = database_module.get_admin_database()
            await (
                database.table(JOBS_TABLE)
                .select("checkpoint")
                .eq("id", PROBE_JOB_ID)
                .limit(1)
                .execute()
            )
        except DatabaseError as exc:
            if exc.status_code in (400, 404):
                logger.warning(
                    "checkpoint_column_missing",
                    extra={"job_id": PROBE_JOB_ID, "error": str(exc)},
                )
                return False
            logger.warning("checkpoint_probe_transient", extra={"error": str(exc)})
            return None
        except Exception as exc:
            logger.warning("checkpoint_probe_failed", extra={"error": str(exc)})
            return None
        return True

    async def _remember(self, checkpoint: Checkpoint) -> None:
        async with self._memory_lock:
            self._memory[checkpoint.task_id] = checkpoint
            while len(self._memory) > MEMORY_LIMIT:
                oldest = next(iter(self._memory))
                self._memory.pop(oldest, None)

    async def _recall(self, task_id: str) -> Checkpoint | None:
        async with self._memory_lock:
            stored = self._memory.get(task_id)
            return stored

    async def save(self, checkpoint: Checkpoint) -> bool:
        if not checkpoint.task_id:
            return False
        settings = get_settings()
        if not settings.agent_checkpoint_enabled:
            return False
        checkpoint.saved_at = iso()
        payload = checkpoint.trimmed(settings.agent_checkpoint_max_bytes).to_payload()
        await self._remember(checkpoint)
        self._saved += 1
        if not await self.supported():
            return False
        try:
            database = database_module.get_admin_database()
            await (
                database.table(JOBS_TABLE)
                .update({"checkpoint": payload, "progress_at": iso()})
                .eq("id", checkpoint.task_id)
                .execute()
            )
        except Exception as exc:
            self._failed += 1
            logger.warning(
                "checkpoint_save_failed",
                extra={"job_id": checkpoint.task_id, "error": str(exc)},
            )
            return False
        self._persisted += 1
        return True

    async def load(self, task_id: str) -> Checkpoint | None:
        if not task_id:
            return None
        if await self.supported():
            try:
                database = database_module.get_admin_database()
                response = await (
                    database.table(JOBS_TABLE)
                    .select("checkpoint")
                    .eq("id", task_id)
                    .limit(1)
                    .execute()
                )
                row = response.maybe_single()
                checkpoint = Checkpoint.from_payload(row.get("checkpoint")) if row else None
                if checkpoint is not None:
                    checkpoint.task_id = task_id
                    self._loaded += 1
                    await self._remember(checkpoint)
                    return checkpoint
            except Exception as exc:
                logger.warning(
                    "checkpoint_load_failed", extra={"job_id": task_id, "error": str(exc)}
                )
        stored = await self._recall(task_id)
        if stored is not None:
            self._loaded += 1
        return stored

    async def touch(self, task_id: str) -> bool:
        if not task_id or not await self.supported():
            return False
        try:
            database = database_module.get_admin_database()
            await (
                database.table(JOBS_TABLE)
                .update({"progress_at": iso()})
                .eq("id", task_id)
                .execute()
            )
        except Exception as exc:
            logger.warning(
                "checkpoint_touch_failed", extra={"job_id": task_id, "error": str(exc)}
            )
            return False
        return True

    async def clear(self, task_id: str) -> None:
        if not task_id:
            return
        async with self._memory_lock:
            self._memory.pop(task_id, None)
        self._cleared += 1
        if not await self.supported():
            return
        try:
            database = database_module.get_admin_database()
            await (
                database.table(JOBS_TABLE)
                .update({"checkpoint": None})
                .eq("id", task_id)
                .execute()
            )
        except Exception as exc:
            logger.warning(
                "checkpoint_clear_failed", extra={"job_id": task_id, "error": str(exc)}
            )

    def reset_capability(self) -> None:
        self._supported = None

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": get_settings().agent_checkpoint_enabled,
            "persistent": bool(self._supported),
            "saved": self._saved,
            "persisted": self._persisted,
            "failed": self._failed,
            "loaded": self._loaded,
            "cleared": self._cleared,
            "memory_entries": len(self._memory),
        }


store = CheckpointStore()


async def save_checkpoint(checkpoint: Checkpoint) -> bool:
    return await store.save(checkpoint)


async def load_checkpoint(task_id: str) -> Checkpoint | None:
    return await store.load(task_id)


async def clear_checkpoint(task_id: str) -> None:
    await store.clear(task_id)


async def touch_checkpoint(task_id: str) -> bool:
    return await store.touch(task_id)


def checkpoint_stats() -> dict[str, Any]:
    return store.stats()
