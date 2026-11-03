from __future__ import annotations

import asyncio
import logging
import os
import random
import socket
import time
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Awaitable, Callable

import httpx

from app import database as database_module
from app.agent import cloud_browser
from app.config import get_settings
from app.database import DatabaseError
from app.services.chat_jobs import (
    dispatch_chat_job_origin,
    fail_chat_job,
    job_queue_summary,
    supervisor_origin,
)
from app.utils.clock import (
    age_ms,
    as_bool,
    as_dict,
    as_int,
    as_list,
    as_str,
    iso,
    now,
    parse_iso,
)

JOBS_TABLE = "chat_jobs"
SESSION_TABLE = "chat_browser_sessions"
LEASE_TABLE = "agent_supervisor_leases"
RECOVERY_TABLE = "agent_recovery_log"
LEASE_NAME = "agent-supervisor"
TERMINAL_STATUSES = ("completed", "cancelled", "failed")
BROWSER_LIVE_STATUSES = ("active", "requested", "human", "agent", "starting")
PROBE_TIMEOUT_SECONDS = 8.0
PROBE_CONNECT_SECONDS = 4.0
MAX_DISPATCH_PER_SWEEP = 12
MAX_BROWSER_PER_SWEEP = 8
RECOVERY_DETAIL_CHARS = 800
RECOVERY_LOG_MEMORY = 60
RECOVERY_LOG_ACTIONS = 20
GUARD_RESTART_DELAY_SECONDS = 5.0
MAX_GUARD_RESTART_DELAY_SECONDS = 120.0
JOB_LIFETIME_LIMIT_MS = 21_600_000
DUE_FLOOR_MS = 5_000

logger = logging.getLogger("app.supervisor")

Probe = Callable[[httpx.AsyncClient], Awaitable[tuple[bool, str, int]]]


def _owner_identity() -> str:
    hostname = socket.gethostname() or "ismeretlen"
    return f"{hostname}:{os.getpid()}:{uuid.uuid4().hex[:12]}"


@dataclass
class CircuitBreaker:
    name: str
    threshold: int = 4
    cooldown_seconds: float = 120.0
    failures: int = 0
    successes: int = 0
    skipped: int = 0
    state: str = "closed"
    opened_at: float = 0.0
    last_error: str = ""
    last_check_at: str = ""
    last_latency_ms: int = 0

    def allow(self) -> bool:
        if self.state != "open":
            return True
        if time.monotonic() - self.opened_at >= self.cooldown_seconds:
            self.state = "half_open"
            return True
        return False

    def record_success(self, latency_ms: int) -> None:
        self.successes += 1
        self.failures = 0
        self.state = "closed"
        self.last_latency_ms = latency_ms
        self.last_error = ""
        self.last_check_at = iso()

    def record_failure(self, error: str) -> None:
        self.failures += 1
        self.last_error = error[:RECOVERY_DETAIL_CHARS]
        self.last_check_at = iso()
        if self.state == "half_open" or self.failures >= self.threshold:
            self.state = "open"
            self.opened_at = time.monotonic()

    def record_skip(self) -> None:
        self.skipped += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "failures": self.failures,
            "successes": self.successes,
            "skipped": self.skipped,
            "threshold": self.threshold,
            "cooldown_seconds": round(self.cooldown_seconds, 3),
            "last_error": self.last_error,
            "last_check_at": self.last_check_at,
            "last_latency_ms": self.last_latency_ms,
        }


@dataclass
class RecoveryAction:
    kind: str
    job_id: str
    detail: str
    created_at: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "job_id": self.job_id,
            "detail": self.detail,
            "created_at": self.created_at,
        }


@dataclass
class SupervisorMetrics:
    sweeps: int = 0
    failed_sweeps: int = 0
    dispatched: int = 0
    requeued: int = 0
    dead_lettered: int = 0
    stopped_requests: int = 0
    browsers_stopped: int = 0
    pruned_rows: int = 0
    probe_failures: int = 0
    leader_changes: int = 0
    guard_restarts: int = 0
    last_sweep_at: str = ""
    last_sweep_duration_ms: int = 0
    last_result: dict[str, Any] = field(default_factory=dict)
    actions: list[RecoveryAction] = field(default_factory=list)

    def add_action(self, kind: str, job_id: str, detail: str) -> RecoveryAction:
        action = RecoveryAction(
            kind=kind,
            job_id=job_id,
            detail=detail[:RECOVERY_DETAIL_CHARS],
            created_at=iso(),
        )
        self.actions.append(action)
        if len(self.actions) > RECOVERY_LOG_MEMORY:
            self.actions = self.actions[-RECOVERY_LOG_MEMORY:]
        return action

    def snapshot(self) -> dict[str, Any]:
        return {
            "sweeps": self.sweeps,
            "failed_sweeps": self.failed_sweeps,
            "dispatched": self.dispatched,
            "requeued": self.requeued,
            "dead_lettered": self.dead_lettered,
            "stopped_requests": self.stopped_requests,
            "browsers_stopped": self.browsers_stopped,
            "pruned_rows": self.pruned_rows,
            "probe_failures": self.probe_failures,
            "leader_changes": self.leader_changes,
            "guard_restarts": self.guard_restarts,
            "last_sweep_at": self.last_sweep_at,
            "last_sweep_duration_ms": self.last_sweep_duration_ms,
            "last_result": dict(self.last_result),
            "recent_actions": [item.to_payload() for item in self.actions[-RECOVERY_LOG_ACTIONS:]],
        }


class Supervisor:
    def __init__(self) -> None:
        self.owner_id = _owner_identity()
        self.metrics = SupervisorMetrics()
        self.breakers: dict[str, CircuitBreaker] = {}
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._sweep_lock = asyncio.Lock()
        self._background: set[asyncio.Task[Any]] = set()
        self._leader = False
        self._running = False
        self._consecutive_failures = 0
        self._sweep_counter = 0
        self._lease_rpc: bool | None = None
        self._retry_rpc: bool | None = None
        self._recovery_log: bool | None = None
        self._columns_supported: bool | None = None
        self._started_at = ""
        self._last_error = ""

    def _database(self) -> Any:
        return database_module.get_admin_database()

    def _database_configured(self) -> bool:
        settings = get_settings()
        return bool(settings.supabase_url and settings.supabase_service_role_key)

    def _breaker(self, name: str) -> CircuitBreaker:
        settings = get_settings()
        breaker = self.breakers.get(name)
        if breaker is None:
            breaker = CircuitBreaker(
                name=name,
                threshold=max(1, settings.supervisor_breaker_threshold),
                cooldown_seconds=max(5.0, settings.supervisor_breaker_cooldown_ms / 1000),
            )
            self.breakers[name] = breaker
        return breaker

    def _retain(self, coroutine: Awaitable[Any]) -> None:
        task = asyncio.ensure_future(coroutine)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def start(self) -> bool:
        if self._running:
            return True
        if not get_settings().supervisor_enabled:
            logger.info("supervisor_disabled")
            return False
        self._stop.clear()
        self._running = True
        self._started_at = iso()
        self._task = asyncio.create_task(self._guard(), name="agent-supervisor")
        logger.info("supervisor_started", extra={"owner_id": self.owner_id})
        return True

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._stop.set()
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except Exception as exc:
                logger.warning("supervisor_stop_error", extra={"error": str(exc)})
        for pending in list(self._background):
            pending.cancel()
        if self._background:
            await asyncio.gather(*self._background, return_exceptions=True)
        self._background.clear()
        await self._release_leadership()
        logger.info("supervisor_stopped", extra={"owner_id": self.owner_id})

    async def _guard(self) -> None:
        delay = GUARD_RESTART_DELAY_SECONDS
        while not self._stop.is_set():
            try:
                await self._loop()
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.metrics.guard_restarts += 1
                self._last_error = str(exc)[:RECOVERY_DETAIL_CHARS]
                logger.exception("supervisor_crashed")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                    return
                except asyncio.TimeoutError:
                    pass
                except asyncio.CancelledError:
                    raise
                delay = min(MAX_GUARD_RESTART_DELAY_SECONDS, delay * 2)

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._next_interval())
                return
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                raise
            try:
                result = await self._guarded_sweep()
                if result.get("ok"):
                    self._consecutive_failures = 0
                else:
                    self._consecutive_failures += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._consecutive_failures += 1
                self._last_error = str(exc)[:RECOVERY_DETAIL_CHARS]
                logger.exception("supervisor_sweep_failed")

    def _next_interval(self) -> float:
        settings = get_settings()
        base = max(1.0, settings.supervisor_interval_ms / 1000)
        if self._consecutive_failures:
            base = min(
                max(base, settings.supervisor_backoff_max_ms / 1000),
                base * (2.0 ** min(self._consecutive_failures, 8)),
            )
        if not self._database_configured():
            base = max(base, 60.0)
        return base * (0.85 + random.random() * 0.3)

    async def _guarded_sweep(self) -> dict[str, Any]:
        async with self._sweep_lock:
            return await self._sweep()

    async def heal_now(self) -> dict[str, Any]:
        if self._sweep_lock.locked():
            return {**self.metrics.last_result, "ok": True, "busy": True}
        return await self._guarded_sweep()

    async def _sweep(self) -> dict[str, Any]:
        started = time.monotonic()
        self._sweep_counter += 1
        self.metrics.sweeps += 1
        result: dict[str, Any] = {
            "ok": True,
            "sweep": self._sweep_counter,
            "leader": False,
            "database": self._database_configured(),
            "dispatched": 0,
            "requeued": 0,
            "dead_lettered": 0,
            "stopped_requests": 0,
            "browsers_stopped": 0,
            "actions": [],
            "probes": {},
            "errors": [],
        }
        if not self._database_configured():
            result["ok"] = False
            result["errors"].append("Nincs konfigurált Supabase adatbázis, a gyógyítás kimaradt.")
            self._record_result(result, started)
            return result

        result["probes"] = await self._probe_dependencies()
        result["leader"] = await self._acquire_leadership(result)
        if not result["leader"]:
            self._record_result(result, started)
            return result

        database_breaker = self._breaker("database")
        if not database_breaker.allow():
            database_breaker.record_skip()
            result["ok"] = False
            result["errors"].append("Az adatbázis áramköre nyitva van, a gyógyítás kimaradt.")
            self._record_result(result, started)
            await self._persist_leadership()
            return result

        sweep_started = time.monotonic()
        try:
            result.update(await self._heal_jobs(result))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            database_breaker.record_failure(str(exc))
            result["ok"] = False
            result["errors"].append(f"Munka gyógyítás: {exc}")
            logger.exception("supervisor_job_healing_failed")
        else:
            database_breaker.record_success(int((time.monotonic() - sweep_started) * 1000))
        try:
            result["browsers_stopped"] = await self._heal_orphan_browsers(result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result["ok"] = False
            result["errors"].append(f"Böngésző gyógyítás: {exc}")
            logger.exception("supervisor_browser_healing_failed")
        if self._sweep_counter % 20 == 0:
            try:
                pruned = await self._prune_recovery_log()
                result["pruned"] = pruned
                self.metrics.pruned_rows += pruned
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("supervisor_prune_failed", extra={"error": str(exc)})
        try:
            result["queue"] = await job_queue_summary()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("supervisor_summary_failed", extra={"error": str(exc)})
        self._record_result(result, started)
        await self._persist_leadership()
        return result

    def _record_result(self, result: dict[str, Any], started: float) -> None:
        duration = int((time.monotonic() - started) * 1000)
        result["duration_ms"] = duration
        result["owner_id"] = self.owner_id
        result["finished_at"] = iso()
        self.metrics.last_sweep_at = str(result["finished_at"])
        self.metrics.last_sweep_duration_ms = duration
        self.metrics.last_result = result
        self.metrics.dispatched += as_int(result.get("dispatched"))
        self.metrics.requeued += as_int(result.get("requeued"))
        self.metrics.dead_lettered += as_int(result.get("dead_lettered"))
        self.metrics.stopped_requests += as_int(result.get("stopped_requests"))
        self.metrics.browsers_stopped += as_int(result.get("browsers_stopped"))
        if not result.get("ok"):
            self.metrics.failed_sweeps += 1
            joined = "; ".join(as_str(item) for item in as_list(result.get("errors")))
            self._last_error = joined[:RECOVERY_DETAIL_CHARS]

    async def _probe_dependencies(self) -> dict[str, Any]:
        settings = get_settings()
        timeout = max(1.0, settings.supervisor_probe_timeout_ms / 1000)
        probes: dict[str, Probe] = {"database": self._probe_database}
        if settings.model_api_key:
            probes["model"] = self._probe_model
        if settings.instavm_api_key:
            probes["sandbox"] = self._probe_sandbox
        if settings.browser_use_api_key:
            probes["browser"] = self._probe_browser
        report: dict[str, Any] = {}
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(max(timeout, PROBE_TIMEOUT_SECONDS), connect=PROBE_CONNECT_SECONDS),
            follow_redirects=False,
        ) as client:
            for name, probe in probes.items():
                breaker = self._breaker(name)
                if not breaker.allow():
                    breaker.record_skip()
                    report[name] = {**breaker.snapshot(), "skipped_now": True}
                    continue
                try:
                    healthy, detail, latency = await asyncio.wait_for(
                        probe(client), timeout=timeout + 2.0
                    )
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError:
                    healthy = False
                    detail = "Az állapotellenőrzés időtúllépés miatt megszakadt."
                    latency = int(timeout * 1000)
                except Exception as exc:
                    healthy = False
                    detail = str(exc)
                    latency = 0
                if healthy:
                    breaker.record_success(latency)
                else:
                    breaker.record_failure(detail)
                    self.metrics.probe_failures += 1
                report[name] = breaker.snapshot()
        return report

    async def _probe_database(self, _client: httpx.AsyncClient) -> tuple[bool, str, int]:
        started = time.monotonic()
        try:
            database = self._database()
            await database.table(JOBS_TABLE).select("id").limit(1).execute()
        except DatabaseError as exc:
            return False, f"Adatbázis hiba: {exc.message}", self._elapsed(started)
        except Exception as exc:
            return False, f"Adatbázis hiba: {exc}", self._elapsed(started)
        return True, "", self._elapsed(started)

    @staticmethod
    def _elapsed(started: float) -> int:
        return int((time.monotonic() - started) * 1000)

    def _model_probe_url(self) -> str:
        base = get_settings().model_api_url.rstrip("/")
        suffix = "/responses"
        if base.endswith(suffix):
            base = base[: -len(suffix)]
        return f"{base}/models"

    async def _probe_http(
        self,
        client: httpx.AsyncClient,
        name: str,
        url: str,
        headers: dict[str, str],
        accept_below: int,
    ) -> tuple[bool, str, int]:
        started = time.monotonic()
        try:
            response = await client.get(url, headers=headers)
        except httpx.TimeoutException as exc:
            return False, f"{name} időtúllépés: {exc}", self._elapsed(started)
        except httpx.HTTPError as exc:
            return False, f"{name} hálózati hiba: {exc}", self._elapsed(started)
        latency = self._elapsed(started)
        if response.status_code >= accept_below:
            return False, f"{name} HTTP {response.status_code}", latency
        return True, "", latency

    async def _probe_model(self, client: httpx.AsyncClient) -> tuple[bool, str, int]:
        settings = get_settings()
        return await self._probe_http(
            client,
            "model",
            self._model_probe_url(),
            {
                "Authorization": f"Bearer {settings.model_api_key}",
                "Accept": "application/json",
            },
            500,
        )

    async def _probe_sandbox(self, client: httpx.AsyncClient) -> tuple[bool, str, int]:
        settings = get_settings()
        return await self._probe_http(
            client,
            "sandbox",
            f"{settings.instavm_url.rstrip('/')}/v1/vms",
            {"X-API-Key": settings.instavm_api_key, "Accept": "application/json"},
            500,
        )

    async def _probe_browser(self, client: httpx.AsyncClient) -> tuple[bool, str, int]:
        settings = get_settings()
        return await self._probe_http(
            client,
            "browser",
            cloud_browser.API_URL,
            {"X-Browser-Use-API-Key": settings.browser_use_api_key, "Accept": "application/json"},
            500,
        )

    async def _acquire_leadership(self, result: dict[str, Any]) -> bool:
        if self._lease_rpc is False:
            if not self._leader:
                self.metrics.leader_changes += 1
            self._leader = True
            result["leadership"] = "local"
            return True
        settings = get_settings()
        try:
            database = self._database()
            response = await (
                database.rpc(
                    "claim_supervisor_lease",
                    {
                        "_name": LEASE_NAME,
                        "_owner_id": self.owner_id,
                        "_lease_ms": max(5_000, settings.supervisor_leader_lease_ms),
                        "_state": self._leadership_state(),
                    },
                ).execute()
            )
        except DatabaseError as exc:
            if exc.status_code in (400, 404):
                self._lease_rpc = False
                self._leader = True
                result["leadership"] = "local"
                logger.warning("supervisor_lease_unavailable", extra={"error": str(exc)})
                return True
            self._leader = False
            result["leadership"] = "unavailable"
            result["ok"] = False
            result["errors"].append(f"Vezetőbérlet: {exc.message}")
            return False
        except Exception as exc:
            self._leader = False
            result["leadership"] = "unavailable"
            result["ok"] = False
            result["errors"].append(f"Vezetőbérlet: {exc}")
            return False
        row = response.maybe_single()
        acquired = as_bool(row.get("acquired")) if row else False
        owner = as_str(row.get("owner_id")) if row else ""
        self._lease_rpc = True
        if acquired and not self._leader:
            self.metrics.leader_changes += 1
        self._leader = acquired
        result["leadership"] = "leader" if acquired else "follower"
        result["leader_owner"] = owner
        return acquired

    def _leadership_state(self) -> dict[str, Any]:
        return {
            "owner_id": self.owner_id,
            "started_at": self._started_at,
            "sweeps": self.metrics.sweeps,
            "healed": self.metrics.requeued
            + self.metrics.dispatched
            + self.metrics.dead_lettered,
            "breakers": {name: breaker.state for name, breaker in self.breakers.items()},
            "updated_at": iso(),
        }

    async def _persist_leadership(self) -> None:
        if self._lease_rpc is not True or not self._leader:
            return
        try:
            database = self._database()
            await (
                database.table(LEASE_TABLE)
                .update(
                    {
                        "sweep_count": self.metrics.sweeps,
                        "healed_count": self.metrics.requeued
                        + self.metrics.dispatched
                        + self.metrics.dead_lettered
                        + self.metrics.browsers_stopped,
                        "state": self._leadership_state(),
                        "renewed_at": iso(),
                    }
                )
                .eq("name", LEASE_NAME)
                .eq("owner_id", self.owner_id)
                .execute()
            )
        except Exception as exc:
            logger.warning("supervisor_state_persist_failed", extra={"error": str(exc)})

    async def _release_leadership(self) -> None:
        was_leader = self._leader
        self._leader = False
        if self._lease_rpc is not True or not was_leader:
            return
        try:
            database = self._database()
            await database.rpc(
                "release_supervisor_lease", {"_name": LEASE_NAME, "_owner_id": self.owner_id}
            ).execute()
        except Exception as exc:
            logger.warning("supervisor_lease_release_failed", extra={"error": str(exc)})

    async def _columns_available(self) -> bool:
        if self._columns_supported is not None:
            return self._columns_supported
        try:
            database = self._database()
            await (
                database.table(JOBS_TABLE)
                .select("id", "attempts", "next_attempt_at", "last_error", "progress_at")
                .limit(1)
                .execute()
            )
        except DatabaseError as exc:
            if exc.status_code in (400, 404):
                self._columns_supported = False
                return False
            return False
        except Exception:
            return False
        self._columns_supported = True
        return True

    async def _heal_jobs(self, result: dict[str, Any]) -> dict[str, Any]:
        counts = {"dispatched": 0, "requeued": 0, "dead_lettered": 0, "stopped_requests": 0}
        extended = await self._columns_available()
        result["extended_columns"] = extended
        for job in await self._collect_due_jobs(extended):
            if counts["dispatched"] >= MAX_DISPATCH_PER_SWEEP:
                break
            job_id = as_str(job.get("id"))
            if not job_id:
                continue
            if await self._dispatch(job_id, result):
                counts["dispatched"] += 1
                self._log_action("dispatch", job_id, "Elárvult sorban álló munka újraindítva.", result)
        for job in await self._collect_expired_running(extended):
            job_id = as_str(job.get("id"))
            if not job_id:
                continue
            outcome = await self._recover_job(job_id, job, result)
            if outcome == "requeued":
                counts["requeued"] += 1
            elif outcome == "dead":
                counts["dead_lettered"] += 1
            elif outcome == "dispatched":
                counts["dispatched"] += 1
        counts["stopped_requests"] = await self._stop_stuck_jobs(result)
        return counts

    async def _collect_due_jobs(self, extended: bool) -> list[dict[str, Any]]:
        settings = get_settings()
        database = self._database()
        columns = "id, status, created_at, updated_at, stop_requested"
        if extended:
            columns = (
                "id, status, created_at, updated_at, stop_requested, "
                "attempts, next_attempt_at, last_error"
            )
        floor = iso(now() - timedelta(milliseconds=max(1_000, DUE_FLOOR_MS)))
        jobs: dict[str, dict[str, Any]] = {}
        try:
            response = await (
                database.table(JOBS_TABLE)
                .select(columns)
                .eq("status", "queued")
                .lt("updated_at", floor)
                .order("updated_at", desc=False, nullsfirst=True)
                .limit(settings.supervisor_batch_size)
                .execute()
            )
        except Exception as exc:
            logger.warning("supervisor_due_jobs_failed", extra={"error": str(exc)})
            return []
        for row in response.rows:
            job_id = as_str(row.get("id"))
            if job_id:
                jobs[job_id] = row
        if extended:
            try:
                pending = await (
                    database.table(JOBS_TABLE)
                    .select(columns)
                    .eq("status", "queued")
                    .lte("next_attempt_at", iso())
                    .order("next_attempt_at", desc=False, nullsfirst=True)
                    .limit(settings.supervisor_batch_size)
                    .execute()
                )
                for row in pending.rows:
                    job_id = as_str(row.get("id"))
                    if job_id:
                        jobs[job_id] = row
            except Exception as exc:
                logger.warning("supervisor_backoff_jobs_failed", extra={"error": str(exc)})
        stale_ms = max(DUE_FLOOR_MS, settings.supervisor_stale_queued_ms)
        ready: list[dict[str, Any]] = []
        for row in jobs.values():
            if as_bool(row.get("stop_requested")):
                continue
            if extended:
                scheduled = parse_iso(row.get("next_attempt_at"))
                if scheduled is not None:
                    if scheduled > now():
                        continue
                else:
                    updated_age = age_ms(row.get("updated_at"))
                    if updated_age is None or updated_age < stale_ms:
                        continue
            else:
                updated_age = age_ms(row.get("updated_at"))
                if updated_age is None:
                    updated_age = age_ms(row.get("created_at"))
                if updated_age is None or updated_age < stale_ms:
                    continue
            ready.append(row)
        ready.sort(key=lambda item: as_str(item.get("updated_at")) or as_str(item.get("created_at")))
        return ready

    async def _collect_expired_running(self, extended: bool) -> list[dict[str, Any]]:
        settings = get_settings()
        database = self._database()
        columns = "id, status, lease_expires_at, stop_requested, updated_at, started_at, created_at"
        if extended:
            columns = (
                "id, status, lease_expires_at, stop_requested, updated_at, started_at, created_at, "
                "attempts, next_attempt_at, last_error, progress_at"
            )
        try:
            response = await (
                database.table(JOBS_TABLE)
                .select(columns)
                .eq("status", "running")
                .lt("lease_expires_at", iso())
                .order("lease_expires_at", desc=False, nullsfirst=True)
                .limit(settings.supervisor_batch_size)
                .execute()
            )
            rows = list(response.rows)
        except Exception as exc:
            logger.warning("supervisor_expired_running_failed", extra={"error": str(exc)})
            return []
        try:
            orphaned = await (
                database.table(JOBS_TABLE)
                .select(columns)
                .eq("status", "running")
                .is_("lease_expires_at", None)
                .limit(settings.supervisor_batch_size)
                .execute()
            )
        except Exception as exc:
            logger.warning("supervisor_orphan_running_failed", extra={"error": str(exc)})
            return rows
        known = {as_str(row.get("id")) for row in rows}
        for row in orphaned.rows:
            if as_str(row.get("id")) not in known:
                rows.append(row)
        return rows

    async def _dispatch(self, job_id: str, result: dict[str, Any]) -> bool:
        try:
            await dispatch_chat_job_origin(job_id, supervisor_origin())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result["ok"] = False
            result["errors"].append(f"Újraküldés sikertelen ({job_id}): {exc}")
            logger.warning("supervisor_dispatch_failed", extra={"job_id": job_id, "error": str(exc)})
            return False
        return True

    async def _recover_job(self, job_id: str, job: dict[str, Any], result: dict[str, Any]) -> str:
        settings = get_settings()
        if as_bool(job.get("stop_requested")):
            await fail_chat_job(
                job_id,
                "CANCELLED",
                "A munka megszakítását kérték, és a futó példány már nem él.",
                outcome="cancelled",
            )
            self._log_action("cancel", job_id, "Megszakításra jelölt, elárvult munka lezárva.", result)
            return "dead"
        attempts = as_int(job.get("attempts"))
        detail = f"Lejárt a munkabérlet; kísérlet: {attempts + 1}/{settings.chat_job_max_attempts}."
        if self._retry_rpc is not False:
            outcome = await self._retry_via_rpc(job_id, detail, result)
            if outcome is not None:
                return outcome
        return await self._legacy_recover(job_id, job, attempts, detail, result)

    async def _retry_via_rpc(
        self, job_id: str, detail: str, result: dict[str, Any]
    ) -> str | None:
        settings = get_settings()
        try:
            database = self._database()
            response = await (
                database.rpc(
                    "claim_chat_job_retry",
                    {
                        "_job_id": job_id,
                        "_max_attempts": settings.chat_job_max_attempts,
                        "_retry_base_ms": settings.chat_job_retry_base_ms,
                        "_retry_max_ms": settings.chat_job_retry_max_ms,
                        "_error": detail,
                    },
                ).execute()
            )
        except DatabaseError as exc:
            if exc.status_code in (400, 404):
                self._retry_rpc = False
                logger.warning("supervisor_retry_rpc_unavailable", extra={"error": str(exc)})
                return None
            result["ok"] = False
            result["errors"].append(f"Újrapróbálkozás ({job_id}): {exc.message}")
            return "skipped"
        except Exception as exc:
            result["ok"] = False
            result["errors"].append(f"Újrapróbálkozás ({job_id}): {exc}")
            return "skipped"
        self._retry_rpc = True
        row = response.maybe_single() or {}
        status = as_str(row.get("status"))
        if as_bool(row.get("retry")):
            self._log_action("requeue", job_id, detail, result)
            if await self._dispatch(job_id, result):
                return "dispatched"
            return "requeued"
        if status in ("failed", "cancelled"):
            await fail_chat_job(
                job_id,
                "RECOVERY_EXHAUSTED",
                f"A munka nem fejeződött be, és az újrapróbálási keret elfogyott. {detail}",
                outcome=status,
            )
            self._log_action("dead_letter", job_id, detail, result)
            return "dead"
        return "skipped"

    async def _legacy_recover(
        self,
        job_id: str,
        job: dict[str, Any],
        attempts: int,
        detail: str,
        result: dict[str, Any],
    ) -> str:
        settings = get_settings()
        prior = await self._count_prior_requeues(job_id)
        total = attempts + prior
        created_age = age_ms(job.get("created_at"))
        if created_age is None:
            created_age = age_ms(job.get("updated_at")) or 0.0
        attempts_exhausted = prior > 0 and total + 1 > settings.chat_job_max_attempts
        lifetime_exhausted = created_age > JOB_LIFETIME_LIMIT_MS
        if attempts_exhausted or lifetime_exhausted:
            reason = (
                "A feladat teljes futásideje lejárt."
                if lifetime_exhausted
                else "Az újrapróbálási keret elfogyott."
            )
            await fail_chat_job(
                job_id,
                "RECOVERY_EXHAUSTED",
                f"A munka nem fejeződött be. {reason} {detail}",
                outcome="failed",
            )
            self._log_action("dead_letter", job_id, f"{reason} {detail}", result)
            return "dead"
        database = self._database()
        try:
            response = await (
                database.table(JOBS_TABLE)
                .update({"status": "queued", "lease_expires_at": None, "updated_at": iso()})
                .eq("id", job_id)
                .eq("status", "running")
                .execute()
            )
        except Exception as exc:
            result["ok"] = False
            result["errors"].append(f"Sorba állítás ({job_id}): {exc}")
            return "skipped"
        if not response.rows:
            return "skipped"
        self._log_action("requeue", job_id, detail, result)
        if await self._dispatch(job_id, result):
            return "dispatched"
        return "requeued"

    async def _count_prior_requeues(self, job_id: str) -> int:
        if self._recovery_log is False:
            return 0
        try:
            database = self._database()
            count = await (
                database.table(RECOVERY_TABLE)
                .select("id", count="exact", head=True)
                .eq("job_id", job_id)
                .in_("kind", ["requeue", "dispatch"])
                .count_exact()
            )
        except DatabaseError as exc:
            if exc.status_code in (400, 404):
                self._recovery_log = False
            return 0
        except Exception:
            return 0
        self._recovery_log = True
        return int(count)

    async def _stop_stuck_jobs(self, result: dict[str, Any]) -> int:
        if not await self._columns_available():
            return 0
        settings = get_settings()
        database = self._database()
        threshold_ms = max(60_000, settings.supervisor_stuck_progress_ms)
        try:
            response = await (
                database.table(JOBS_TABLE)
                .select("id, progress_at, updated_at, started_at, stop_requested")
                .eq("status", "running")
                .gte("lease_expires_at", iso())
                .eq("stop_requested", False)
                .limit(settings.supervisor_batch_size)
                .execute()
            )
            rows = list(response.rows)
        except Exception as exc:
            logger.warning("supervisor_stuck_scan_failed", extra={"error": str(exc)})
            return 0
        stopped = 0
        for row in rows:
            job_id = as_str(row.get("id"))
            if not job_id:
                continue
            progress_age = age_ms(row.get("progress_at"))
            if progress_age is None:
                progress_age = age_ms(row.get("started_at"))
            if progress_age is None:
                progress_age = age_ms(row.get("updated_at"))
            if progress_age is None or progress_age < threshold_ms:
                continue
            try:
                response = await (
                    database.table(JOBS_TABLE)
                    .update({"stop_requested": True, "updated_at": iso()})
                    .eq("id", job_id)
                    .eq("status", "running")
                    .eq("stop_requested", False)
                    .execute()
                )
            except Exception as exc:
                result["ok"] = False
                result["errors"].append(f"Elakadt munka ({job_id}): {exc}")
                continue
            if not response.rows:
                continue
            stopped += 1
            self._log_action(
                "stuck_stop",
                job_id,
                f"A végrehajtó hurok {int(progress_age / 1000)} másodperce nem haladt, "
                "ezért megszakításra jelöltem.",
                result,
            )
        return stopped

    async def _heal_orphan_browsers(self, result: dict[str, Any]) -> int:
        settings = get_settings()
        database = self._database()
        grace_ms = max(30_000, settings.supervisor_orphan_browser_ms)
        sessions = await self._live_browser_sessions(database)
        if not sessions:
            return 0
        job_ids = list(
            dict.fromkeys(
                as_str(session.get("job_id")) for session in sessions if as_str(session.get("job_id"))
            )
        )
        statuses: dict[str, str] = {}
        if job_ids:
            try:
                response = await (
                    database.table(JOBS_TABLE)
                    .select("id, status")
                    .in_("id", job_ids)
                    .limit(len(job_ids))
                    .execute()
                )
            except Exception as exc:
                logger.warning("supervisor_browser_jobs_failed", extra={"error": str(exc)})
                return 0
            for row in response.rows:
                statuses[as_str(row.get("id"))] = as_str(row.get("status"))
        stopped = 0
        for session in sessions:
            if stopped >= MAX_BROWSER_PER_SWEEP:
                break
            browser_id = as_str(session.get("browser_id"))
            job_id = as_str(session.get("job_id"))
            if not browser_id:
                continue
            job_status = statuses.get(job_id, "")
            orphaned = job_status in TERMINAL_STATUSES or (bool(job_id) and job_id not in statuses)
            if not orphaned:
                continue
            session_age = age_ms(session.get("updated_at"))
            if session_age is None:
                session_age = age_ms(session.get("created_at"))
            if session_age is not None and session_age < grace_ms:
                continue
            try:
                outcome = await cloud_browser.stop_cloud_browser(browser_id, None)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                result["ok"] = False
                result["errors"].append(f"Felhőböngésző ({browser_id}): {exc}")
                logger.warning(
                    "supervisor_browser_stop_failed",
                    extra={"browser_id": browser_id, "error": str(exc)},
                )
                continue
            try:
                await cloud_browser.mark_chat_browser_stopped(browser_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "supervisor_browser_mark_failed",
                    extra={"browser_id": browser_id, "error": str(exc)},
                )
            stopped += 1
            self._log_action(
                "browser_stop",
                job_id,
                f"Elárvult felhőböngésző leállítva: {browser_id} "
                f"({as_str(as_dict(outcome).get('status')) or 'ismeretlen'}).",
                result,
            )
        return stopped

    async def _live_browser_sessions(self, database: Any) -> list[dict[str, Any]]:
        settings = get_settings()
        columns = "id, job_id, browser_id, status, control_mode, created_at"
        try:
            response = await (
                database.table(SESSION_TABLE)
                .select(columns)
                .in_("status", list(BROWSER_LIVE_STATUSES))
                .order("updated_at", desc=False, nullsfirst=True)
                .limit(settings.supervisor_batch_size)
                .execute()
            )
            return list(response.rows)
        except DatabaseError as exc:
            if exc.status_code not in (400, 404):
                logger.warning("supervisor_browser_scan_failed", extra={"error": str(exc)})
                return []
        except Exception as exc:
            logger.warning("supervisor_browser_scan_failed", extra={"error": str(exc)})
            return []
        try:
            response = await (
                database.table(SESSION_TABLE)
                .select("id, job_id, browser_id, status")
                .eq("status", "active")
                .limit(settings.supervisor_batch_size)
                .execute()
            )
            return list(response.rows)
        except Exception as exc:
            logger.warning("supervisor_browser_scan_failed", extra={"error": str(exc)})
            return []

    async def _prune_recovery_log(self) -> int:
        if self._recovery_log is False:
            return 0
        settings = get_settings()
        database = self._database()
        limit = max(50, settings.supervisor_recovery_log_limit)
        try:
            response = await (
                database.table(RECOVERY_TABLE)
                .select("id")
                .order("id", desc=True)
                .range(limit, limit + 1)
                .execute()
            )
            rows = response.rows
            if not rows:
                self._recovery_log = True
                return 0
            cutoff = as_int(rows[0].get("id"))
            if cutoff <= 0:
                return 0
            deleted = await database.table(RECOVERY_TABLE).delete().lte("id", cutoff).execute()
            self._recovery_log = True
            return len(deleted.rows)
        except DatabaseError as exc:
            if exc.status_code in (400, 404):
                self._recovery_log = False
            return 0
        except Exception as exc:
            logger.warning("supervisor_prune_failed", extra={"error": str(exc)})
            return 0

    def _log_action(self, kind: str, job_id: str, detail: str, result: dict[str, Any]) -> None:
        action = self.metrics.add_action(kind, job_id, detail)
        as_list(result.get("actions")).append(action.to_payload())
        if self._recovery_log is False:
            return
        self._retain(self._persist_action(action))

    async def _persist_action(self, action: RecoveryAction) -> None:
        try:
            database = self._database()
            payload: dict[str, Any] = {
                "kind": action.kind,
                "owner_id": self.owner_id,
                "detail": {"message": action.detail, "created_at": action.created_at},
                "created_at": action.created_at,
            }
            if action.job_id:
                payload["job_id"] = action.job_id
            await database.table(RECOVERY_TABLE).insert(payload).execute()
            self._recovery_log = True
        except DatabaseError as exc:
            if exc.status_code in (400, 404):
                self._recovery_log = False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("supervisor_action_log_failed", extra={"error": str(exc)})

    def status(self) -> dict[str, Any]:
        settings = get_settings()
        started = parse_iso(self._started_at)
        uptime = 0.0
        if started is not None:
            uptime = max(0.0, (now() - started).total_seconds())
        return {
            "enabled": bool(settings.supervisor_enabled),
            "running": self._running,
            "leader": self._leader,
            "owner_id": self.owner_id,
            "started_at": self._started_at,
            "uptime_seconds": round(uptime, 3),
            "interval_ms": settings.supervisor_interval_ms,
            "next_interval_seconds": round(self._next_interval(), 3),
            "consecutive_failures": self._consecutive_failures,
            "last_error": self._last_error,
            "database_configured": self._database_configured(),
            "capabilities": {
                "lease_rpc": self._lease_rpc,
                "retry_rpc": self._retry_rpc,
                "recovery_log": self._recovery_log,
                "extended_columns": self._columns_supported,
            },
            "breakers": {name: breaker.snapshot() for name, breaker in self.breakers.items()},
            "metrics": self.metrics.snapshot(),
            "worker_origin": supervisor_origin(),
            "checked_at": iso(),
        }


supervisor = Supervisor()


async def start_supervisor() -> bool:
    return await supervisor.start()


async def stop_supervisor() -> None:
    await supervisor.stop()


def supervisor_status() -> dict[str, Any]:
    return supervisor.status()


async def heal_now() -> dict[str, Any]:
    return await supervisor.heal_now()
