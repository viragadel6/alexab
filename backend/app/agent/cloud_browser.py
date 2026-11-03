from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, TypeVar
from urllib.parse import quote, urlparse

import httpx

from app import database as database_module
from app.agent.browser_cdp import BrowserCdpError, browser_command, normalize_cdp_url
from app.config import get_settings
from app.database import DatabaseError, SupabaseDatabase

logger = logging.getLogger(__name__)

T = TypeVar("T")

API_URL = "https://api.browser-use.com/api/v4/browsers"
BROWSER_SCREEN_WIDTH = 1280
BROWSER_SCREEN_HEIGHT = 1029
MAX_TIMEOUT_MINUTES = 240
DEFAULT_TIMEOUT_MINUTES = 30
STOP_POLL_SECONDS = 2.0
STOP_DEADLINE_SECONDS = 90.0
STOP_REQUEST_DEADLINE_SECONDS = 75.0
RECORDING_UNAVAILABLE_GRACE_SECONDS = 20.0
CONTROL_POLL_SECONDS = 0.7
CONTROL_DEADLINE_SECONDS = 3600.0
CONTROL_ERROR_BACKOFF_SECONDS = 2.0
MAX_CONTROL_ERRORS = 5
CLEANUP_DEADLINE_SECONDS = 120.0
CREATION_CANCEL_GRACE_SECONDS = 5.0
MAX_PROTECTED_CANCELLATIONS = 3
MAX_RELEASE_ATTEMPTS = 5
REQUEST_TIMEOUT_SECONDS = 60.0
CONNECT_TIMEOUT_SECONDS = 10.0
CALL_DEADLINE_SECONDS = 90.0
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_CLIENT_SLOTS = 8
MAX_ERROR_CHARS = 500
MAX_BROWSER_ID_CHARS = 200
MAX_COUNTRY_CHARS = 8
DEFAULT_PROXY_COUNTRY = "us"
SESSION_TABLE = "chat_browser_sessions"
STOPPED_STATES = frozenset({"stopped", "stopping", "terminated", "closed", "failed", "error"})
CLIENT_LIMITS = httpx.Limits(
    max_connections=8, max_keepalive_connections=4, keepalive_expiry=60.0
)
_TIMEOUT_ERRORS: tuple[type[BaseException], ...] = (TimeoutError, asyncio.TimeoutError)

_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()

_client_slots: dict[int, "_ClientSlot"] = {}


class CloudBrowserError(RuntimeError):
    pass


class CloudBrowserCancelled(asyncio.CancelledError):
    pass


@dataclass
class _ClientSlot:
    loop: asyncio.AbstractEventLoop
    client: httpx.AsyncClient | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _api_key() -> str:
    configured: Any = get_settings().browser_use_api_key
    secret = (
        configured.get_secret_value() if hasattr(configured, "get_secret_value") else configured
    )
    key = str(secret or "").strip()
    if not key:
        raise CloudBrowserError("The BROWSER_USE_API_KEY is not set.")
    return key


def _log_dispose_result(future: concurrent.futures.Future[Any]) -> None:
    if future.cancelled():
        return
    failure = future.exception()
    if failure is not None:
        logger.warning("cloud_browser: az elavult HTTP kliens lezárása nem sikerült: %r", failure)


async def _dispose_client(
    client: httpx.AsyncClient, owner: asyncio.AbstractEventLoop | None
) -> None:
    if client.is_closed:
        return
    current = asyncio.get_running_loop()
    if (
        owner is not None
        and owner is not current
        and not owner.is_closed()
        and owner.is_running()
    ):
        try:
            future = asyncio.run_coroutine_threadsafe(client.aclose(), owner)
        except RuntimeError as exc:
            logger.warning(
                "cloud_browser: az elavult HTTP kliens lezárása nem ütemezhető: %r", exc
            )
            return
        future.add_done_callback(_log_dispose_result)
        return
    try:
        await client.aclose()
    except Exception as exc:
        logger.warning("cloud_browser: az elavult HTTP kliens lezárása nem sikerült: %r", exc)


async def _client_slot() -> "_ClientSlot":
    loop = asyncio.get_running_loop()
    key = id(loop)
    slot = _client_slots.get(key)
    if slot is not None and slot.loop is loop:
        return slot
    discarded: list[_ClientSlot] = []
    if slot is not None:
        discarded.append(slot)
    slot = _ClientSlot(loop=loop)
    _client_slots[key] = slot
    if len(_client_slots) > MAX_CLIENT_SLOTS:
        for stale_key in [
            item
            for item, value in _client_slots.items()
            if item != key and value.loop.is_closed()
        ]:
            stale = _client_slots.pop(stale_key, None)
            if stale is not None:
                discarded.append(stale)
    if len(_client_slots) > MAX_CLIENT_SLOTS:
        overflow = len(_client_slots) - MAX_CLIENT_SLOTS
        for stale_key in [item for item in _client_slots if item != key][:overflow]:
            stale = _client_slots.pop(stale_key, None)
            if stale is not None:
                discarded.append(stale)
    for stale in discarded:
        stale_client = stale.client
        stale.client = None
        if stale_client is not None and not stale_client.is_closed:
            logger.warning("cloud_browser: elavult HTTP kliens eldobva.")
            await _dispose_client(stale_client, stale.loop)
    return slot


async def get_client() -> httpx.AsyncClient:
    slot = await _client_slot()
    client = slot.client
    if client is not None and not client.is_closed:
        return client
    async with slot.lock:
        current = slot.client
        if current is not None and not current.is_closed:
            return current
        if current is not None:
            logger.warning("cloud_browser: bezárt HTTP kliens újra cserélve.")
        slot.client = httpx.AsyncClient(
            timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS),
            limits=CLIENT_LIMITS,
            follow_redirects=False,
        )
        return slot.client


async def close_client() -> None:
    slot = await _client_slot()
    async with slot.lock:
        client = slot.client
        slot.client = None
        if client is not None and not client.is_closed:
            await client.aclose()


def _raise_if_cancelled(cancel: asyncio.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise CloudBrowserCancelled()


def _clear_cancellation_requests() -> None:
    current = asyncio.current_task()
    if current is None:
        return
    uncancel = getattr(current, "uncancel", None)
    cancelling = getattr(current, "cancelling", None)
    if not callable(uncancel) or not callable(cancelling):
        return
    while cancelling() > 0:
        uncancel()


def _reap_exception(task: asyncio.Future[Any]) -> None:
    if task.done() and not task.cancelled():
        task.exception()


async def _cancel_and_reap(tasks: list[asyncio.Future[Any]]) -> None:
    pending = [task for task in tasks if not task.done()]
    for task in pending:
        task.cancel()
    try:
        if pending:
            await asyncio.wait(pending)
    finally:
        for task in tasks:
            if task.done():
                _reap_exception(task)
            else:
                task.add_done_callback(_reap_exception)


async def _reap(task: asyncio.Task[Any], label: str) -> None:
    await asyncio.wait({task})
    if task.cancelled():
        return
    failure = task.exception()
    if failure is not None:
        logger.warning("cloud_browser: %s hibával ért véget: %r", label, failure)


async def _sleep_unless_cancelled(seconds: float, cancel: asyncio.Event | None) -> None:
    _raise_if_cancelled(cancel)
    delay = max(0.0, seconds)
    if cancel is None:
        if delay > 0:
            await asyncio.sleep(delay)
        return
    if delay <= 0:
        return
    try:
        await asyncio.wait_for(cancel.wait(), timeout=delay)
    except _TIMEOUT_ERRORS:
        return
    _raise_if_cancelled(cancel)


def _retain_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    task: asyncio.Task[Any] = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_release_task)
    return task


def _release_task(task: asyncio.Task[Any]) -> None:
    _BACKGROUND_TASKS.discard(task)
    if task.cancelled():
        return
    failure = task.exception()
    if failure is not None:
        logger.warning("cloud_browser: háttérfeladat hibával ért véget: %r", failure)


def _normalized_browser_id(browser_id: Any) -> str:
    candidate = "" if browser_id is None else str(browser_id).strip()
    if not candidate:
        raise CloudBrowserError("The browser identifier is missing.")
    if len(candidate) > MAX_BROWSER_ID_CHARS:
        raise CloudBrowserError("The browser identifier is too long.")
    if any(character in candidate for character in "/?#%\\"):
        raise CloudBrowserError("The browser identifier is invalid.")
    return candidate


def _browser_path(browser_id: Any) -> str:
    return "/" + quote(_normalized_browser_id(browser_id), safe="")


def _normalized_job_id(job_id: Any) -> str:
    candidate = "" if job_id is None else str(job_id).strip()
    if not candidate:
        raise CloudBrowserError("The job identifier is missing.")
    return candidate


def _timeout_minutes(timeout_minutes: Any) -> int:
    try:
        value = int(timeout_minutes)
    except (TypeError, ValueError, OverflowError):
        logger.warning("cloud_browser: érvénytelen időkorlát: %r", timeout_minutes)
        return DEFAULT_TIMEOUT_MINUTES
    if value <= 0:
        logger.warning("cloud_browser: érvénytelen időkorlát: %r", timeout_minutes)
        return DEFAULT_TIMEOUT_MINUTES
    return min(MAX_TIMEOUT_MINUTES, value)


def _proxy_country(country_code: Any) -> str | None:
    if country_code is None:
        return None
    raw = str(country_code)
    country = raw.strip().lower()
    if (
        not country
        or len(country) > MAX_COUNTRY_CHARS
        or not country.isascii()
        or not country.isalpha()
    ):
        logger.warning("cloud_browser: érvénytelen proxy országkód: %r", country_code)
        return DEFAULT_PROXY_COUNTRY
    return country


def _live_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    parsed = urlparse(candidate)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        logger.warning("cloud_browser: érvénytelen élő nézet cím: %.200s", candidate)
        return None
    return candidate


def _cdp_url(value: Any) -> str:
    if not isinstance(value, str):
        raise CloudBrowserError("The browser did not return a CDP URL.")
    try:
        return normalize_cdp_url(value)
    except BrowserCdpError as exc:
        raise CloudBrowserError("The browser returned an invalid CDP URL.") from exc


async def _discard_result(
    task: asyncio.Future[T], discard: Callable[[T], Awaitable[Any]] | None
) -> None:
    if discard is None or not task.done() or task.cancelled():
        return
    if task.exception() is not None:
        return
    try:
        await discard(task.result())
    except Exception as exc:
        logger.warning("cloud_browser: az eldobott eredmény lezárása nem sikerült: %r", exc)


async def _await_cancellable(
    awaitable: Awaitable[T],
    cancel: asyncio.Event | None,
    discard: Callable[[T], Awaitable[Any]] | None = None,
) -> T:
    task: asyncio.Future[T] = asyncio.ensure_future(awaitable)
    waiter: asyncio.Future[Any] | None = (
        asyncio.ensure_future(cancel.wait()) if cancel is not None else None
    )
    watchers: list[asyncio.Future[Any]] = [task]
    if waiter is not None:
        watchers.append(waiter)
    delivered = False
    try:
        if waiter is None or cancel is None:
            value = await task
            delivered = True
            return value
        done, _pending = await asyncio.wait(
            set(watchers), return_when=asyncio.FIRST_COMPLETED
        )
        if waiter in done or cancel.is_set():
            raise CloudBrowserCancelled()
        value = task.result()
        delivered = True
        return value
    finally:
        await _cancel_and_reap(watchers)
        if not delivered:
            await _discard_result(task, discard)


async def _close_response(response: httpx.Response) -> None:
    await response.aclose()


async def _next_chunk(iterator: AsyncIterator[bytes]) -> bytes | None:
    try:
        return await iterator.__anext__()
    except StopAsyncIteration:
        return None


async def _read_bounded(response: httpx.Response, cancel: asyncio.Event | None) -> bytes:
    _raise_if_cancelled(cancel)
    chunks: list[bytes] = []
    total = 0
    iterator: AsyncIterator[bytes] = response.aiter_bytes().__aiter__()
    try:
        while True:
            chunk = await _await_cancellable(_next_chunk(iterator), cancel)
            if chunk is None:
                break
            _raise_if_cancelled(cancel)
            total += len(chunk)
            if total > MAX_RESPONSE_BYTES:
                raise CloudBrowserError(
                    f"Browser Use: the response exceeds {MAX_RESPONSE_BYTES} bytes."
                )
            chunks.append(chunk)
    finally:
        closer = getattr(iterator, "aclose", None)
        if callable(closer):
            try:
                await closer()
            except Exception as exc:
                logger.debug("cloud_browser: a válaszfolyam lezárása nem sikerült: %r", exc)
    _raise_if_cancelled(cancel)
    return b"".join(chunks)


async def _request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    headers: dict[str, str],
    body: Any,
    timeout_seconds: float,
    cancel: asyncio.Event | None,
) -> dict[str, Any]:
    timeout = httpx.Timeout(timeout_seconds, connect=CONNECT_TIMEOUT_SECONDS)
    request = client.build_request(
        method, f"{API_URL}{path}", headers=headers, json=body, timeout=timeout
    )
    response = await _await_cancellable(
        client.send(request, stream=True), cancel, _close_response
    )
    try:
        content = await _read_bounded(response, cancel)
    finally:
        await response.aclose()
    text = content.decode("utf-8", errors="replace")
    status_code = response.status_code
    if status_code < 200 or status_code >= 300:
        raise CloudBrowserError(f"Browser Use ({status_code}): {text[:MAX_ERROR_CHARS]}")
    if not text.strip():
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CloudBrowserError("Browser Use: invalid JSON response.") from exc
    if not isinstance(parsed, dict):
        raise CloudBrowserError("Browser Use: invalid response object.")
    return parsed


async def _call(
    path: str,
    method: str,
    body: Any,
    cancel: asyncio.Event | None,
    timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    _raise_if_cancelled(cancel)
    headers = {
        "X-Browser-Use-API-Key": _api_key(),
        "Accept": "application/json",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    client = await get_client()
    deadline = max(1.0, min(CALL_DEADLINE_SECONDS, timeout_seconds + CONNECT_TIMEOUT_SECONDS + 5.0))
    try:
        return await asyncio.wait_for(
            _request(client, method, path, headers, body, timeout_seconds, cancel),
            timeout=deadline,
        )
    except asyncio.CancelledError as exc:
        _raise_if_cancelled(cancel)
        raise exc
    except CloudBrowserError as exc:
        _raise_if_cancelled(cancel)
        raise exc
    except _TIMEOUT_ERRORS as exc:
        _raise_if_cancelled(cancel)
        raise CloudBrowserError(
            f"Browser Use: the {method} {path or '/'} request exceeded its deadline."
        ) from exc
    except httpx.HTTPError as exc:
        _raise_if_cancelled(cancel)
        detail = str(exc) or exc.__class__.__name__
        raise CloudBrowserError(f"Browser Use: {detail}") from exc
    except (httpx.InvalidURL, ValueError, OSError) as exc:
        _raise_if_cancelled(cancel)
        detail = str(exc) or exc.__class__.__name__
        raise CloudBrowserError(f"Browser Use: {detail}") from exc


def _failure(error: BaseException) -> dict[str, Any]:
    message = str(error) or "The cloud browser stopped with an error."
    return {
        "status": "error",
        "result": {"code": "CLOUD_BROWSER_FAILED", "message": message},
    }


def _stopped_payload(
    browser_id: str,
    recording_url: str | None,
    confirmed: bool,
    message: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "browser_id": browser_id,
        "stopped": True,
        "stop_confirmed": confirmed,
        "recording_url": recording_url,
    }
    if message:
        result["message"] = message
    return {"status": "completed", "result": result}


async def _await_protected(
    task: asyncio.Task[Any], deadline_seconds: float, label: str
) -> asyncio.CancelledError | None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, deadline_seconds)
    cancellation: asyncio.CancelledError | None = None
    absorbed = 0
    while not task.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            logger.warning("cloud_browser: %s nem fejeződött be a határidőn belül.", label)
            break
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except _TIMEOUT_ERRORS:
            if task.done():
                break
            logger.warning("cloud_browser: %s határidőn belül nem fejeződött be.", label)
            break
        except asyncio.CancelledError as exc:
            if task.cancelled():
                break
            absorbed += 1
            if absorbed > MAX_PROTECTED_CANCELLATIONS:
                logger.warning("cloud_browser: %s várakozása megszakadt.", label)
                raise
            if cancellation is None:
                cancellation = exc
        except Exception:
            break
    if task.cancelled():
        logger.warning("cloud_browser: %s megszakítva ért véget.", label)
        if cancellation is None:
            raise CloudBrowserError(f"The protected operation was cancelled: {label}")
        return cancellation
    if task.done():
        failure = task.exception()
        if failure is not None:
            logger.warning("cloud_browser: %s hibával ért véget: %s", label, failure)
            if cancellation is None:
                raise failure
    return cancellation


async def _stop_browser_quietly(browser_id: Any) -> bool:
    try:
        path = _browser_path(browser_id)
    except CloudBrowserError as exc:
        logger.warning("cloud_browser: érvénytelen böngészőazonosító a leállításhoz: %s", exc)
        return False
    stop_task: asyncio.Task[dict[str, Any]] = _retain_task(
        _call(path, "PATCH", {"action": "stop"}, None)
    )
    try:
        interruption = await _await_protected(
            stop_task, STOP_REQUEST_DEADLINE_SECONDS, "takarítási leállítási kérés"
        )
    except asyncio.CancelledError:
        _clear_cancellation_requests()
        logger.warning(
            "cloud_browser: a %s böngésző leállításának várakozása megszakadt; "
            "a kérés a háttérben folytatódik.",
            path,
        )
        return False
    except Exception as exc:
        logger.warning("cloud_browser: a %s böngésző leállítása nem sikerült: %s", path, exc)
        return False
    if interruption is not None:
        _clear_cancellation_requests()
    if not stop_task.done():
        logger.warning(
            "cloud_browser: a %s böngésző leállítása nem fejeződött be a határidőn belül.", path
        )
        return False
    if stop_task.cancelled():
        logger.warning("cloud_browser: a %s böngésző leállítási kérése megszakadt.", path)
        return False
    failure = stop_task.exception()
    if failure is not None:
        logger.warning("cloud_browser: a %s böngésző leállítása nem sikerült: %s", path, failure)
        return False
    return True


def _schedule_late_cleanup(creation_task: asyncio.Task[dict[str, Any]]) -> None:
    if creation_task.cancelled():
        return
    if creation_task.exception() is not None:
        return
    try:
        _retain_task(_cleanup_created_browser(creation_task))
    except RuntimeError as exc:
        logger.error(
            "cloud_browser: a késői takarítás nem ütemezhető, a böngészőt a szerver oldali "
            "időkorlát állítja le: %r",
            exc,
        )


async def _cleanup_created_browser(creation_task: asyncio.Task[dict[str, Any]]) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + CLEANUP_DEADLINE_SECONDS
    absorbed = 0
    while not creation_task.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        try:
            await asyncio.wait({creation_task}, timeout=remaining)
        except asyncio.CancelledError:
            absorbed += 1
            _clear_cancellation_requests()
            if absorbed > MAX_PROTECTED_CANCELLATIONS:
                logger.warning(
                    "cloud_browser: a takarítás várakozása többször megszakadt; "
                    "a létrehozás befejeződése után késői takarítás következik."
                )
                break
    if not creation_task.done():
        logger.warning(
            "cloud_browser: a létrehozás nem fejeződött be a takarítási határidőn belül; "
            "a befejeződés után késői leállítás következik."
        )
        creation_task.add_done_callback(_schedule_late_cleanup)
        return
    if creation_task.cancelled():
        logger.warning(
            "cloud_browser: a létrehozás megszakadt, az azonosító ismeretlen; "
            "a szerver oldali időkorlát állítja le a böngészőt."
        )
        return
    failure = creation_task.exception()
    if failure is not None:
        logger.warning(
            "cloud_browser: a létrehozás hibával ért véget, nincs leállítandó azonosító: %s",
            failure,
        )
        return
    browser = creation_task.result()
    record = browser if isinstance(browser, dict) else {}
    raw_id = record.get("id")
    browser_id = str(raw_id or "").strip()
    if not browser_id:
        logger.warning(
            "cloud_browser: a létrehozás válasza nem tartalmaz azonosítót; a böngészőt "
            "a szerver oldali időkorlát állítja le."
        )
        return
    if not await _stop_browser_quietly(browser_id):
        logger.error("cloud_browser: a %s böngészőt nem sikerült leállítani.", browser_id)


async def _await_creation(
    creation_task: asyncio.Task[dict[str, Any]], cancel: asyncio.Event | None
) -> dict[str, Any]:
    if cancel is None:
        try:
            return await creation_task
        finally:
            _reap_exception(creation_task)
    waiter: asyncio.Task[bool] = asyncio.create_task(cancel.wait())
    watchers: set[asyncio.Task[Any]] = {creation_task, waiter}
    try:
        done, _pending = await asyncio.wait(watchers, return_when=asyncio.FIRST_COMPLETED)
        if waiter in done or cancel.is_set():
            if not creation_task.done():
                logger.warning(
                    "cloud_browser: megszakítás a böngésző létrehozása közben, "
                    "%.1f másodperc türelmi idő.",
                    CREATION_CANCEL_GRACE_SECONDS,
                )
                await asyncio.wait({creation_task}, timeout=CREATION_CANCEL_GRACE_SECONDS)
            raise CloudBrowserCancelled()
        return creation_task.result()
    finally:
        waiter.cancel()
        await _reap(waiter, "megszakítás figyelő")


async def start_cloud_browser(
    country_code: Any,
    timeout_minutes: Any,
    cancel: asyncio.Event | None,
    enable_recording: bool = True,
) -> dict[str, Any]:
    creation_task: asyncio.Task[dict[str, Any]] | None = None
    try:
        _raise_if_cancelled(cancel)
        proxy_country = _proxy_country(country_code)
        timeout = _timeout_minutes(timeout_minutes)
        creation_task = asyncio.create_task(
            _call(
                "",
                "POST",
                {
                    "proxyCountryCode": proxy_country,
                    "timeout": timeout,
                    "allowResizing": True,
                    "enableRecording": enable_recording,
                    "browserScreenWidth": BROWSER_SCREEN_WIDTH,
                    "browserScreenHeight": BROWSER_SCREEN_HEIGHT,
                },
                None,
            )
        )
        browser = await _await_creation(creation_task, cancel)
        _raise_if_cancelled(cancel)
        raw_id = browser.get("id") if isinstance(browser, dict) else None
        browser_id = str(raw_id or "").strip() if raw_id is not None else ""
        if not browser_id:
            raise CloudBrowserError("The browser did not return an identifier.")
        cdp_url = _cdp_url(browser.get("cdpUrl") if isinstance(browser, dict) else None)
        return {
            "status": "completed",
            "result": {
                "browser_id": browser_id,
                "cdp_url": cdp_url,
                "live_url": _live_url(browser.get("liveUrl") if isinstance(browser, dict) else None),
                "timeout_minutes": timeout,
                "screen_width": BROWSER_SCREEN_WIDTH,
                "screen_height": BROWSER_SCREEN_HEIGHT,
                "usage": (
                    "Use the cloud_browser_navigate, cloud_browser_inspect, "
                    "cloud_browser_click, cloud_browser_type, and cloud_browser_press_key "
                    "tools to control this browser directly. Call cloud_browser_stop when done."
                ),
            },
        }
    except (asyncio.CancelledError, Exception) as exc:
        if creation_task is not None:
            cleanup_task = _retain_task(_cleanup_created_browser(creation_task))
            try:
                cancellation = await _await_protected(
                    cleanup_task, CLEANUP_DEADLINE_SECONDS, "felhőböngésző takarítás"
                )
            except asyncio.CancelledError:
                raise
            except Exception as cleanup_exc:
                logger.warning(
                    "cloud_browser: a felhőböngésző takarítása hibával ért véget: %r",
                    cleanup_exc,
                )
                cancellation = None
            if cancellation is not None and isinstance(exc, Exception):
                raise cancellation
        if isinstance(exc, Exception):
            _raise_if_cancelled(cancel)
            return _failure(exc)
        raise


async def _await_recording(
    path: str, cancel: asyncio.Event | None
) -> tuple[str | None, bool]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + STOP_DEADLINE_SECONDS
    recording_rejected = False
    confirmed = False
    while True:
        _raise_if_cancelled(cancel)
        remaining = deadline - loop.time()
        if remaining <= 0:
            return None, confirmed
        state = await _call(
            path,
            "GET",
            None,
            cancel,
            timeout_seconds=min(REQUEST_TIMEOUT_SECONDS, max(5.0, remaining)),
        )
        recording = state.get("recordingUrl")
        if isinstance(recording, str) and recording.strip():
            return recording.strip(), True
        confirmed = confirmed or _stopped_state(state)
        if state.get("recordingAvailable") is False and not recording_rejected:
            recording_rejected = True
            deadline = min(deadline, loop.time() + RECORDING_UNAVAILABLE_GRACE_SECONDS)
        remaining = deadline - loop.time()
        if remaining <= 0:
            return None, confirmed
        await _sleep_unless_cancelled(min(STOP_POLL_SECONDS, remaining), cancel)


def _stopped_state(state: dict[str, Any]) -> bool:
    for key in ("state", "status"):
        value = state.get(key)
        if isinstance(value, str) and value.strip().lower() in STOPPED_STATES:
            return True
    return False


async def stop_cloud_browser(
    browser_id: Any,
    cancel: asyncio.Event | None,
    retrieve_recording: bool = True,
) -> dict[str, Any]:
    try:
        normalized_id = _normalized_browser_id(browser_id)
    except CloudBrowserError as exc:
        return _failure(exc)
    path = _browser_path(normalized_id)
    try:
        stop_task: asyncio.Task[dict[str, Any]] = _retain_task(
            _call(path, "PATCH", {"action": "stop"}, None)
        )
        interruption = await _await_protected(
            stop_task, STOP_REQUEST_DEADLINE_SECONDS, "leállítási kérés"
        )
        if not stop_task.done():
            raise CloudBrowserError("The stop request did not finish in time.")
        if stop_task.cancelled():
            raise CloudBrowserError("The stop request was cancelled.")
        failure = stop_task.exception()
        if failure is not None:
            raise failure
        if interruption is not None:
            raise interruption
    except asyncio.CancelledError as exc:
        if isinstance(exc, CloudBrowserCancelled) or (cancel is not None and cancel.is_set()):
            return _stopped_payload(
                normalized_id,
                None,
                False,
                "The stop request was interrupted; the browser is being stopped.",
            )
        raise
    except CloudBrowserError as exc:
        return _failure(exc)
    except Exception as exc:
        logger.warning("cloud_browser: a leállítási kérés váratlan hibája: %r", exc)
        return _failure(exc)
    if not retrieve_recording:
        return _stopped_payload(
            normalized_id,
            None,
            False,
            "The browser stop request was accepted; recording retrieval was disabled.",
        )
    recording_url: str | None = None
    confirmed = False
    recording_error: str | None = None
    try:
        recording_url, confirmed = await _await_recording(path, cancel)
    except asyncio.CancelledError as exc:
        if not isinstance(exc, CloudBrowserCancelled) and not (
            cancel is not None and cancel.is_set()
        ):
            raise
        return _stopped_payload(
            normalized_id,
            None,
            confirmed,
            "The browser was stopped, the recording is still being processed.",
        )
    except CloudBrowserError as exc:
        recording_error = str(exc) or exc.__class__.__name__
        logger.warning(
            "cloud_browser: a %s felvétel lekérdezése nem sikerült: %s", normalized_id, exc
        )
    except Exception as exc:
        recording_error = str(exc) or exc.__class__.__name__
        logger.warning(
            "cloud_browser: a %s felvétel lekérdezése váratlanul megszakadt: %r",
            normalized_id,
            exc,
        )
    message = (
        None
        if recording_url is not None
        else "The browser was stopped, the recording is still being processed."
    )
    payload = _stopped_payload(normalized_id, recording_url, confirmed, message)
    if recording_error is not None:
        payload["result"]["recording_error"] = recording_error
    return payload


def _response_rows(response: Any) -> list[dict[str, Any]]:
    data = getattr(response, "data", None)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _first_row(response: Any) -> dict[str, Any] | None:
    rows = _response_rows(response)
    if not rows:
        return None
    return rows[0]


async def _find_session_id(database: SupabaseDatabase, browser_id: str) -> str | None:
    try:
        response = await (
            database.table(SESSION_TABLE)
            .select("id")
            .eq("browser_id", browser_id)
            .limit(1)
            .execute()
        )
    except DatabaseError as exc:
        logger.warning("cloud_browser: a munkamenet kikeresése nem sikerült: %s", exc)
        return None
    row = _first_row(response)
    if row is None:
        return None
    session_id = row.get("id")
    if session_id is None:
        return None
    candidate = str(session_id).strip()
    return candidate or None


async def register_chat_browser(job_id: Any, result: Any) -> str:
    raw_job_id = _normalized_job_id(job_id)
    record = result if isinstance(result, dict) else None
    if record is None:
        raise CloudBrowserError("Saving the browser session failed: the result is not an object.")
    nested = record.get("result")
    if "browser_id" not in record and isinstance(nested, dict):
        status = record.get("status")
        if isinstance(status, str) and status.strip().lower() == "error":
            detail = nested.get("message") or "the browser could not be started."
            raise CloudBrowserError(f"Saving the browser session failed: {detail}")
        record = nested
    raw_browser_id = record.get("browser_id")
    browser_id = str(raw_browser_id).strip() if raw_browser_id is not None else ""
    if not browser_id:
        raise CloudBrowserError(
            "Saving the browser session failed: the browser identifier is missing."
        )
    cdp_url = _cdp_url(record.get("cdp_url"))
    live_url = _live_url(record.get("live_url"))
    database = database_module.get_admin_database()
    existing = await _find_session_id(database, browser_id)
    if existing is not None:
        logger.info("cloud_browser: a %s böngésző munkamenet már létezik.", browser_id)
        return existing
    try:
        response = await (
            database.table(SESSION_TABLE)
            .insert(
                {
                    "job_id": raw_job_id,
                    "browser_id": browser_id,
                    "cdp_url": cdp_url,
                    "live_url": live_url,
                    "status": "active",
                    "control_mode": "agent",
                    "agent_busy": True,
                }
            )
            .execute()
        )
    except DatabaseError as exc:
        recovered = await _find_session_id(database, browser_id)
        if recovered is not None:
            return recovered
        raise CloudBrowserError(f"Saving the browser session failed: {exc}") from exc
    row = _first_row(response)
    if row is None:
        recovered = await _find_session_id(database, browser_id)
        if recovered is not None:
            return recovered
        raise CloudBrowserError("Saving the browser session failed: no row was returned.")
    session_id = row.get("id")
    if session_id is None or not str(session_id).strip():
        raise CloudBrowserError("Saving the browser session failed: the session id is missing.")
    return str(session_id).strip()


async def chat_browser_cdp_url(job_id: Any, browser_id: Any) -> str:
    raw_job_id = _normalized_job_id(job_id)
    normalized_id = _normalized_browser_id(browser_id)
    database = database_module.get_admin_database()
    try:
        response = await (
            database.table(SESSION_TABLE)
            .select("cdp_url")
            .eq("job_id", raw_job_id)
            .eq("browser_id", normalized_id)
            .eq("status", "active")
            .limit(1)
            .execute()
        )
    except DatabaseError as exc:
        raise CloudBrowserError(f"Reading the active browser session failed: {exc}") from exc
    row = _first_row(response)
    if row is None:
        raise CloudBrowserError("The active browser session does not belong to this task.")
    return _cdp_url(row.get("cdp_url"))


async def browse_url(url: str, cancel: asyncio.Event | None) -> dict[str, Any]:
    browser_id: str | None = None
    try:
        started = await start_cloud_browser("us", 1, cancel, enable_recording=False)
        if started.get("status") != "completed":
            return started
        result = started.get("result")
        record = result if isinstance(result, dict) else {}
        browser_id = str(record.get("browser_id") or "").strip()
        if not browser_id:
            raise CloudBrowserError("The browser did not return an identifier.")
        cdp_url = _cdp_url(record.get("cdp_url"))
        await browser_command(cdp_url, "navigate", {"url": url})
        page = await browser_command(cdp_url, "inspect")
        payload = {"status": "completed", "result": page}
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _raise_if_cancelled(cancel)
        payload = _failure(exc)
    finally:
        if browser_id:
            cleanup_task = _retain_task(
                stop_cloud_browser(browser_id, None, retrieve_recording=False)
            )
            try:
                interruption = await _await_protected(
                    cleanup_task, CLEANUP_DEADLINE_SECONDS, "temporary browser cleanup"
                )
                if interruption is not None:
                    raise interruption
                cleanup_result = cleanup_task.result()
                if cleanup_result.get("status") != "completed":
                    logger.warning(
                        "cloud_browser: ideiglenes böngésző lezárása sikertelen: %r",
                        cleanup_result.get("result"),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("cloud_browser: ideiglenes böngésző lezárása sikertelen: %r", exc)
    return payload


def _claimed(data: Any) -> bool:
    if data is True:
        return True
    return isinstance(data, list) and len(data) == 1 and data[0] is True


async def _release_claim(job_id: str) -> None:
    try:
        await set_browser_agent_busy(job_id, False)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("cloud_browser: az irányítási igény felszabadítása nem sikerült: %s", exc)


async def wait_for_browser_control(job_id: Any, cancel: asyncio.Event | None) -> None:
    raw_job_id = _normalized_job_id(job_id)
    database = database_module.get_admin_database()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + CONTROL_DEADLINE_SECONDS
    consecutive_errors = 0
    while True:
        _raise_if_cancelled(cancel)
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise CloudBrowserError("Waiting for the browser control timed out.")
        try:
            response = await (
                database.rpc("claim_chat_browser_step", {"_job_id": raw_job_id}).execute()
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            consecutive_errors += 1
            if consecutive_errors > MAX_CONTROL_ERRORS:
                raise CloudBrowserError(
                    f"Waiting for the browser control failed: {exc}"
                ) from exc
            logger.warning(
                "cloud_browser: az irányítás lekérdezése nem sikerült (%d/%d): %s",
                consecutive_errors,
                MAX_CONTROL_ERRORS,
                exc,
            )
            await _sleep_unless_cancelled(
                min(CONTROL_ERROR_BACKOFF_SECONDS * consecutive_errors, remaining), cancel
            )
            continue
        consecutive_errors = 0
        if _claimed(response.data):
            if cancel is not None and cancel.is_set():
                await _release_claim(raw_job_id)
                raise CloudBrowserCancelled()
            return
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise CloudBrowserError("Waiting for the browser control timed out.")
        await _sleep_unless_cancelled(min(CONTROL_POLL_SECONDS, remaining), cancel)


async def _update_session(
    database: SupabaseDatabase,
    payload: dict[str, Any],
    filters: list[tuple[str, Any]],
    label: str,
) -> int:
    query: Any = database.table(SESSION_TABLE).update(payload)
    for column, value in filters:
        query = query.eq(column, value)
    selector = getattr(query, "select", None)
    if callable(selector):
        query = selector("id")
    try:
        response = await query.execute()
    except DatabaseError as exc:
        raise CloudBrowserError(f"{label} failed: {exc}") from exc
    affected = len(_response_rows(response))
    if affected == 0:
        logger.debug("cloud_browser: %s nem érintett sort.", label)
    return affected


async def _active_sessions(database: SupabaseDatabase, job_id: str) -> list[dict[str, Any]]:
    try:
        response = await (
            database.table(SESSION_TABLE)
            .select("id, control_mode, agent_busy")
            .eq("job_id", job_id)
            .eq("status", "active")
            .execute()
        )
    except DatabaseError as exc:
        raise CloudBrowserError(f"Reading the browser session failed: {exc}") from exc
    return _response_rows(response)


def _release_transition(
    row: dict[str, Any], job_id: str
) -> tuple[dict[str, Any], list[tuple[str, Any]], str] | None:
    session_id = row.get("id")
    if session_id is None:
        return None
    mode = row.get("control_mode")
    busy = row.get("agent_busy")
    if mode == "requested" and busy is True:
        return (
            {"control_mode": "human", "agent_busy": False},
            [
                ("id", session_id),
                ("job_id", job_id),
                ("status", "active"),
                ("control_mode", "requested"),
                ("agent_busy", True),
            ],
            "Handing the browser control over",
        )
    if mode == "agent" and busy is True:
        return (
            {"agent_busy": False},
            [
                ("id", session_id),
                ("job_id", job_id),
                ("status", "active"),
                ("control_mode", "agent"),
                ("agent_busy", True),
            ],
            "Releasing the browser agent busy flag",
        )
    return None


async def set_browser_agent_busy(job_id: Any, busy: bool) -> None:
    raw_job_id = _normalized_job_id(job_id)
    database = database_module.get_admin_database()
    if busy:
        await _update_session(
            database,
            {"agent_busy": True},
            [
                ("job_id", raw_job_id),
                ("status", "active"),
                ("control_mode", "agent"),
            ],
            "Setting the browser agent busy flag",
        )
        return
    for _attempt in range(MAX_RELEASE_ATTEMPTS):
        rows = await _active_sessions(database, raw_job_id)
        conflicted = False
        for row in rows:
            transition = _release_transition(row, raw_job_id)
            if transition is None:
                continue
            payload, filters, label = transition
            affected = await _update_session(database, payload, filters, label)
            if affected == 0:
                conflicted = True
        if not conflicted:
            return
        logger.debug(
            "cloud_browser: az irányítás felszabadítása ütközött, újrapróbálás (%d/%d).",
            _attempt + 1,
            MAX_RELEASE_ATTEMPTS,
        )
    raise CloudBrowserError(
        "Releasing the browser control failed: the session kept changing concurrently."
    )


async def mark_chat_browser_stopped(browser_id: Any) -> None:
    normalized_id = _normalized_browser_id(browser_id)
    database = database_module.get_admin_database()
    await _update_session(
        database,
        {"status": "stopped", "control_mode": "agent", "agent_busy": False},
        [("browser_id", normalized_id)],
        "Marking the chat browser stopped",
    )