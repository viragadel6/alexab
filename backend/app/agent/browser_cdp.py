from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import socket
import time
from dataclasses import dataclass, replace
from typing import Any, Coroutine
from urllib.parse import urlparse, urlunparse

from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed, WebSocketException
from websockets.protocol import State

from app.config import get_settings

logger = logging.getLogger(__name__)

CDP_CONNECT_TIMEOUT_SECONDS = 15
CDP_COMMAND_TIMEOUT_SECONDS = 15
CDP_CLOSE_TIMEOUT_SECONDS = 5
CDP_OPERATION_TIMEOUT_SECONDS = 60
CDP_IDLE_TIMEOUT_SECONDS = 45
CDP_JANITOR_INTERVAL_SECONDS = 15
CDP_DNS_TIMEOUT_SECONDS = 5
CDP_PING_INTERVAL_SECONDS = 20
CDP_PING_TIMEOUT_SECONDS = 20
CDP_MAX_CONNECTIONS = 16
MAX_ACQUIRE_ATTEMPTS = 3
CDP_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
CDP_ERROR_DETAIL_CHARS = 400
CDP_MAX_URL_LENGTH = 8192
METHOD_NOT_FOUND_CODE = -32601
SCREENSHOT_FORMAT = "jpeg"
SCREENSHOT_QUALITY = 75
ALLOWED_ACTIONS = frozenset({"frame", "input", "navigate", "inspect"})
INPUT_KINDS = frozenset({"text", "key", "mouse", "wheel"})
KEY_EVENT_TYPES = frozenset({"keyDown", "keyUp", "rawKeyDown", "char"})
TEXT_GENERATING_KEY_TYPES = frozenset({"keyDown", "char"})
MOUSE_EVENT_TYPES = frozenset({"mouseMoved", "mousePressed", "mouseReleased"})
MOUSE_BUTTON_NAMES = frozenset({"none", "left", "middle", "right", "back", "forward"})
CLICKABLE_MOUSE_BUTTONS = frozenset({"left", "middle", "right", "back", "forward"})
MOUSE_BUTTON_MASKS = {"left": 1, "right": 2, "middle": 4, "back": 8, "forward": 16}
POINTER_TYPES = frozenset({"mouse", "pen", "touch"})
KEY_TEXT = {"Enter": "\r", "Tab": "\t"}
MODIFIER_ALT = 1
MODIFIER_CTRL = 2
MODIFIER_META = 4
MODIFIER_SHIFT = 8
TEXT_SUPPRESSING_MODIFIERS = MODIFIER_ALT | MODIFIER_CTRL | MODIFIER_META
MAX_MODIFIERS = MODIFIER_ALT | MODIFIER_CTRL | MODIFIER_META | MODIFIER_SHIFT
MAX_CLICK_COUNT = 3
MAX_BUTTON_MASK = 31
MAX_KEY_CODE = 65535
MAX_KEY_LOCATION = 3
MAX_KEY_LENGTH = 64
MAX_TEXT_LENGTH = 100_000
MAX_PAGE_TEXT_LENGTH = 6_000
MAX_PAGE_LINKS = 20
MAX_COORDINATE = 100_000.0
MAX_WHEEL_DELTA = 100_000.0
DELTA_MODE_PIXEL = 0
DELTA_MODE_LINE = 1
DELTA_MODE_PAGE = 2
LINE_DELTA_PIXELS = 16.0
PAGE_DELTA_PIXELS = 800.0
MIN_SCALE_FACTOR = 0.1
MAX_SCALE_FACTOR = 10.0
WEBSOCKET_SCHEMES = frozenset({"ws", "wss"})
HTTP_SCHEME_MAP = {"http": "ws", "https": "wss"}
EXCLUDED_TARGET_PREFIXES = (
    "chrome://",
    "chrome-extension://",
    "chrome-search://",
    "chrome-untrusted://",
    "devtools://",
    "edge://",
    "edge-extension://",
    "moz-extension://",
    "about:srcdoc",
    "view-source:chrome",
)
STALE_SESSION_MARKERS = (
    "session with given id not found",
    "no session with given id",
    "cannot find session",
    "session closed",
    "target closed",
    "target not found",
    "not attached to the target",
    "frame with the given id not found",
)
BLANK_PAGE_URLS = frozenset({"", "about:blank"})

_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()
_registry_map: dict[str, "_CdpConnection"] = {}
_registry_lock: asyncio.Lock | None = None
_registry_loop: asyncio.AbstractEventLoop | None = None
_janitor: asyncio.Task[None] | None = None


class BrowserCdpError(RuntimeError):
    def __init__(
        self,
        message: str,
        code: int | None = None,
        data: Any = None,
        stale_session: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.data = data
        self.stale_session = stale_session

    def clone(self) -> "BrowserCdpError":
        clone = BrowserCdpError(
            self.message, code=self.code, data=self.data, stale_session=self.stale_session
        )
        clone.__cause__ = self.__cause__
        clone.__context__ = None
        return clone


@dataclass(slots=True)
class _PendingCommand:
    request_id: int
    future: asyncio.Future[dict[str, Any]]
    method: str
    session_id: str | None


@dataclass(slots=True)
class _PageMetrics:
    url: str = ""
    width: float = 0.0
    height: float = 0.0
    scale: float = 1.0


def _retain(awaitable: Coroutine[Any, Any, Any] | asyncio.Task[Any]) -> asyncio.Task[Any]:
    task: asyncio.Task[Any] = (
        awaitable if isinstance(awaitable, asyncio.Task) else asyncio.create_task(awaitable)
    )
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_release_task)
    return task


def _release_task(task: asyncio.Task[Any]) -> None:
    _BACKGROUND_TASKS.discard(task)
    if task.cancelled():
        return
    failure = task.exception()
    if failure is not None:
        logger.warning("cdp: háttérfeladat hibával ért véget: %r", failure)


async def _reap(task: asyncio.Task[Any], label: str) -> BaseException | None:
    await asyncio.wait({task})
    if task.cancelled():
        return None
    failure = task.exception()
    if failure is not None:
        logger.warning("cdp: %s hibával ért véget: %r", label, failure)
    return failure


def _consume_future(future: asyncio.Future[Any]) -> None:
    if future.cancelled():
        return
    future.exception()


def _record(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _text(value: Any, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or len(stripped) > limit:
        return None
    return stripped


def _truncated_text(value: Any, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped[:limit]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _positive_number(value: Any) -> float:
    number = _number(value)
    if number is None or number <= 0:
        return 0.0
    return number


def _clamp(number: float, minimum: float, maximum: float) -> float:
    return min(max(number, minimum), maximum)


def _required_choice(value: Any, allowed: frozenset[str], message: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise BrowserCdpError(message)
    return value


def _optional_choice(value: Any, allowed: frozenset[str], message: str) -> str | None:
    if value is None:
        return None
    return _required_choice(value, allowed, message)


def _required_int(value: Any, minimum: int, maximum: int, message: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise BrowserCdpError(message)
    return value


def _bounded_int(
    value: Any, minimum: int, maximum: int, message: str, default: int
) -> int:
    if value is None:
        return default
    return _required_int(value, minimum, maximum, message)


def _optional_int(value: Any, minimum: int, maximum: int, message: str) -> int | None:
    if value is None:
        return None
    return _required_int(value, minimum, maximum, message)


def _required_number(value: Any, minimum: float, maximum: float, message: str) -> float:
    number = _number(value)
    if number is None or not minimum <= number <= maximum:
        raise BrowserCdpError(message)
    return number


def _required_text(value: Any, message: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_TEXT_LENGTH:
        raise BrowserCdpError(message)
    return value


def _optional_text(value: Any, message: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, message)


def _derived_key_text(key: str | None, modifiers: int) -> str | None:
    if key is None:
        return None
    if modifiers & TEXT_SUPPRESSING_MODIFIERS:
        return None
    mapped = KEY_TEXT.get(key)
    if mapped is not None:
        return mapped
    if len(key) == 1:
        return key
    return None


def _key_event_params(input_data: dict[str, Any]) -> dict[str, Any]:
    event_type = _required_choice(
        input_data.get("type"), KEY_EVENT_TYPES, "Érvénytelen billentyűesemény típus."
    )
    modifiers = _bounded_int(
        input_data.get("modifiers"),
        0,
        MAX_MODIFIERS,
        "Érvénytelen billentyűmódosító.",
        0,
    )
    key = _optional_text(input_data.get("key"), "Érvénytelen billentyű.")
    if key is not None and len(key) > MAX_KEY_LENGTH:
        raise BrowserCdpError("Érvénytelen billentyű.")
    code = _optional_text(input_data.get("code"), "Érvénytelen billentyűkód.")
    if code is not None and len(code) > MAX_KEY_LENGTH:
        raise BrowserCdpError("Érvénytelen billentyűkód.")
    params: dict[str, Any] = {"type": event_type, "modifiers": modifiers}
    if key is not None:
        params["key"] = key
    if code is not None:
        params["code"] = code
    windows_code = _optional_int(
        input_data.get("keyCode"), 0, MAX_KEY_CODE, "Érvénytelen billentyűkód."
    )
    if windows_code is not None:
        params["windowsVirtualKeyCode"] = windows_code
    native_code = _optional_int(
        input_data.get("nativeKeyCode"), 0, MAX_KEY_CODE, "Érvénytelen natív billentyűkód."
    )
    if native_code is not None:
        params["nativeVirtualKeyCode"] = native_code
    location = _optional_int(
        input_data.get("location"), 0, MAX_KEY_LOCATION, "Érvénytelen billentyűhely."
    )
    if location is not None:
        params["location"] = location
        params["isKeypad"] = location == MAX_KEY_LOCATION
    keypad = input_data.get("isKeypad")
    if isinstance(keypad, bool):
        params["isKeypad"] = keypad
    auto_repeat = input_data.get("autoRepeat")
    if isinstance(auto_repeat, bool):
        params["autoRepeat"] = auto_repeat
    if event_type not in TEXT_GENERATING_KEY_TYPES:
        return params
    text = input_data.get("text")
    if text is None:
        text = _derived_key_text(key, modifiers)
    if text is None:
        if event_type == "char":
            raise BrowserCdpError("A karakteresemény szövege hiányzik.")
        return params
    character = _required_text(text, "Érvénytelen billentyűszöveg.")
    params["text"] = character
    params["unmodifiedText"] = character
    return params


def _scaled_coordinate(value: Any, label: str, scale: float, extent: float) -> float:
    coordinate = _required_number(
        value, -MAX_COORDINATE, MAX_COORDINATE, f"Érvénytelen koordináta: {label}."
    )
    scaled = coordinate / scale
    if extent > 0:
        return _clamp(scaled, 0.0, extent)
    return scaled


def _wheel_delta(value: Any, delta_mode: Any, label: str, extent: float) -> float:
    mode = DELTA_MODE_PIXEL
    if delta_mode is not None:
        mode = _required_int(delta_mode, DELTA_MODE_PIXEL, DELTA_MODE_PAGE, "Érvénytelen görgetési mód.")
    delta = 0.0
    if value is not None:
        delta = _required_number(
            value, -MAX_WHEEL_DELTA, MAX_WHEEL_DELTA, f"Érvénytelen görgetési érték: {label}."
        )
    if mode == DELTA_MODE_LINE:
        delta *= LINE_DELTA_PIXELS
    elif mode == DELTA_MODE_PAGE:
        delta *= extent if extent > 0 else PAGE_DELTA_PIXELS
    return _clamp(delta, -MAX_WHEEL_DELTA, MAX_WHEEL_DELTA)


def _mouse_event_params(
    kind: str, input_data: dict[str, Any], metrics: _PageMetrics
) -> dict[str, Any]:
    if kind == "wheel":
        supplied_type = input_data.get("type")
        if supplied_type is not None and supplied_type != "mouseWheel":
            raise BrowserCdpError("Érvénytelen görgetésesemény típus.")
        event_type = "mouseWheel"
    else:
        event_type = _required_choice(
            input_data.get("type"), MOUSE_EVENT_TYPES, "Érvénytelen egéresemény típus."
        )
    scale = metrics.scale if MIN_SCALE_FACTOR <= metrics.scale <= MAX_SCALE_FACTOR else 1.0
    modifiers = _bounded_int(
        input_data.get("modifiers"), 0, MAX_MODIFIERS, "Érvénytelen egérmódosító.", 0
    )
    params: dict[str, Any] = {
        "type": event_type,
        "modifiers": modifiers,
        "x": _scaled_coordinate(input_data.get("x"), "x", scale, metrics.width),
        "y": _scaled_coordinate(input_data.get("y"), "y", scale, metrics.height),
        "pointerType": _optional_choice(
            input_data.get("pointerType"), POINTER_TYPES, "Érvénytelen mutatótípus."
        )
        or "mouse",
    }
    if event_type in ("mousePressed", "mouseReleased"):
        button = _required_choice(
            input_data.get("button"), CLICKABLE_MOUSE_BUTTONS, "Érvénytelen egérgomb."
        )
        click_count = _bounded_int(
            input_data.get("clickCount"), 1, MAX_CLICK_COUNT, "Érvénytelen kattintásszám.", 1
        )
        held_buttons = MOUSE_BUTTON_MASKS[button] if event_type == "mousePressed" else 0
    else:
        button = (
            _optional_choice(input_data.get("button"), MOUSE_BUTTON_NAMES, "Érvénytelen egérgomb.")
            or "none"
        )
        click_count = _bounded_int(
            input_data.get("clickCount"), 0, MAX_CLICK_COUNT, "Érvénytelen kattintásszám.", 0
        )
        held_buttons = 0
    params["button"] = button
    params["clickCount"] = click_count
    params["buttons"] = _bounded_int(
        input_data.get("buttons"), 0, MAX_BUTTON_MASK, "Érvénytelen egérgomb állapot.", held_buttons
    )
    if event_type == "mouseWheel":
        delta_mode = input_data.get("deltaMode")
        params["deltaX"] = _wheel_delta(input_data.get("deltaX"), delta_mode, "deltaX", metrics.width)
        params["deltaY"] = _wheel_delta(input_data.get("deltaY"), delta_mode, "deltaY", metrics.height)
    return params


def _is_excluded_url(url: str) -> bool:
    normalized = url.strip().lower()
    if normalized in BLANK_PAGE_URLS:
        return False
    return normalized.startswith(EXCLUDED_TARGET_PREFIXES)


def _is_page_target(target: Any) -> bool:
    record = _record(target)
    if record is None:
        return False
    if record.get("type") != "page":
        return False
    if _text(record.get("targetId"), 256) is None:
        return False
    url = record.get("url")
    if url is not None and not isinstance(url, str):
        return False
    return not _is_excluded_url(url or "")


def _target_rank(indexed: tuple[int, dict[str, Any]]) -> tuple[int, int]:
    index, target = indexed
    url = str(target.get("url") or "").strip()
    blank = 1 if url in BLANK_PAGE_URLS else 0
    return (blank, -index)


def _select_target(
    candidates: list[dict[str, Any]], preferred: str | None
) -> dict[str, Any]:
    if preferred:
        for target in candidates:
            if target.get("targetId") == preferred:
                return target
    return min(list(enumerate(candidates)), key=_target_rank)[1]


def _stale_session_error(error: BrowserCdpError) -> BrowserCdpError:
    lowered = error.message.lower()
    if any(marker in lowered for marker in STALE_SESSION_MARKERS):
        return BrowserCdpError(
            error.message, code=error.code, data=error.data, stale_session=True
        )
    return error


class _CdpConnection:
    def __init__(self, url: str, loop: asyncio.AbstractEventLoop) -> None:
        self.url = url
        self.loop = loop
        self.command_lock = asyncio.Lock()
        self.socket: ClientConnection | None = None
        self.receiver: asyncio.Task[None] | None = None
        self.pending: dict[int, _PendingCommand] = {}
        self.failure: BrowserCdpError | None = None
        self.generation = 0
        self.target_id: str | None = None
        self.session_id: str | None = None
        self.page_level = False
        self.metrics = _PageMetrics()
        self.next_id = 0
        self.last_used = time.monotonic()

    def touch(self) -> None:
        self.last_used = time.monotonic()

    def is_alive(self) -> bool:
        return (
            self.failure is None
            and self.socket is not None
            and self.socket.state is State.OPEN
            and self.receiver is not None
            and not self.receiver.done()
        )

    def is_idle(self) -> bool:
        return not self.command_lock.locked() and not self.pending

    def _abort(self, error: BrowserCdpError, cause: BaseException | None = None) -> None:
        if cause is not None:
            error.__cause__ = cause
            logger.warning("cdp: %s (%s: %s)", error.message, type(cause).__name__, cause)
        if self.failure is None:
            self.failure = error
        for command in list(self.pending.values()):
            self.pending.pop(command.request_id, None)
            if not command.future.done():
                command.future.set_exception(error.clone())

    def _fail_session(self, session_id: str | None, message: str) -> None:
        for command in list(self.pending.values()):
            if session_id is None or command.session_id == session_id:
                self.pending.pop(command.request_id, None)
                if not command.future.done():
                    command.future.set_exception(
                        BrowserCdpError(message, stale_session=True)
                    )

    async def ensure_connected(self) -> None:
        if self.is_alive():
            return
        await self.shutdown(self.failure)
        self.failure = None
        generation = self.generation
        try:
            connection = await ws_connect(
                self.url,
                open_timeout=CDP_CONNECT_TIMEOUT_SECONDS,
                close_timeout=CDP_CLOSE_TIMEOUT_SECONDS,
                max_size=CDP_MAX_MESSAGE_BYTES,
                ping_interval=CDP_PING_INTERVAL_SECONDS,
                ping_timeout=CDP_PING_TIMEOUT_SECONDS,
                compression=None,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise BrowserCdpError("Nem sikerült csatlakozni a böngészőhöz.") from exc
        if generation != self.generation:
            await _close_socket(connection)
            raise BrowserCdpError("A böngésző kapcsolat a csatlakozás közben lezárult.")
        self.socket = connection
        self.page_level = False
        self.receiver = asyncio.create_task(self._receive_loop(), name="cdp-receiver")
        try:
            await self.send("Target.setDiscoverTargets", {"discover": True})
        except BrowserCdpError as exc:
            if exc.code == METHOD_NOT_FOUND_CODE:
                logger.warning("cdp: a Target tartomány nem elérhető, lap szintű kapcsolat.")
                self.page_level = True
                return
            await self.shutdown(BrowserCdpError("A böngésző vezérlése nem elérhető."))
            raise

    def detach(self, reason: BrowserCdpError | None = None) -> None:
        _retain(self.shutdown(reason))

    async def shutdown(self, reason: BrowserCdpError | None = None) -> None:
        self.generation += 1
        error = reason or self.failure or BrowserCdpError("A böngésző kapcsolat lezárult.")
        self._abort(error)
        receiver = self.receiver
        connection = self.socket
        self.receiver = None
        self.socket = None
        self.session_id = None
        self.target_id = None
        if receiver is not None:
            receiver.cancel()
            await _reap(receiver, "cdp-figyelő")
        if connection is not None:
            await _close_socket(connection)

    async def send(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        connection = self.socket
        receiver = self.receiver
        if connection is None or receiver is None or receiver.done() or self.failure is not None:
            raise (
                self.failure.clone()
                if self.failure is not None
                else BrowserCdpError("A böngésző kapcsolata megszakadt.")
            )
        self.next_id += 1
        request_id = self.next_id
        payload: dict[str, Any] = {
            "id": request_id,
            "method": method,
            "params": params if params is not None else {},
        }
        if session_id is not None:
            payload["sessionId"] = session_id
        try:
            raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise BrowserCdpError(f"A CDP kérés nem szerializálható: {method}.") from exc
        future: asyncio.Future[dict[str, Any]] = self.loop.create_future()
        future.add_done_callback(_consume_future)
        command = _PendingCommand(
            request_id=request_id, future=future, method=method, session_id=session_id
        )
        self.pending[request_id] = command
        try:
            return await asyncio.wait_for(
                self._exchange(connection, request_id, raw, future),
                timeout=CDP_COMMAND_TIMEOUT_SECONDS,
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise BrowserCdpError(f"A CDP parancs nem válaszolt időben: {method}.") from exc

    async def _exchange(
        self,
        connection: ClientConnection,
        request_id: int,
        raw: str,
        future: asyncio.Future[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            await connection.send(raw)
            return await future
        except ConnectionClosed as exc:
            raise BrowserCdpError("A böngésző kapcsolata megszakadt.") from exc
        except WebSocketException as exc:
            raise BrowserCdpError("A böngésző üzenet továbbítása nem sikerült.") from exc
        except OSError as exc:
            raise BrowserCdpError("A böngésző kapcsolat hálózati hibája.") from exc
        finally:
            self.pending.pop(request_id, None)
            if not future.done():
                future.cancel()

    async def _receive_loop(self) -> None:
        connection = self.socket
        if connection is None:
            return
        try:
            async for raw in connection:
                if not self._consume(raw):
                    return
        except asyncio.CancelledError:
            self._abort(BrowserCdpError("A böngésző kapcsolat bontva."))
            raise
        except ConnectionClosed as exc:
            self._abort(BrowserCdpError("A böngésző kapcsolata megszakadt."), exc)
        except WebSocketException as exc:
            self._abort(BrowserCdpError("A böngésző vezérlő kapcsolata megszakadt."), exc)
        except OSError as exc:
            self._abort(BrowserCdpError("A böngésző kapcsolat hálózati hibája."), exc)
        except Exception as exc:
            self._abort(BrowserCdpError("A böngésző vezérlése váratlanul megszakadt."), exc)
        else:
            self._abort(BrowserCdpError("A böngésző kapcsolata lezárult."))

    def _consume(self, raw: str | bytes) -> bool:
        if isinstance(raw, (bytes, bytearray, memoryview)):
            try:
                text = bytes(raw).decode("utf-8")
            except UnicodeDecodeError as exc:
                self._abort(BrowserCdpError("Érvénytelen CDP üzenet."), exc)
                return False
        else:
            text = raw
        try:
            message = json.loads(text)
        except (ValueError, TypeError) as exc:
            self._abort(BrowserCdpError("Érvénytelen CDP üzenet."), exc)
            return False
        record = _record(message)
        if record is None:
            self._abort(BrowserCdpError("Érvénytelen CDP üzenet."))
            return False
        raw_id = record.get("id")
        if raw_id is not None:
            self._resolve(raw_id, record)
            return True
        self._handle_event(record)
        return True

    def _resolve(self, raw_id: Any, message: dict[str, Any]) -> None:
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            logger.warning("cdp: érvénytelen válaszazonosító: %r", raw_id)
            return
        command = self.pending.pop(raw_id, None)
        if command is None or command.future.done():
            return
        error = message.get("error")
        if error is not None:
            command.future.set_exception(self._command_error(command.method, error))
            return
        if "result" not in message:
            command.future.set_exception(
                BrowserCdpError(f"A CDP válasz eredménye hiányzik: {command.method}.")
            )
            return
        result = _record(message.get("result"))
        if result is None:
            command.future.set_exception(
                BrowserCdpError(f"Érvénytelen CDP válasz: {command.method}.")
            )
            return
        command.future.set_result(result)

    def _command_error(self, method: str, error: Any) -> BrowserCdpError:
        record = _record(error)
        if record is None:
            return BrowserCdpError(
                f"CDP hiba ({method}): {str(error)[:CDP_ERROR_DETAIL_CHARS]}"
            )
        message = (
            _truncated_text(record.get("message"), CDP_ERROR_DETAIL_CHARS) or "ismeretlen hiba"
        )
        raw_code = record.get("code")
        code = raw_code if isinstance(raw_code, int) and not isinstance(raw_code, bool) else None
        data = record.get("data")
        rendered = f"CDP hiba ({method})"
        if code is not None:
            rendered += f" [{code}]"
        rendered += f": {message}"
        if data is not None:
            try:
                detail = (
                    json.dumps(data, ensure_ascii=False)
                    if isinstance(data, (dict, list))
                    else str(data)
                )
            except (TypeError, ValueError):
                detail = str(data)
            rendered += f" | {detail[:CDP_ERROR_DETAIL_CHARS]}"
        failure = BrowserCdpError(rendered[: 4 * CDP_ERROR_DETAIL_CHARS], code=code, data=data)
        return _stale_session_error(failure)

    def _handle_event(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        if not isinstance(method, str):
            logger.warning("cdp: esemény metódus nélkül: %r", message.get("id"))
            return
        params = _record(message.get("params")) or {}
        session = _text(message.get("sessionId"), 256)
        if method == "Target.detachedFromTarget":
            detached = _text(params.get("sessionId"), 256) or session
            self._session_lost(detached, "A böngésző munkamenet leválasztásra került.")
        elif method in ("Inspector.targetCrashed", "Target.targetCrashed"):
            crashed = _text(params.get("targetId"), 256)
            if crashed is None or crashed == self.target_id:
                self._session_lost(
                    session, "A böngészőlap összeomlott.", clear_target=True
                )
        elif method == "Target.targetDestroyed":
            destroyed = _text(params.get("targetId"), 256)
            if destroyed is not None and destroyed == self.target_id:
                self._session_lost(session, "A böngészőlap bezáródott.", clear_target=True)
        elif method == "Target.targetInfoChanged":
            info = _record(params.get("targetInfo"))
            if info is not None and _text(info.get("targetId"), 256) == self.target_id:
                url = info.get("url")
                if isinstance(url, str):
                    self.metrics = replace(self.metrics, url=url)

    def _session_lost(
        self, session_id: str | None, message: str, clear_target: bool = False
    ) -> None:
        matched = session_id is None or session_id == self.session_id
        if not matched:
            return
        if clear_target:
            self.target_id = None
        self.session_id = None
        self._fail_session(session_id, message)

    async def execute(
        self, action: str, input_data: dict[str, Any] | None
    ) -> dict[str, Any]:
        if action not in ALLOWED_ACTIONS:
            raise BrowserCdpError("Érvénytelen böngésző művelet.")
        attempt = 0
        while True:
            attempt += 1
            await self.ensure_connected()
            try:
                session_id, target_id = await self._acquire_session()
                if action == "frame":
                    return await self._capture_frame(session_id, target_id)
                if action == "navigate":
                    return await self._navigate(session_id, target_id, input_data)
                if action == "inspect":
                    return await self._inspect(session_id)
                return await self._dispatch_input(session_id, input_data)
            except BrowserCdpError as exc:
                if attempt < 2 and exc.stale_session:
                    logger.warning("cdp: elavult munkamenet, újracsatolás: %s", exc.message)
                    self.session_id = None
                    self.target_id = None
                    continue
                raise

    async def _acquire_session(self) -> tuple[str | None, str | None]:
        if self.page_level:
            return None, None
        if self.session_id is not None and self.target_id is not None:
            return self.session_id, self.target_id
        try:
            targets = await self.send("Target.getTargets")
        except BrowserCdpError as exc:
            if exc.code == METHOD_NOT_FOUND_CODE:
                logger.warning("cdp: a Target.getTargets nem elérhető, lap szintű kapcsolat.")
                self.page_level = True
                self.target_id = None
                self.session_id = None
                return None, None
            raise
        infos = targets.get("targetInfos")
        if infos is None:
            raise BrowserCdpError("A böngésző nem adott vissza laplistát.")
        if not isinstance(infos, list):
            raise BrowserCdpError("Érvénytelen böngésző laplista.")
        candidates = [target for target in infos if _is_page_target(target)]
        if not candidates:
            raise BrowserCdpError("Nincs megnyitott böngészőlap.")
        chosen = _select_target(candidates, self.target_id)
        target_id = _text(chosen.get("targetId"), 256)
        if target_id is None:
            raise BrowserCdpError("Érvénytelen böngészőlap azonosító.")
        url = chosen.get("url")
        self.metrics = replace(self.metrics, url=url if isinstance(url, str) else "")
        attached = await self.send(
            "Target.attachToTarget", {"targetId": target_id, "flatten": True}
        )
        session_id = _text(attached.get("sessionId"), 256)
        if session_id is None:
            raise BrowserCdpError("A böngészőlap csatolása nem sikerült.")
        self.target_id = target_id
        self.session_id = session_id
        return session_id, target_id

    async def _activate(self, target_id: str | None) -> None:
        if target_id is None:
            return
        try:
            await self.send("Target.activateTarget", {"targetId": target_id})
        except BrowserCdpError as exc:
            if exc.stale_session:
                raise
            logger.warning("cdp: a böngészőlap előtérbe hozása nem sikerült: %s", exc.message)

    async def _capture_frame(
        self, session_id: str | None, target_id: str | None
    ) -> dict[str, Any]:
        await self._activate(target_id)
        screenshot = await self.send(
            "Page.captureScreenshot",
            {
                "format": SCREENSHOT_FORMAT,
                "quality": SCREENSHOT_QUALITY,
                "captureBeyondViewport": False,
                "fromSurface": True,
            },
            session_id,
        )
        data = screenshot.get("data")
        if not isinstance(data, str) or not data:
            raise BrowserCdpError("A képernyőkép nem érkezett meg.")
        metrics = await self._read_metrics(session_id)
        url = await self._read_url(session_id, target_id)
        return {
            "data": data,
            "url": url,
            "width": metrics.width,
            "height": metrics.height,
            "deviceScaleFactor": metrics.scale,
        }

    async def _read_metrics(self, session_id: str | None) -> _PageMetrics:
        try:
            layout = await self.send("Page.getLayoutMetrics", {}, session_id)
        except BrowserCdpError as exc:
            if exc.stale_session:
                raise
            logger.warning("cdp: a lapmetrikák nem érhetők el: %s", exc.message)
            return self.metrics
        css = _record(layout.get("cssVisualViewport")) or _record(layout.get("cssLayoutViewport"))
        device = _record(layout.get("visualViewport")) or _record(layout.get("layoutViewport"))
        width = _positive_number(css.get("clientWidth")) if css is not None else 0.0
        height = _positive_number(css.get("clientHeight")) if css is not None else 0.0
        scale = 1.0
        if css is not None and device is not None and width > 0:
            device_width = _positive_number(device.get("clientWidth"))
            if device_width > 0:
                ratio = device_width / width
                if MIN_SCALE_FACTOR <= ratio <= MAX_SCALE_FACTOR:
                    scale = ratio
        metrics = _PageMetrics(
            url=self.metrics.url,
            width=width or self.metrics.width,
            height=height or self.metrics.height,
            scale=scale,
        )
        self.metrics = metrics
        return metrics

    async def _read_url(self, session_id: str | None, target_id: str | None) -> str:
        if target_id is not None:
            try:
                info = await self.send("Target.getTargetInfo", {"targetId": target_id})
            except BrowserCdpError as exc:
                if exc.stale_session:
                    raise
                logger.warning("cdp: a lap címe nem kérdezhető le: %s", exc.message)
                return self.metrics.url
            target = _record(info.get("targetInfo"))
            url = _text(target.get("url"), CDP_MAX_URL_LENGTH) if target is not None else None
            if url is not None:
                self.metrics = replace(self.metrics, url=url)
                return url
            return self.metrics.url
        try:
            history = await self.send("Page.getNavigationHistory", {}, session_id)
        except BrowserCdpError as exc:
            if exc.stale_session:
                raise
            logger.warning("cdp: a lap címe nem kérdezhető le: %s", exc.message)
            return self.metrics.url
        entries = history.get("entries")
        index = history.get("currentIndex")
        if isinstance(entries, list) and isinstance(index, int) and not isinstance(index, bool):
            if 0 <= index < len(entries):
                entry = _record(entries[index])
                url = _text(entry.get("url"), CDP_MAX_URL_LENGTH) if entry is not None else None
                if url is not None:
                    self.metrics = replace(self.metrics, url=url)
                    return url
        return self.metrics.url

    async def _dispatch_input(
        self, session_id: str | None, input_data: dict[str, Any] | None
    ) -> dict[str, Any]:
        if not isinstance(input_data, dict) or not input_data:
            raise BrowserCdpError("Hiányzó beviteli művelet.")
        kind = input_data.get("kind")
        if not isinstance(kind, str) or kind not in INPUT_KINDS:
            raise BrowserCdpError("Érvénytelen beviteli művelet.")
        if kind in ("mouse", "wheel") and self.metrics.width <= 0:
            await self._read_metrics(session_id)
        if kind == "text":
            await self.send(
                "Input.insertText",
                {"text": _required_text(input_data.get("text"), "Érvénytelen szöveges bevitel.")},
                session_id,
            )
            return {"ok": True}
        if kind == "key":
            await self.send(
                "Input.dispatchKeyEvent", _key_event_params(input_data), session_id
            )
            return {"ok": True}
        await self.send(
            "Input.dispatchMouseEvent",
            _mouse_event_params(kind, input_data, self.metrics),
            session_id,
        )
        return {"ok": True}

    async def _navigate(
        self,
        session_id: str | None,
        target_id: str | None,
        input_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(input_data, dict):
            raise BrowserCdpError("Hiányzó navigációs adat.")
        value = input_data.get("url")
        if not isinstance(value, str) or not value.strip() or len(value) > CDP_MAX_URL_LENGTH:
            raise BrowserCdpError("Érvénytelen navigációs cím.")
        url = value.strip()
        try:
            parsed = urlparse(url)
            port = parsed.port
        except ValueError as exc:
            raise BrowserCdpError("Érvénytelen navigációs cím.") from exc
        if (
            parsed.scheme.lower() not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or (port is not None and not 0 < port <= 65535)
            or any(character.isspace() or ord(character) < 32 for character in url)
        ):
            raise BrowserCdpError("Érvénytelen navigációs cím.")
        navigation = await self.send("Page.navigate", {"url": url}, session_id)
        error_text = navigation.get("errorText")
        if isinstance(error_text, str) and error_text:
            raise BrowserCdpError(f"A böngésző navigációja sikertelen: {error_text[:200]}")
        self.metrics = replace(self.metrics, url=url)
        ready_state = ""
        deadline = asyncio.get_running_loop().time() + 15.0
        while asyncio.get_running_loop().time() < deadline:
            try:
                evaluated = await self.send(
                    "Runtime.evaluate",
                    {"expression": "document.readyState", "returnByValue": True},
                    session_id,
                )
            except BrowserCdpError as exc:
                if exc.stale_session:
                    raise
                break
            result = _record(evaluated.get("result"))
            state = result.get("value") if result is not None else None
            ready_state = state if isinstance(state, str) else ""
            if ready_state in ("interactive", "complete"):
                break
            await asyncio.sleep(0.2)
        return {
            "url": await self._read_url(session_id, target_id),
            "ready_state": ready_state or "loading",
            "loader_id": navigation.get("loaderId"),
        }

    async def _inspect(self, session_id: str | None) -> dict[str, Any]:
        expression = (
            "(() => { const body = document.body; "
            "const links = Array.from(document.querySelectorAll('a[href]')).slice(0, 20)"
            ".map((a) => ({text: (a.innerText || a.getAttribute('aria-label') || '').trim()"
            ".slice(0, 160), url: a.href})); "
            "return {title: document.title || '', url: location.href, "
            "text: (body ? body.innerText : '').slice(0, 6000), links}; })()"
        )
        evaluated = await self.send(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": True,
            },
            session_id,
        )
        if evaluated.get("exceptionDetails") is not None:
            raise BrowserCdpError("A böngészőoldal tartalmának kiolvasása sikertelen.")
        result = _record(evaluated.get("result"))
        value = _record(result.get("value")) if result is not None else None
        if value is None:
            raise BrowserCdpError("A böngésző nem adott vissza olvasható oldaltartalmat.")
        title = value.get("title")
        url = value.get("url")
        text = value.get("text")
        raw_links = value.get("links")
        links: list[dict[str, str]] = []
        if isinstance(raw_links, list):
            for item in raw_links[:MAX_PAGE_LINKS]:
                link = _record(item)
                if link is None:
                    continue
                link_text = link.get("text")
                link_url = link.get("url")
                if not isinstance(link_url, str) or not link_url:
                    continue
                if not isinstance(link_text, str):
                    link_text = ""
                links.append({"text": link_text[:160], "url": link_url[:CDP_MAX_URL_LENGTH]})
        return {
            "title": title[:500] if isinstance(title, str) else "",
            "url": url[:CDP_MAX_URL_LENGTH] if isinstance(url, str) else "",
            "text": text[:MAX_PAGE_TEXT_LENGTH] if isinstance(text, str) else "",
            "links": links,
        }


def _bind_registry() -> None:
    global _registry_lock, _registry_loop, _registry_map, _janitor
    loop = asyncio.get_running_loop()
    if _registry_lock is not None and _registry_loop is loop:
        return
    if _registry_map:
        logger.warning(
            "cdp: %d elavult böngészőkapcsolat eldobva (új eseményciklus).", len(_registry_map)
        )
        _registry_map = {}
    _janitor = None
    _registry_lock = asyncio.Lock()
    _registry_loop = loop


def _registry_table() -> dict[str, _CdpConnection]:
    _bind_registry()
    return _registry_map


def _registry_guard() -> asyncio.Lock:
    global _registry_lock
    _bind_registry()
    lock = _registry_lock
    if lock is None:
        lock = asyncio.Lock()
        _registry_lock = lock
    return lock


def _is_expired(connection: _CdpConnection) -> bool:
    if not connection.is_idle():
        return False
    return time.monotonic() - connection.last_used > CDP_IDLE_TIMEOUT_SECONDS


async def _close_socket(connection: ClientConnection) -> None:
    try:
        await asyncio.wait_for(connection.close(), timeout=CDP_CLOSE_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("cdp: a kapcsolat lezárása nem sikerült: %r", exc)


def _discard_connection(connection: _CdpConnection, reason: BrowserCdpError | None) -> None:
    registry = _registry_table()
    if registry.get(connection.url) is connection:
        registry.pop(connection.url, None)
    connection.detach(reason)


def _ensure_janitor() -> None:
    global _janitor
    janitor = _janitor
    if janitor is not None and not janitor.done():
        return
    _janitor = _retain(_janitor_loop())


async def _janitor_loop() -> None:
    global _janitor
    while True:
        await asyncio.sleep(CDP_JANITOR_INTERVAL_SECONDS)
        registry = _registry_table()
        if not registry:
            _janitor = None
            return
        for connection in [item for item in registry.values() if _is_expired(item)]:
            _discard_connection(
                connection,
                BrowserCdpError("A böngésző kapcsolat tétlenség miatt lezárult."),
            )


def _blocked_address(value: str) -> bool:
    candidate = value.split("%", 1)[0]
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return True
    if isinstance(address, ipaddress.IPv6Address):
        mapped = address.ipv4_mapped
        if mapped is not None:
            address = mapped
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def _settings_value(name: str, default: Any) -> Any:
    try:
        settings = get_settings()
    except Exception as exc:
        logger.warning("cdp: a beállítások nem olvashatók: %r", exc)
        return default
    value = getattr(settings, name, default)
    return default if value is None else value


def _allowed_browser_hosts() -> frozenset[str]:
    raw = _settings_value("chat_browser_allowed_hosts", "")
    if isinstance(raw, str):
        items = raw.split(",")
    elif isinstance(raw, (list, tuple, set, frozenset)):
        items = [str(item) for item in raw]
    else:
        items = []
    return frozenset(item.strip().lower() for item in items if item and item.strip())


def _private_hosts_allowed() -> bool:
    raw = _settings_value("chat_browser_allow_private_hosts", False)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(raw, int):
        return raw != 0
    return False


def normalize_cdp_url(url: Any) -> str:
    if not isinstance(url, str):
        raise BrowserCdpError("Hiányzó böngésző cím.")
    candidate = url.strip()
    if not candidate:
        raise BrowserCdpError("Hiányzó böngésző cím.")
    if len(candidate) > CDP_MAX_URL_LENGTH:
        raise BrowserCdpError("A böngésző cím túl hosszú.")
    if any(character.isspace() or ord(character) < 32 for character in candidate):
        raise BrowserCdpError("A böngésző cím érvénytelen karaktert tartalmaz.")
    try:
        parsed = urlparse(candidate)
    except ValueError as exc:
        raise BrowserCdpError("A böngésző cím nem értelmezhető.") from exc
    scheme = (parsed.scheme or "").lower()
    if scheme in HTTP_SCHEME_MAP:
        scheme = HTTP_SCHEME_MAP[scheme]
    if scheme not in WEBSOCKET_SCHEMES:
        raise BrowserCdpError("A böngésző cím csak ws vagy wss cím lehet.")
    if not parsed.netloc:
        raise BrowserCdpError("A böngésző cím gazdagépe hiányzik.")
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise BrowserCdpError("A böngésző cím portja érvénytelen.") from exc
    if not hostname or not hostname.strip():
        raise BrowserCdpError("A böngésző cím gazdagépe hiányzik.")
    if port is not None and not 0 < port <= 65535:
        raise BrowserCdpError("A böngésző cím portja érvénytelen.")
    if parsed.fragment:
        raise BrowserCdpError("A böngésző cím nem tartalmazhat töredéket.")
    path = parsed.path or "/"
    return urlunparse((scheme, parsed.netloc, path, parsed.params, parsed.query, ""))


async def _assert_connectable_cdp_url(url: str) -> None:
    normalized = normalize_cdp_url(url)
    parsed = urlparse(normalized)
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise BrowserCdpError("A böngésző cím gazdagépe hiányzik.")
    if hostname in _allowed_browser_hosts():
        return
    if _private_hosts_allowed():
        return
    literals: list[str] = []
    try:
        ipaddress.ip_address(hostname.split("%", 1)[0])
    except ValueError:
        literals = []
    else:
        literals = [hostname]
    if not literals:
        loop = asyncio.get_running_loop()
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        try:
            infos = await asyncio.wait_for(
                loop.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP),
                timeout=CDP_DNS_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise BrowserCdpError("A böngésző gazdagépe nem oldható fel.") from exc
        literals = [str(info[4][0]) for info in infos]
    if not literals:
        raise BrowserCdpError("A böngésző gazdagépe nem oldható fel.")
    for literal in literals:
        if _blocked_address(literal):
            raise BrowserCdpError("A böngésző cím privát hálózati címre mutat.")


async def _acquire_connection(url: str) -> _CdpConnection:
    loop = asyncio.get_running_loop()
    guard = _registry_guard()
    async with guard:
        _ensure_janitor()
        registry = _registry_table()
        for expired in [item for item in registry.values() if _is_expired(item)]:
            registry.pop(expired.url, None)
            expired.detach(
                BrowserCdpError("A böngésző kapcsolat tétlenség miatt lezárult.")
            )
        connection = registry.get(url)
        if connection is not None and connection.loop is loop and connection.failure is None:
            connection.touch()
            return connection
        if connection is not None:
            registry.pop(url, None)
            connection.detach(connection.failure)
        await _assert_connectable_cdp_url(url)
        while len(registry) >= CDP_MAX_CONNECTIONS:
            idle = [item for item in registry.values() if item.is_idle()]
            if not idle:
                break
            victim = min(idle, key=lambda item: item.last_used)
            registry.pop(victim.url, None)
            victim.detach(
                BrowserCdpError("A böngésző kapcsolat kapacitáskorlát miatt lezárult.")
            )
        connection = _CdpConnection(url, loop)
        registry[url] = connection
        return connection


async def close_all_connections() -> None:
    global _janitor
    janitor = _janitor
    _janitor = None
    if janitor is not None:
        janitor.cancel()
        await _reap(janitor, "cdp-karbantartó")
    guard = _registry_guard()
    async with guard:
        registry = _registry_table()
        connections = list(registry.values())
        registry.clear()
    await asyncio.gather(
        *(
            connection.shutdown(
                BrowserCdpError("A böngésző kapcsolat lezárult.")
            )
            for connection in connections
        ),
        return_exceptions=True,
    )


async def browser_command(
    url: str, action: str, input_data: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not isinstance(url, str) or not url.strip():
        raise BrowserCdpError("Hiányzó böngésző cím.")
    if not isinstance(action, str) or action not in ALLOWED_ACTIONS:
        raise BrowserCdpError("Érvénytelen böngésző művelet.")
    if input_data is not None and not isinstance(input_data, dict):
        raise BrowserCdpError("Érvénytelen beviteli művelet.")
    normalized_url = normalize_cdp_url(url)
    for _attempt in range(MAX_ACQUIRE_ATTEMPTS):
        connection = await _acquire_connection(normalized_url)
        async with connection.command_lock:
            if _registry_table().get(normalized_url) is not connection:
                logger.warning("cdp: a böngészőkapcsolat a várakozás közben lecserélődött.")
                connection.detach(
                    BrowserCdpError("A böngésző kapcsolat lecserélésre került.")
                )
                continue
            connection.touch()
            try:
                return await asyncio.wait_for(
                    connection.execute(action, input_data),
                    timeout=CDP_OPERATION_TIMEOUT_SECONDS,
                )
            except (asyncio.TimeoutError, TimeoutError) as exc:
                raise BrowserCdpError("A böngésző művelet időtúllépett.") from exc
            finally:
                connection.touch()
    raise BrowserCdpError("A böngésző kapcsolat nem érhető el.")