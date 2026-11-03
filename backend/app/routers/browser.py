from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app import database as database_module
from app.agent.browser_cdp import browser_command
from app.agent.cognitive import post_operator_message
from app.utils.access import conversation_access, load_access
from app.utils.rate_limit import rate_limiter
from app.utils.security import is_control_mode, is_uuid, verify_origin

router = APIRouter()

NO_STORE_HEADERS = {"Cache-Control": "private, no-store"}
MAX_BODY_BYTES = 16_384
DEFAULT_VIEWPORT_WIDTH = 1280
DEFAULT_VIEWPORT_HEIGHT = 1029
MAX_WHEEL_DELTA = 10_000
MAX_KEY_CODE = 255
MAX_MODIFIERS = 15
MAX_CLICK_COUNT = 3
MAX_MOUSE_BUTTONS = 31
MAX_TEXT_LENGTH = 10_000
MAX_KEY_LENGTH = 40
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100
MAX_OFFSET = 100_000
MOUSE_TYPES = ("mousePressed", "mouseReleased", "mouseMoved")
MOUSE_BUTTONS = ("none", "left", "middle", "right", "back", "forward")
KEY_TYPES = ("keyDown", "keyUp")
AGENT_RESUME_MESSAGE = (
    "A felhasználó visszaadta a böngésző irányítását. Vizsgáld meg az aktuális oldalt, és onnan "
    "folytasd; ne ismételd meg a már elvégzett műveleteket."
)

_session_queues: dict[str, asyncio.Lock] = {}
_last_sequences: dict[str, int] = {}
_auto_sequences: dict[str, int] = {}


def _next_sequence(session_id: str) -> int:
    last = _last_sequences.get(session_id, 0)
    current = _auto_sequences.get(session_id, 0)
    value = max(current, last) + 1
    _auto_sequences[session_id] = value
    return value


def _json(body: Any, status_code: int = 200, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=body,
        headers={**NO_STORE_HEADERS, **(headers or {})},
    )


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


def _is_integer_in_range(value: Any, minimum: int, maximum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum


def _is_finite_in_range(value: Any, minimum: float, maximum: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and minimum <= float(value) <= maximum
    )


def _is_coordinate(value: Any, size: int) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) >= 0
        and float(value) < size
    )


def _is_positive_dimension(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 < value <= 10_000
    )


def _parse_bounded_integer(
    raw: str | None, fallback: int, minimum: int, maximum: int
) -> int | None:
    if raw is None:
        return fallback
    if not re.fullmatch(r"\d+", raw.strip()):
        return None
    value = int(raw.strip())
    if value < minimum or value > maximum:
        return None
    return value


def _optional_modifiers(value: Any) -> int | None:
    if value is None:
        return 0
    if not _is_integer_in_range(value, 0, MAX_MODIFIERS):
        return None
    return value


def _validate_input_shape(value: Any) -> dict[str, Any] | None:
    if not _is_record(value):
        return None
    kind = value.get("kind")
    if kind == "mouse":
        mouse_type = value.get("type")
        button = value.get("button")
        click_count = value.get("clickCount")
        buttons = value.get("buttons")
        modifiers = _optional_modifiers(value.get("modifiers"))
        x = value.get("x")
        y = value.get("y")
        if not isinstance(mouse_type, str) or mouse_type not in MOUSE_TYPES:
            return None
        if not isinstance(button, str) or button not in MOUSE_BUTTONS:
            return None
        if not _is_integer_in_range(click_count, 0, MAX_CLICK_COUNT):
            return None
        if not _is_integer_in_range(buttons, 0, MAX_MOUSE_BUTTONS):
            return None
        if modifiers is None:
            return None
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            return None
        if not isinstance(y, (int, float)) or isinstance(y, bool):
            return None
        if mouse_type != "mouseMoved" and (button == "none" or click_count < 1):
            return None
        return {
            "kind": "mouse",
            "type": mouse_type,
            "x": x,
            "y": y,
            "button": button,
            "clickCount": click_count,
            "buttons": buttons,
            "modifiers": modifiers,
        }
    if kind == "wheel":
        if value.get("type") != "mouseWheel":
            return None
        delta_x = value.get("deltaX")
        delta_y = value.get("deltaY")
        modifiers = _optional_modifiers(value.get("modifiers"))
        x = value.get("x")
        y = value.get("y")
        if not _is_finite_in_range(delta_x, -MAX_WHEEL_DELTA, MAX_WHEEL_DELTA):
            return None
        if not _is_finite_in_range(delta_y, -MAX_WHEEL_DELTA, MAX_WHEEL_DELTA):
            return None
        if modifiers is None:
            return None
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            return None
        if not isinstance(y, (int, float)) or isinstance(y, bool):
            return None
        return {
            "kind": "wheel",
            "type": "mouseWheel",
            "x": x,
            "y": y,
            "deltaX": delta_x,
            "deltaY": delta_y,
            "modifiers": modifiers,
        }
    if kind == "key":
        key_type = value.get("type")
        key = value.get("key")
        code = value.get("code")
        key_code = value.get("keyCode")
        modifiers = value.get("modifiers")
        if not isinstance(key_type, str) or key_type not in KEY_TYPES:
            return None
        if not isinstance(key, str) or not 0 < len(key) <= MAX_KEY_LENGTH:
            return None
        if not isinstance(code, str) or not 0 < len(code) <= MAX_KEY_LENGTH:
            return None
        if not _is_integer_in_range(key_code, 0, MAX_KEY_CODE):
            return None
        if not _is_integer_in_range(modifiers, 0, MAX_MODIFIERS):
            return None
        return {
            "kind": "key",
            "type": key_type,
            "key": key,
            "code": code,
            "keyCode": key_code,
            "modifiers": modifiers,
        }
    if kind == "text":
        text = value.get("text")
        if not isinstance(text, str) or not 0 < len(text) <= MAX_TEXT_LENGTH:
            return None
        return {"kind": "text", "text": text}
    return None


def _input_within_viewport(
    input_data: dict[str, Any], width: int, height: int
) -> bool:
    if input_data["kind"] in ("mouse", "wheel"):
        return _is_coordinate(input_data["x"], width) and _is_coordinate(
            input_data["y"], height
        )
    return True


async def _authenticate(request: Request) -> tuple[str | None, JSONResponse | None]:
    try:
        access = load_access(request)
    except HTTPException as exc:
        return None, _json({"error": exc.detail}, exc.status_code, exc.headers)
    return access.client_id, None


async def _read_json_body(request: Request) -> tuple[Any, JSONResponse | None]:
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        return None, _json({"error": "A kérés tartalma JSON kell legyen."}, 415)
    declared = request.headers.get("content-length")
    if declared is not None and (not declared.isdigit() or int(declared) > MAX_BODY_BYTES):
        return None, _json({"error": "A kérés túl nagy."}, 413)
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        return None, _json({"error": "A kérés túl nagy."}, 413)
    if not raw:
        return None, _json({"error": "Hiányzó kéréstartalom."}, 400)
    try:
        return json.loads(raw.decode("utf-8")), None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, _json({"error": "Érvénytelen JSON."}, 400)


async def _session_lock(session_id: str) -> asyncio.Lock:
    lock = _session_queues.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _session_queues[session_id] = lock
    return lock


async def _load_session(
    session_id: str, client_id: str
) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    database = database_module.get_admin_database()
    response = await (
        database.table("chat_browser_sessions")
        .select(
            "id",
            "cdp_url",
            "control_mode",
            "status",
            "viewport_width",
            "viewport_height",
            "chat_jobs!inner(conversation_id)",
        )
        .eq("id", session_id)
        .limit(1)
        .execute()
    )
    if not response.rows:
        return None, _json({"error": "A böngészőnézet nem található."}, 404)
    row = response.rows[0]
    related_job = row.get("chat_jobs")
    if isinstance(related_job, list):
        related_job = related_job[0] if related_job else None
    conversation_id = (
        related_job.get("conversation_id") if isinstance(related_job, dict) else None
    )
    if not isinstance(conversation_id, str) or not is_uuid(conversation_id):
        return None, _json({"error": "A böngészőnézet nem található."}, 404)
    owner = await conversation_access(database, conversation_id)
    if not owner.owned_by(client_id):
        return None, _json({"error": "A böngészőnézet nem található."}, 404)
    control_mode = row.get("control_mode")
    status = row.get("status")
    cdp_url = row.get("cdp_url")
    if (
        not isinstance(row.get("id"), str)
        or not isinstance(cdp_url, str)
        or not isinstance(control_mode, str)
        or not is_control_mode(control_mode)
        or not isinstance(status, str)
    ):
        return None, _json({"error": "Belső szerverhiba."}, 500)
    viewport_width = row.get("viewport_width")
    viewport_height = row.get("viewport_height")
    return (
        {
            "id": row["id"],
            "cdp_url": cdp_url,
            "control_mode": control_mode,
            "status": status,
            "viewport_width": viewport_width
            if _is_positive_dimension(viewport_width)
            else DEFAULT_VIEWPORT_WIDTH,
            "viewport_height": viewport_height
            if _is_positive_dimension(viewport_height)
            else DEFAULT_VIEWPORT_HEIGHT,
        },
        None,
    )


async def _handle_conversation_list(
    request: Request, conversation: str, client_id: str
) -> JSONResponse:
    rate_limiter.check("list", client_id)
    page_size = _parse_bounded_integer(
        request.query_params.get("limit"), DEFAULT_PAGE_SIZE, 1, MAX_PAGE_SIZE
    )
    offset = _parse_bounded_integer(request.query_params.get("offset"), 0, 0, MAX_OFFSET)
    if page_size is None or offset is None:
        return _json({"error": "Érvénytelen lapozási paraméter."}, 400)
    database = database_module.get_admin_database()
    owner = await conversation_access(database, conversation)
    if not owner.owned_by(client_id):
        return _json({"sessions": [], "nextOffset": None}, 200)
    job_response = await (
        database.table("chat_jobs")
        .select("id")
        .eq("conversation_id", conversation)
        .limit(1)
        .execute()
    )
    if not job_response.rows:
        return _json({"sessions": [], "nextOffset": None}, 200)
    response = await (
        database.table("chat_browser_sessions")
        .select(
            "id",
            "control_mode",
            "status",
            "created_at",
            "chat_jobs!inner(conversation_id)",
        )
        .eq("chat_jobs.conversation_id", conversation)
        .order("created_at", desc=True)
        .range(offset, offset + page_size)
        .execute()
    )
    sessions = []
    for row in response.rows:
        control_mode = row.get("control_mode")
        status = row.get("status")
        created_at = row.get("created_at")
        if (
            isinstance(row.get("id"), str)
            and isinstance(control_mode, str)
            and is_control_mode(control_mode)
            and isinstance(status, str)
            and isinstance(created_at, str)
        ):
            sessions.append(
                {
                    "id": row["id"],
                    "control_mode": control_mode,
                    "status": status,
                    "created_at": created_at,
                }
            )
    has_more = len(sessions) > page_size
    if has_more:
        sessions = sessions[:page_size]
    return _json(
        {"sessions": sessions, "nextOffset": offset + page_size if has_more else None}, 200
    )


async def _handle_session_status(
    request: Request, session_id: str, client_id: str
) -> JSONResponse:
    wants_frame = "frame" in request.query_params
    rate_limiter.check("frame" if wants_frame else "status", f"{client_id}:{session_id}")
    session, failure = await _load_session(session_id, client_id)
    if failure is not None:
        return failure
    assert session is not None
    if not wants_frame:
        return _json(
            {
                "id": session["id"],
                "control_mode": session["control_mode"],
                "status": session["status"],
            }
        )
    if session["status"] != "active":
        return _json(
            {
                "id": session["id"],
                "error": "A böngésző már leállt.",
                "control_mode": session["control_mode"],
                "status": session["status"],
            },
            409,
        )
    try:
        frame = await browser_command(session["cdp_url"], "frame")
    except Exception:
        return _json({"id": session["id"], "error": "A képernyő nem érhető el."}, 502)
    payload = frame if isinstance(frame, dict) else {"result": frame}
    return _json(
        {
            **payload,
            "id": session["id"],
            "control_mode": session["control_mode"],
            "status": session["status"],
        }
    )


async def _handle_input(
    session_id: str, sequence: int, input_data: dict[str, Any], client_id: str
) -> JSONResponse:
    rate_limiter.check("input", f"{client_id}:{session_id}")
    lock = await _session_lock(session_id)
    async with lock:
        last_sequence = _last_sequences.get(session_id)
        if last_sequence is not None and sequence <= last_sequence:
            return _json(
                {"error": "Elavult vagy ismételt bevitel.", "lastSeq": last_sequence}, 409
            )
        session, failure = await _load_session(session_id, client_id)
        if failure is not None:
            return failure
        assert session is not None
        if session["status"] != "active":
            return _json({"error": "A böngésző már leállt."}, 409)
        if session["control_mode"] != "human":
            return _json({"error": "Előbb vedd át az irányítást."}, 409)
        if not _input_within_viewport(
            input_data, session["viewport_width"], session["viewport_height"]
        ):
            return _json({"error": "Érvénytelen bevitel."}, 400)
        _last_sequences[session_id] = sequence
        if len(_last_sequences) > 10_000:
            for key in list(_last_sequences.keys())[:1000]:
                if key != session_id:
                    _last_sequences.pop(key, None)
                    _auto_sequences.pop(key, None)
        try:
            result = await browser_command(session["cdp_url"], "input", input_data)
        except Exception:
            return _json({"error": "A bevitel sikertelen."}, 502)
        return _json({"ok": True, "seq": sequence, "result": result})


async def _handle_takeover(session_id: str, client_id: str) -> JSONResponse:
    session, failure = await _load_session(session_id, client_id)
    if failure is not None:
        return failure
    database = database_module.get_admin_database()
    granted = await (
        database.table("chat_browser_sessions")
        .update({"control_mode": "human"})
        .eq("id", session_id)
        .eq("status", "active")
        .eq("agent_busy", False)
        .in_("control_mode", ["agent", "requested"])
        .execute()
    )
    if granted.rows:
        return _json({"ok": True, "id": session_id, "control_mode": "human"})
    requested = await (
        database.table("chat_browser_sessions")
        .update({"control_mode": "requested"})
        .eq("id", session_id)
        .eq("status", "active")
        .eq("control_mode", "agent")
        .execute()
    )
    if requested.rows:
        return _json({"ok": True, "id": session_id, "control_mode": "requested"}, 202)
    session, failure = await _load_session(session_id, client_id)
    if failure is not None:
        return failure
    assert session is not None
    if session["status"] != "active":
        return _json({"error": "A böngésző már leállt."}, 409)
    if session["control_mode"] == "human":
        return _json({"ok": True, "id": session_id, "control_mode": "human"})
    if session["control_mode"] == "requested":
        return _json({"ok": True, "id": session_id, "control_mode": "requested"}, 202)
    return _json({"error": "Az irányítás átvétele nem sikerült. Próbáld újra."}, 409)


async def _handle_handback(session_id: str, client_id: str) -> JSONResponse:
    session, failure = await _load_session(session_id, client_id)
    if failure is not None:
        return failure
    database = database_module.get_admin_database()
    updated = await (
        database.table("chat_browser_sessions")
        .update({"control_mode": "agent"})
        .eq("id", session_id)
        .eq("status", "active")
        .in_("control_mode", ["human", "requested"])
        .execute()
    )
    if not updated.rows:
        session, failure = await _load_session(session_id, client_id)
        if failure is not None:
            return failure
        assert session is not None
        if session["status"] != "active":
            return _json({"error": "A böngésző már leállt."}, 409)
        if session["control_mode"] == "agent":
            return _json({"ok": True, "id": session_id, "control_mode": "agent"})
        return _json({"error": "Az irányítás visszaadása nem sikerült. Próbáld újra."}, 409)
    job_id = updated.rows[0].get("job_id")
    if not isinstance(job_id, str) or not job_id:
        return _json({"error": "Belső szerverhiba."}, 500)
    try:
        await post_operator_message(job_id, AGENT_RESUME_MESSAGE)
    except Exception:
        await (
            database.table("chat_browser_sessions")
            .update({"control_mode": "human"})
            .eq("id", session_id)
            .eq("status", "active")
            .eq("control_mode", "agent")
            .execute()
        )
        return _json(
            {"error": "Az ügynök értesítése sikertelen; az irányítás nálad maradt."}, 502
        )
    return _json({"ok": True, "id": session_id, "control_mode": "agent"})


@router.get("/api/browser")
async def get_browser(request: Request) -> Any:
    try:
        client_id, failure = await _authenticate(request)
        if failure is not None:
            return failure
        assert client_id is not None
        conversation = request.query_params.get("conversation")
        if conversation is not None:
            if not is_uuid(conversation):
                return _json({"error": "Érvénytelen beszélgetés."}, 400)
            return await _handle_conversation_list(request, conversation, client_id)
        session_id = request.query_params.get("id")
        if not is_uuid(session_id):
            return _json({"error": "Érvénytelen böngészőnézet."}, 400)
        return await _handle_session_status(request, session_id, client_id)
    except Exception:
        return _json({"error": "Belső szerverhiba."}, 500)


@router.post("/api/browser")
async def post_browser(request: Request) -> Any:
    try:
        origin_error = verify_origin(request)
        if origin_error is not None:
            return _json({"error": origin_error}, 403)
        client_id, failure = await _authenticate(request)
        if failure is not None:
            return failure
        assert client_id is not None
        body, failure = await _read_json_body(request)
        if failure is not None:
            return failure
        if not _is_record(body):
            return _json({"error": "Érvénytelen kérés."}, 400)
        if "input" in body:
            session_id = body.get("id")
            if not is_uuid(session_id):
                return _json({"error": "Érvénytelen bevitel."}, 400)
            sequence = body.get("seq")
            if sequence is None:
                sequence = _next_sequence(session_id)
            elif not _is_integer_in_range(sequence, 0, 2**53 - 1):
                return _json({"error": "Érvénytelen bevitel."}, 400)
            input_data = _validate_input_shape(body.get("input"))
            if input_data is None:
                return _json({"error": "Érvénytelen bevitel."}, 400)
            return await _handle_input(session_id, sequence, input_data, client_id)
        session_id = body.get("id")
        mode = body.get("mode")
        if not is_uuid(session_id) or mode not in ("human", "agent"):
            return _json({"error": "Érvénytelen irányítási kérés."}, 400)
        rate_limiter.check("mode", f"{client_id}:{session_id}")
        lock = await _session_lock(session_id)
        async with lock:
            if mode == "human":
                return await _handle_takeover(session_id, client_id)
            return await _handle_handback(session_id, client_id)
    except Exception:
        return _json({"error": "Belső szerverhiba."}, 500)
