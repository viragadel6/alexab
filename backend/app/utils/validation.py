from __future__ import annotations

import json
from typing import Any, NoReturn, TypeGuard

from fastapi import Request

from app.errors import PayloadTooLargeFault, RequestFault
from app.utils.security import LOOSE_UUID_PATTERN

MAX_TEXT_LENGTH = 400_000
MAX_TOTAL_TEXT_LENGTH = 4_000_000
MAX_HISTORY_MESSAGES = 300
MAX_BODY_BYTES = 32 * 1024 * 1024
ALLOWED_CLIENT_ROLES = ("user", "assistant")
MAX_CONTENT_LENGTH_DIGITS = 12


def is_record(value: Any) -> TypeGuard[dict[str, Any]]:
    return isinstance(value, dict)


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"Nem megengedett JSON konstans: {name}")


def _decode_utf8(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RequestFault("Hibás UTF-8 kérés.") from exc


def _loads(text: str) -> Any:
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        raise RequestFault("Hibás JSON kérés.") from exc


def parse_json_body(raw: bytes, limit: int = MAX_BODY_BYTES) -> Any:
    if limit <= 0:
        raise ValueError("A kérésméret korlátjának pozitívnak kell lennie.")
    if not raw:
        raise RequestFault("Hiányzó kéréstartalom.")
    if len(raw) > limit:
        raise PayloadTooLargeFault()
    return _loads(_decode_utf8(raw))


def _declared_length(raw: Any) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise RequestFault("Hibás Content-Length fejléc.", code="INVALID_CONTENT_LENGTH")
    text = raw.strip()
    if not text.isascii() or not text.isdigit() or len(text) > MAX_CONTENT_LENGTH_DIGITS:
        raise RequestFault("Hibás Content-Length fejléc.", code="INVALID_CONTENT_LENGTH")
    return int(text)


async def read_request_body(request: Request, limit: int) -> bytes:
    if limit <= 0:
        raise ValueError("A kérésméret korlátjának pozitívnak kell lennie.")
    declared = _declared_length(request.headers.get("content-length"))
    if declared is not None and declared > limit:
        raise PayloadTooLargeFault()
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        if not chunk:
            continue
        total += len(chunk)
        if total > limit:
            raise PayloadTooLargeFault()
        chunks.append(chunk)
    return b"".join(chunks)


def _encode_check(content: str) -> None:
    try:
        content.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RequestFault("Az üzenet tartalma nem kódolható UTF-8 szövegként.") from exc


def validate_client_history(payload: Any) -> list[dict[str, str]]:
    if not is_record(payload):
        raise RequestFault("Hiányzó vagy hibás üzenetlista.")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_HISTORY_MESSAGES:
        raise RequestFault("Hiányzó vagy hibás üzenetlista.")
    validated: list[dict[str, str]] = []
    last_role: str | None = None
    total_length = 0
    for raw in messages:
        if not is_record(raw):
            raise RequestFault("Az üzenet csak objektum lehet.")
        for key in raw:
            if key not in ("role", "content"):
                raise RequestFault("Kliens oldali utasítás- vagy eszköztörténet nem fogadható el.")
        role = raw.get("role")
        if not isinstance(role, str) or role not in ALLOWED_CLIENT_ROLES:
            raise RequestFault("Az üzenet szerepköre nem megengedett.")
        if last_role is None and role != "user":
            raise RequestFault("Az első üzenetnek felhasználói üzenetnek kell lennie.")
        if last_role is not None and role == last_role:
            raise RequestFault("Az üzenetek szerepkörének váltakoznia kell.")
        content = raw.get("content")
        if not isinstance(content, str) or len(content) > MAX_TEXT_LENGTH or not content.strip():
            raise RequestFault("Az üzenet tartalma hibás vagy túl nagy.")
        total_length += len(content)
        if total_length > MAX_TOTAL_TEXT_LENGTH:
            raise RequestFault("Az üzenetlista teljes tartalma túl nagy.")
        _encode_check(content)
        validated.append({"role": role, "content": content})
        last_role = role
    if last_role != "user":
        raise RequestFault("A legutolsó üzenetnek felhasználói feladatnak kell lennie.")
    return validated


def parse_conversation_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value if LOOSE_UUID_PATTERN.fullmatch(value) is not None else None