from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any

from fastapi import HTTPException, Request
from pydantic import SecretBytes, SecretStr

from app.config import get_settings
from app.database import SupabaseDatabase
from app.utils.clock import as_str

SESSION_COOKIE = "chat_session"
SESSION_TOKEN_VERSION = "v1"
SESSION_MAX_AGE_SECONDS = 31_536_000
SESSION_CLOCK_SKEW_SECONDS = 300
SESSION_ID_LENGTH = 32
SESSION_HEX_DIGITS = frozenset("0123456789abcdef")
SESSION_BOOTSTRAP_HEADER = "x-session-bootstrap"
SESSION_BOOTSTRAP_LENGTH = 64
SESSION_SCOPE_KEY = "app.access.session"
ADMIN_TOKEN_HEADER = "x-admin-token"
WORKER_TOKEN_HEADER = "x-worker-token"
MAX_TOKEN_LENGTH = 4096

logger = logging.getLogger("app.access")


def _secret_value(value: Any) -> str:
    if isinstance(value, (SecretStr, SecretBytes)):
        value = value.get_secret_value()
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise TypeError("The configured secret must be a string or UTF-8 bytes.")
    value.encode("utf-8")
    return value


def _session_secret() -> str:
    try:
        configured = _secret_value(get_settings().session_secret)
    except (TypeError, UnicodeError):
        configured = ""
    if not configured.strip():
        raise HTTPException(
            status_code=503,
            detail=(
                "A session_secret értékét tartósan és minden folyamatban "
                "azonosan kell beállítani."
            ),
            headers={"Cache-Control": "no-store"},
        )
    return configured


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in SESSION_HEX_DIGITS for character in value)
    )


def _is_valid_client_id(value: Any) -> bool:
    return _is_lower_hex(value, SESSION_ID_LENGTH)


def _signature(payload: str) -> str:
    return hmac.new(
        _session_secret().encode("utf-8"),
        payload.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()


def _encode_session(client_id: str) -> str:
    if not _is_valid_client_id(client_id):
        raise ValueError("client_id must contain 32 lowercase hexadecimal characters.")
    issued_at = int(time.time())
    if issued_at <= 0 or issued_at >= 10**12:
        raise ValueError("The current timestamp cannot be represented by this format.")
    payload = f"{SESSION_TOKEN_VERSION}.{client_id}.{issued_at}"
    return f"{payload}.{_signature(payload)}"


def _decode_session(token: Any) -> str | None:
    if not isinstance(token, str) or len(token) > 256 or not token.isascii():
        return None
    parts = token.split(".")
    if len(parts) != 4:
        return None
    version, client_id, issued_raw, signature = parts
    if version != SESSION_TOKEN_VERSION:
        return None
    if not _is_valid_client_id(client_id):
        return None
    if (
        not issued_raw
        or len(issued_raw) > 12
        or not issued_raw.isdigit()
        or issued_raw.startswith("0")
    ):
        return None
    if not _is_lower_hex(signature, hashlib.sha256().digest_size * 2):
        return None
    issued_at = int(issued_raw)
    current = int(time.time())
    if issued_at > current + SESSION_CLOCK_SKEW_SECONDS:
        return None
    if current - issued_at >= SESSION_MAX_AGE_SECONDS:
        return None
    payload = f"{version}.{client_id}.{issued_raw}"
    if not hmac.compare_digest(signature, _signature(payload)):
        return None
    return client_id


def _is_secure_request(request: Request) -> bool:
    return request.url.scheme == "https"


def _cookie_header(client_id: str, secure: bool) -> str:
    attributes = [
        f"{SESSION_COOKIE}={_encode_session(client_id)}",
        "Path=/",
        f"Max-Age={SESSION_MAX_AGE_SECONDS}",
        "HttpOnly",
        "SameSite=Lax",
    ]
    if secure:
        attributes.append("Secure")
    return "; ".join(attributes)


def _derive_client_id(request: Request) -> str:
    supplied = request.headers.get(SESSION_BOOTSTRAP_HEADER)
    if supplied is None or not supplied.strip(" \t"):
        raise HTTPException(
            status_code=428,
            detail="A kérés nem feldolgozható.",
            headers={"Cache-Control": "no-store"},
        )
    if len(supplied) > MAX_TOKEN_LENGTH:
        raise HTTPException(
            status_code=400,
            detail="Érvénytelen munkamenet-indító kulcs.",
            headers={"Cache-Control": "no-store"},
        )
    bootstrap = supplied.strip(" \t")
    if not _is_lower_hex(bootstrap, SESSION_BOOTSTRAP_LENGTH):
        raise HTTPException(
            status_code=400,
            detail="Érvénytelen munkamenet-indító kulcs.",
            headers={"Cache-Control": "no-store"},
        )
    payload = f"chat-session:bootstrap:{SESSION_TOKEN_VERSION}:{bootstrap}"
    return hmac.new(
        _session_secret().encode("utf-8"),
        payload.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()[:SESSION_ID_LENGTH]


@dataclass(frozen=True)
class Access:
    client_id: str
    set_cookie: str | None

    def headers(self) -> dict[str, str]:
        if self.set_cookie is None:
            return {}
        return {
            "Set-Cookie": self.set_cookie,
            "Cache-Control": "no-store",
        }


def load_access(request: Request) -> Access:
    cached = request.scope.get(SESSION_SCOPE_KEY)
    if isinstance(cached, Access):
        return cached
    _session_secret()
    token = request.cookies.get(SESSION_COOKIE)
    if token is None:
        client_id = _derive_client_id(request)
    else:
        client_id = _decode_session(token)
        if client_id is None:
            raise HTTPException(
                status_code=401,
                detail=(
                    "A munkamenet érvénytelen vagy lejárt. "
                    "A korábbi kliensazonosító nem cserélhető le automatikusan."
                ),
                headers={"Cache-Control": "no-store"},
            )
    access = Access(
        client_id=client_id,
        set_cookie=_cookie_header(client_id, _is_secure_request(request)),
    )
    request.scope[SESSION_SCOPE_KEY] = access
    return access


def _header_bytes(value: Any) -> bytes | None:
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        try:
            raw = value.encode("latin-1")
        except UnicodeEncodeError:
            return None
    else:
        return None
    if any((byte < 32 and byte != 9) or byte == 127 for byte in raw):
        return None
    return raw


def _expected_bytes(value: Any) -> bytes | None:
    if not isinstance(value, str):
        return None
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError:
        return None


def _normalized_token(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) > MAX_TOKEN_LENGTH:
        return None
    token = value.strip()
    if not token:
        return None
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in token
    ):
        return None
    encoded = _expected_bytes(token)
    if encoded is None or len(encoded) > MAX_TOKEN_LENGTH:
        return None
    return token


def _clean_header_token(value: Any) -> str | None:
    raw = _header_bytes(value)
    if raw is None or len(raw) > MAX_TOKEN_LENGTH:
        return None
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return _normalized_token(decoded)


def _bearer_token(request: Request) -> str | None:
    raw = _header_bytes(request.headers.get("authorization"))
    if raw is None or len(raw) > MAX_TOKEN_LENGTH + len(b"Bearer "):
        return None
    parts = raw.strip(b" \t").split(None, 1)
    if len(parts) != 2 or parts[0].lower() != b"bearer":
        return None
    return _clean_header_token(parts[1])


def _token_matches(provided: str | None, expected: str) -> bool:
    if not provided or not expected:
        return False
    provided_bytes = _expected_bytes(provided)
    expected_bytes = _expected_bytes(expected)
    if provided_bytes is None or expected_bytes is None:
        return False
    return secrets.compare_digest(provided_bytes, expected_bytes)


def _token_error(
    request: Request,
    header_name: str,
    expected: Any,
    missing_message: str,
) -> tuple[str, int] | None:
    try:
        configured = _normalized_token(_secret_value(expected))
    except (TypeError, UnicodeError):
        configured = None
    if configured is None:
        logger.error("access_token_not_configured", extra={"header": header_name})
        return (missing_message, 503)
    provided = _clean_header_token(request.headers.get(header_name))
    if provided is None:
        provided = _bearer_token(request)
    if provided is None:
        return ("Bejelentkezés szükséges.", 401)
    if not _token_matches(provided, configured):
        return ("Tiltott művelet.", 403)
    return None


def admin_error(request: Request) -> tuple[str, int] | None:
    return _token_error(
        request,
        ADMIN_TOKEN_HEADER,
        get_settings().admin_token,
        "A rendszerfelügyeleti végpont nincs konfigurálva.",
    )


def worker_error(request: Request) -> tuple[str, int] | None:
    return _token_error(
        request,
        WORKER_TOKEN_HEADER,
        get_settings().worker_token,
        "A háttérmunkás végpont nincs konfigurálva.",
    )


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return str(value)
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _canonical_uuid(value: Any) -> str | None:
    if isinstance(value, uuid.UUID):
        return str(value)
    if not isinstance(value, str) or len(value) not in (32, 36):
        return None
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return None
    canonical = str(parsed)
    if value.lower() not in (parsed.hex, canonical):
        return None
    return canonical


def _normalize_identifier(value: Any) -> str | None:
    canonical = _canonical_uuid(value)
    if canonical is None:
        return None
    return canonical.replace("-", "")


def _first_row(response: Any) -> dict[str, Any] | None:
    data = getattr(response, "data", response)
    if isinstance(data, Mapping) and "data" in data:
        data = data["data"]
    if isinstance(data, Mapping):
        return dict(data)
    if (
        isinstance(data, (list, tuple))
        and len(data) == 1
        and isinstance(data[0], Mapping)
    ):
        return dict(data[0])
    return None


async def _select_one(
    database: SupabaseDatabase,
    table_name: str,
    columns: str,
    identifier: str,
) -> dict[str, Any] | None:
    try:
        response = await (
            database.table(table_name)
            .select(columns)
            .eq("id", identifier)
            .limit(1)
            .execute()
        )
        data = getattr(response, "data", response)
        if data is not None and not isinstance(data, (Mapping, list, tuple)):
            reader = getattr(response, "maybe_single", None)
            if callable(reader):
                data = reader()
                if isawaitable(data):
                    data = await data
        row = _first_row(data)
        if row is None or _canonical_uuid(row.get("id")) != identifier:
            return None
        return row
    except Exception:
        return None


@dataclass(frozen=True)
class ConversationAccess:
    exists: bool
    owner: str | None

    def owned_by(self, client_id: str) -> bool:
        if not self.exists:
            return False
        owner = _normalize_identifier(self.owner)
        candidate = _normalize_identifier(client_id)
        if owner is None or candidate is None:
            return False
        return secrets.compare_digest(owner, candidate)


@dataclass(frozen=True)
class JobAccess:
    exists: bool
    conversation_id: str | None
    owner: str | None
    status: str | None

    def owned_by(self, client_id: str) -> bool:
        if not self.exists:
            return False
        owner = _normalize_identifier(self.owner)
        candidate = _normalize_identifier(client_id)
        if owner is None or candidate is None:
            return False
        return secrets.compare_digest(owner, candidate)


async def conversation_access(
    database: SupabaseDatabase,
    conversation_id: str,
) -> ConversationAccess:
    identifier = _canonical_uuid(conversation_id)
    if identifier is None:
        return ConversationAccess(exists=False, owner=None)
    row = await _select_one(
        database,
        "chat_conversations",
        "id, client_id",
        identifier,
    )
    if row is None:
        return ConversationAccess(exists=False, owner=None)
    return ConversationAccess(
        exists=True,
        owner=_normalize_identifier(row.get("client_id")),
    )


async def job_access(
    database: SupabaseDatabase,
    job_id: str,
) -> JobAccess:
    identifier = _canonical_uuid(job_id)
    if identifier is None:
        return JobAccess(
            exists=False,
            conversation_id=None,
            owner=None,
            status=None,
        )
    row = await _select_one(
        database,
        "chat_jobs",
        "id, conversation_id, status",
        identifier,
    )
    if row is None:
        return JobAccess(
            exists=False,
            conversation_id=None,
            owner=None,
            status=None,
        )
    conversation_id = _canonical_uuid(row.get("conversation_id"))
    owner: str | None = None
    if conversation_id is not None:
        conversation = await conversation_access(database, conversation_id)
        if conversation.exists:
            owner = conversation.owner
        else:
            conversation_id = None
    return JobAccess(
        exists=True,
        conversation_id=conversation_id,
        owner=owner,
        status=_as_optional_str(row.get("status")),
    )