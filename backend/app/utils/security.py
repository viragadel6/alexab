from __future__ import annotations

import re
from urllib.parse import urlparse

from fastapi import Request

from app.config import get_settings

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
LOOSE_UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
BEARER_PATTERN = re.compile(r"^Bearer\s+([A-Za-z0-9\-_.]+)$")
HOST_PATTERN = re.compile(r"^[a-z0-9.-]+$")
CONTROL_MODES = ("agent", "requested", "human")


def is_uuid(value: str | None) -> bool:
    return bool(value) and bool(UUID_PATTERN.match(value))


def is_loose_uuid(value: str | None) -> bool:
    return bool(value) and bool(LOOSE_UUID_PATTERN.match(value))


def is_control_mode(value: object) -> bool:
    return isinstance(value, str) and value in CONTROL_MODES


def get_bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if not header:
        return None
    match = BEARER_PATTERN.match(header.strip())
    return match.group(1) if match else None


def allowed_origins(request: Request) -> set[str]:
    origins: set[str] = set()
    settings = get_settings()
    configured = settings.app_origins or settings.app_origin or ""
    for entry in configured.split(","):
        trimmed = entry.strip()
        if not trimmed:
            continue
        parsed = urlparse(trimmed)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            origins.add(f"{parsed.scheme}://{parsed.netloc}")
    try:
        parsed = urlparse(str(request.url))
        if parsed.scheme and parsed.netloc:
            origins.add(f"{parsed.scheme}://{parsed.netloc}")
    except ValueError:
        host = request.headers.get("host")
        if host:
            scheme = request.headers.get("x-forwarded-proto")
            if scheme not in ("http", "https"):
                scheme = "https"
            origins.add(f"{scheme}://{host}")
    return origins


def verify_origin(request: Request) -> str | None:
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site and fetch_site not in ("same-origin", "none"):
        return "Érvénytelen kérés eredete."
    allowed = allowed_origins(request)
    origin = request.headers.get("origin")
    if origin:
        return None if origin in allowed else "Érvénytelen kérés eredete."
    referer = request.headers.get("referer")
    if referer:
        parsed = urlparse(referer)
        if not parsed.scheme or not parsed.netloc:
            return "Érvénytelen kérés eredete."
        return None if f"{parsed.scheme}://{parsed.netloc}" in allowed else "Érvénytelen kérés eredete."
    return "Érvénytelen kérés eredete."


def host_list(raw: str, message: str) -> list[str]:
    hosts = [item.strip().lower() for item in (raw or "").split(",")]
    hosts = [host for host in hosts if host]
    for host in hosts:
        if (
            not HOST_PATTERN.match(host)
            or host.startswith(".")
            or host.endswith(".")
            or ".." in host
        ):
            raise ValueError(message)
    return hosts


def worker_origin(request_url: str) -> str | None:
    if not get_settings().worker_token:
        return None
    parsed = urlparse(request_url)
    hostname = (parsed.hostname or "").lower()
    if hostname in ("localhost", "127.0.0.1"):
        return None
    if hostname.startswith("id-preview--"):
        return get_settings().published_origin
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"
