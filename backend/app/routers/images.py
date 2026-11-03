from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import json
import logging
import re
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import FormData, UploadFile

from app.config import get_settings
from app.errors import RateLimitFault
from app.utils.rate_limit import client_key, rate_limiter
from app.utils.security import verify_origin

logger = logging.getLogger("app.images")

router = APIRouter()

DEFAULT_IMAGE_MODEL = "flymyai/nano-banana-2_1"
DEFAULT_BASE_URL = "https://api.flymy.ai"

ASPECT_RATIOS: tuple[str, ...] = (
    "auto",
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
    "1:4",
    "4:1",
    "1:8",
    "8:1",
)
SAFETY_TOLERANCES: tuple[str, ...] = ("1", "2", "3", "4", "5", "6")
REFERENCE_MIME_TYPES: tuple[str, ...] = (
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
    "image/gif",
)

DEFAULT_ASPECT_RATIO = "auto"
DEFAULT_RESOLUTION = "4K"
DEFAULT_NUM_IMAGES = 1
DEFAULT_THINKING_LEVEL = "high"
DEFAULT_SAFETY_TOLERANCE = "4"
DEFAULT_ENABLE_WEB_SEARCH = False

MAX_REFERENCE_IMAGES = 14
MAX_IMAGES_PER_REQUEST = 10
MAX_OUTPUT_IMAGES = 16
ASYNC_NUM_IMAGES_THRESHOLD = 4

MAX_REFERENCE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_REFERENCE_BYTES = 80 * 1024 * 1024
MAX_FORM_BODY_BYTES = MAX_TOTAL_REFERENCE_BYTES + 4 * 1024 * 1024
MAX_JSON_BODY_BYTES = 64 * 1024 * 1024
MAX_DATA_URL_CHARS = 20 * 1024 * 1024
MAX_RESPONSE_BYTES = 96 * 1024 * 1024
INLINE_IMAGE_MAX_BYTES = 9 * 1024 * 1024
INLINE_TOTAL_MAX_BYTES = 24 * 1024 * 1024
DOWNLOAD_IMAGE_MAX_BYTES = 48 * 1024 * 1024
MAX_ERROR_BYTES = 64 * 1024
MAX_ERROR_CHARS = 400
MAX_DESCRIPTION_CHARS = 2000
MAX_PROMPT_CHARS = 32_000
MAX_URL_CHARS = 4096

CONNECT_TIMEOUT_SECONDS = 15.0
CALL_TIMEOUT_SECONDS = 120.0
CANCEL_TIMEOUT_SECONDS = 20.0
PREDICT_DEADLINE_SECONDS = 900.0
ASYNC_POLL_SECONDS = 1.5
REFERENCE_DOWNLOAD_TIMEOUT_SECONDS = 120.0
DNS_TIMEOUT_SECONDS = 5.0
REMOTE_LINK_HOURS = 24

ALLOWED_REFERENCE_TYPES = frozenset(REFERENCE_MIME_TYPES) | {"image/jpg"}
REFERENCE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/heif": ".heif",
    "image/heic": ".heic",
    "image/gif": ".gif",
}
DATA_URL_PATTERN = re.compile(
    r"^data:(image/[A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=\r\n]+)$", re.IGNORECASE
)
FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]")
LEGACY_SIZES = {
    "auto": "auto",
    "square": "1:1",
    "landscape": "3:2",
    "portrait": "2:3",
    "1024x1024": "1:1",
    "1536x1024": "3:2",
    "1024x1536": "2:3",
    "1792x1024": "16:9",
    "1024x1792": "9:16",
}
TRUTHY_VALUES = frozenset({"1", "true", "yes", "on", "igen"})
FALSY_VALUES = frozenset({"0", "false", "no", "off", "nem", ""})
REFERENCE_KEYS = ("image_urls", "images", "image", "references", "input_images")
OPTION_KEYS = (
    "prompt",
    "aspect_ratio",
    "resolution",
    "num_images",
    "thinking_level",
    "enable_web_search",
    "web_search",
    "safety_tolerance",
    "async",
    "size",
    "quality",
)


class ImageRequestError(Exception):
    def __init__(
        self, message: str, status_code: int = 400, code: str = "INVALID_IMAGE_REQUEST"
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class _UseSyncRoute(Exception):
    def __init__(self, detail: str = "") -> None:
        super().__init__(detail or "A FlyMyAI sor nem elérhető.")
        self.detail = detail


@dataclass(frozen=True)
class ImageReference:
    filename: str
    mime_type: str
    url: str | None = None
    data: bytes | None = None

    @property
    def is_remote(self) -> bool:
        return self.data is None and bool(self.url)


@dataclass(frozen=True)
class ImageRequest:
    prompt: str
    references: tuple[ImageReference, ...]
    aspect_ratio: str
    resolution: str
    num_images: int
    thinking_level: str
    enable_web_search: bool
    safety_tolerance: str
    prefer_async: bool

    @property
    def mode(self) -> str:
        return "edit" if self.references else "generate"

    @property
    def needs_queue(self) -> bool:
        return (
            self.prefer_async
            or self.num_images > ASYNC_NUM_IMAGES_THRESHOLD
            or self.resolution == "4K"
        )

    def with_references(self, references: tuple[ImageReference, ...]) -> "ImageRequest":
        return ImageRequest(
            prompt=self.prompt,
            references=references,
            aspect_ratio=self.aspect_ratio,
            resolution=self.resolution,
            num_images=self.num_images,
            thinking_level=self.thinking_level,
            enable_web_search=self.enable_web_search,
            safety_tolerance=self.safety_tolerance,
            prefer_async=self.prefer_async,
        )


def _json(payload: dict[str, Any], status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content=payload, headers={"Cache-Control": "no-store"}
    )


def _failure(
    message: str, status_code: int = 400, code: str = "IMAGE_REQUEST_FAILED"
) -> JSONResponse:
    return _json({"error": {"code": code, "message": message}}, status_code)


def _rate_limit_failure(exc: RateLimitFault) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"error": {"code": exc.code, "message": exc.message}},
        headers={"Cache-Control": "no-store", "Retry-After": str(exc.retry_after)},
    )


def _clip(value: Any, limit: int = MAX_ERROR_CHARS) -> str:
    text = " ".join(str(value if value is not None else "").split())
    return text[:limit]


def _model_name() -> str:
    configured = str(get_settings().flymyai_image_model or "").strip()
    return configured or DEFAULT_IMAGE_MODEL


def _base_url() -> str:
    configured = str(get_settings().flymyai_base_url or "").strip()
    return (configured or DEFAULT_BASE_URL).rstrip("/")


def _predict_url() -> str:
    return f"{_base_url()}/api/v1/{_model_name()}/predict"


def _async_submit_url() -> str:
    return f"{_base_url()}/api/v1/{_model_name()}/predict/async/"


def _async_result_url() -> str:
    return f"{_base_url()}/api/v1/{_model_name()}/predict/async/result/"


def _cancel_url() -> str:
    return f"{_base_url()}/api/v1/{_model_name()}/predict/cancel/"


def _api_key() -> str:
    key = str(get_settings().flymyai_api_key or "").strip()
    if not key:
        raise ImageRequestError(
            "A FlyMyAI kulcs nincs beállítva.", 503, "IMAGE_PROVIDER_UNAVAILABLE"
        )
    return key


def _timeout(read_seconds: float, connect: float = CONNECT_TIMEOUT_SECONDS) -> httpx.Timeout:
    return httpx.Timeout(
        read_seconds,
        connect=connect,
        write=max(30.0, min(read_seconds, CALL_TIMEOUT_SECONDS)),
        pool=connect,
    )


def _text_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _sniff_mime(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[4:8] == b"ftyp":
        brand = data[8:12].lower()
        if brand in (b"heic", b"heix", b"hevc", b"heim", b"heis"):
            return "image/heic"
        if brand in (b"heif", b"mif1", b"msf1"):
            return "image/heif"
    return None


def _normalized_mime(value: Any, data: bytes | None) -> str:
    candidate = _text_value(value).lower().split(";")[0].strip()
    if candidate in ALLOWED_REFERENCE_TYPES:
        return "image/jpeg" if candidate == "image/jpg" else candidate
    sniffed = _sniff_mime(data) if data else None
    if sniffed is not None:
        return sniffed
    return candidate


def _safe_filename(value: Any, mime_type: str, index: int) -> str:
    raw = _text_value(value)
    name = raw.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    name = FILENAME_PATTERN.sub("_", name)[:80].strip("._")
    extension = REFERENCE_EXTENSIONS.get(mime_type, ".png")
    if not name:
        return f"reference-{index + 1}{extension}"
    if not name.lower().endswith(tuple(REFERENCE_EXTENSIONS.values())):
        return f"{name}{extension}"
    return name


def _blocked_address(value: str) -> bool:
    candidate = value.split("%", 1)[0]
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return True
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


async def _assert_public_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ImageRequestError(
            "A kép címe csak http vagy https cím lehet.", 400, "INVALID_REFERENCE"
        )
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise ImageRequestError(
            "A kép címéből hiányzik a gazdagép.", 400, "INVALID_REFERENCE"
        )
    if parsed.username or parsed.password:
        raise ImageRequestError(
            "A kép címe nem tartalmazhat hitelesítő adatot.", 400, "INVALID_REFERENCE"
        )
    if len(url) > MAX_URL_CHARS:
        raise ImageRequestError("A kép címe túl hosszú.", 400, "INVALID_REFERENCE")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        literal = False
    else:
        literal = True
    if literal:
        if _blocked_address(hostname):
            raise ImageRequestError(
                "A kép címe nem mutathat belső hálózatra.", 403, "REFERENCE_DENIED"
            )
        return
    loop = asyncio.get_running_loop()
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = await asyncio.wait_for(
            loop.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP),
            timeout=DNS_TIMEOUT_SECONDS,
        )
    except (socket.gaierror, asyncio.TimeoutError, UnicodeError, OSError) as exc:
        raise ImageRequestError(
            "A kép gazdagépe nem oldható fel.", 400, "INVALID_REFERENCE"
        ) from exc
    if not infos:
        raise ImageRequestError(
            "A kép gazdagépe nem oldható fel.", 400, "INVALID_REFERENCE"
        )
    for info in infos:
        if _blocked_address(str(info[4][0])):
            raise ImageRequestError(
                "A kép címe nem mutathat belső hálózatra.", 403, "REFERENCE_DENIED"
            )


def _reference_from_bytes(data: bytes, filename: Any, declared: Any, index: int) -> ImageReference:
    if not data:
        raise ImageRequestError("A referencia kép üres.", 400, "INVALID_REFERENCE")
    if len(data) > MAX_REFERENCE_BYTES:
        raise ImageRequestError(
            f"Egy referencia kép legfeljebb {MAX_REFERENCE_BYTES // (1024 * 1024)} MB lehet.",
            413,
            "REFERENCE_TOO_LARGE",
        )
    mime_type = _normalized_mime(declared, data)
    if mime_type not in ALLOWED_REFERENCE_TYPES:
        raise ImageRequestError(
            "A referencia kép formátuma nem támogatott (PNG, JPEG, WebP, HEIF, GIF).",
            415,
            "REFERENCE_UNSUPPORTED",
        )
    return ImageReference(
        filename=_safe_filename(filename, mime_type, index),
        mime_type=mime_type,
        data=data,
    )


async def _read_upload(upload: UploadFile, index: int) -> ImageReference:
    declared_size = upload.size
    if isinstance(declared_size, int) and declared_size > MAX_REFERENCE_BYTES:
        raise ImageRequestError(
            f"Egy referencia kép legfeljebb {MAX_REFERENCE_BYTES // (1024 * 1024)} MB lehet.",
            413,
            "REFERENCE_TOO_LARGE",
        )
    try:
        data = await upload.read()
    except Exception as exc:
        raise ImageRequestError(
            "A feltöltött referencia kép nem olvasható.", 400, "INVALID_REFERENCE"
        ) from exc
    return _reference_from_bytes(data, upload.filename, upload.content_type, index)


def _reference_from_data_url(value: str, index: int) -> ImageReference:
    text = value.strip()
    if len(text) > MAX_DATA_URL_CHARS:
        raise ImageRequestError(
            f"Egy referencia kép legfeljebb {MAX_REFERENCE_BYTES // (1024 * 1024)} MB lehet.",
            413,
            "REFERENCE_TOO_LARGE",
        )
    match = DATA_URL_PATTERN.match(text)
    if match is None:
        raise ImageRequestError(
            "A beágyazott referencia kép nem értelmezhető.", 400, "INVALID_REFERENCE"
        )
    try:
        data = base64.b64decode(match.group(2), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageRequestError(
            "A beágyazott referencia kép nem értelmezhető.", 400, "INVALID_REFERENCE"
        ) from exc
    return _reference_from_bytes(data, f"reference-{index + 1}", match.group(1), index)


def _reference_from_url(value: str, index: int) -> ImageReference:
    url = value.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ImageRequestError(
            "A referencia kép címe csak http vagy https cím lehet.", 400, "INVALID_REFERENCE"
        )
    if len(url) > MAX_URL_CHARS:
        raise ImageRequestError("A referencia kép címe túl hosszú.", 400, "INVALID_REFERENCE")
    name = parsed.path.rsplit("/", 1)[-1]
    return ImageReference(
        filename=_safe_filename(name, "image/png", index),
        mime_type="image/png",
        url=url,
    )


def _json_array(text: str) -> list[Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ImageRequestError(
            "A referencia kép lista nem értelmezhető.", 400, "INVALID_REFERENCE"
        ) from exc
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, str):
        return [parsed]
    raise ImageRequestError(
        "A referencia kép lista nem értelmezhető.", 400, "INVALID_REFERENCE"
    )


async def _reference_from_value(value: Any, index: int) -> ImageReference:
    if isinstance(value, (bytes, bytearray)):
        return _reference_from_bytes(bytes(value), f"reference-{index + 1}", None, index)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ImageRequestError(
                "A referencia kép megadása üres.", 400, "INVALID_REFERENCE"
            )
        if text.startswith("data:"):
            return _reference_from_data_url(text, index)
        if text.startswith("["):
            entries = _json_array(text)
            if not entries:
                raise ImageRequestError(
                    "A referencia kép lista üres.", 400, "INVALID_REFERENCE"
                )
            return await _reference_from_value(entries[0], index)
        return _reference_from_url(text, index)
    if isinstance(value, UploadFile):
        return await _read_upload(value, index)
    if isinstance(value, dict):
        candidate = value.get("url") or value.get("image_url") or value.get("data")
        if candidate is None:
            raise ImageRequestError(
                "A referencia kép megadása nem értelmezhető.", 400, "INVALID_REFERENCE"
            )
        return await _reference_from_value(candidate, index)
    if isinstance(value, list):
        if not value:
            raise ImageRequestError(
                "A referencia kép lista üres.", 400, "INVALID_REFERENCE"
            )
        return await _reference_from_value(value[0], index)
    raise ImageRequestError(
        "A referencia kép megadása nem értelmezhető.", 400, "INVALID_REFERENCE"
    )


async def _collect_references(values: list[Any]) -> tuple[ImageReference, ...]:
    references: list[ImageReference] = []
    for value in values:
        if isinstance(value, str) and value.strip().startswith("["):
            entries: list[Any] = _json_array(value)
        elif isinstance(value, list):
            entries = value
        else:
            entries = [value]
        for entry in entries:
            if entry is None:
                continue
            if isinstance(entry, str) and not entry.strip():
                continue
            if len(references) >= MAX_REFERENCE_IMAGES:
                raise ImageRequestError(
                    f"Legfeljebb {MAX_REFERENCE_IMAGES} referencia kép adható meg.",
                    400,
                    "TOO_MANY_REFERENCES",
                )
            references.append(await _reference_from_value(entry, len(references)))
    total = sum(len(item.data or b"") for item in references)
    if total > MAX_TOTAL_REFERENCE_BYTES:
        raise ImageRequestError(
            "A referencia képek összesített mérete túl nagy.", 413, "REFERENCE_TOO_LARGE"
        )
    return tuple(references)


def _validated_choice(value: Any, allowed: tuple[str, ...], fallback: str, label: str) -> str:
    text = _text_value(value)
    if not text:
        return fallback
    for option in allowed:
        if text.lower() == option.lower():
            return option
    raise ImageRequestError(
        f"A {label} értéke nem megengedett. Használható értékek: {', '.join(allowed)}.",
        400,
        "INVALID_IMAGE_OPTION",
    )


def _validated_count(value: Any, fallback: int, label: str) -> int:
    if value is None or (isinstance(value, str) and not value.strip()):
        return fallback
    if isinstance(value, bool):
        raise ImageRequestError(
            f"A {label} 1 és {MAX_IMAGES_PER_REQUEST} közötti egész szám lehet.",
            400,
            "INVALID_IMAGE_OPTION",
        )
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ImageRequestError(
            f"A {label} 1 és {MAX_IMAGES_PER_REQUEST} közötti egész szám lehet.",
            400,
            "INVALID_IMAGE_OPTION",
        ) from exc
    if not 1 <= number <= MAX_IMAGES_PER_REQUEST:
        raise ImageRequestError(
            f"A {label} 1 és {MAX_IMAGES_PER_REQUEST} közötti egész szám lehet.",
            400,
            "INVALID_IMAGE_OPTION",
        )
    return number


def _validated_flag(value: Any, fallback: bool, label: str) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = _text_value(value).lower()
    if text in TRUTHY_VALUES:
        return True
    if text in FALSY_VALUES:
        return False
    raise ImageRequestError(
        f"A {label} kapcsoló csak igaz vagy hamis érték lehet.",
        400,
        "INVALID_IMAGE_OPTION",
    )


def _validated_prompt(value: Any) -> str:
    prompt = _text_value(value)
    if not prompt:
        raise ImageRequestError("Add meg, milyen kép készüljön.", 400, "MISSING_PROMPT")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ImageRequestError(
            f"A képleírás legfeljebb {MAX_PROMPT_CHARS} karakter lehet.",
            413,
            "PROMPT_TOO_LARGE",
        )
    return prompt


@dataclass(frozen=True)
class ImageOptions:
    aspect_ratio: str
    resolution: str
    num_images: int
    thinking_level: str
    enable_web_search: bool
    safety_tolerance: str
    prefer_async: bool


def _options_from(values: dict[str, Any]) -> ImageOptions:
    legacy_ratio = LEGACY_SIZES.get(_text_value(values.get("size")).lower(), "")
    web_search = values.get("enable_web_search")
    if web_search is None:
        web_search = values.get("web_search")
    return ImageOptions(
        aspect_ratio=_validated_choice(
            values.get("aspect_ratio"),
            ASPECT_RATIOS,
            legacy_ratio or DEFAULT_ASPECT_RATIO,
            "képarány",
        ),
        resolution=DEFAULT_RESOLUTION,
        num_images=_validated_count(
            values.get("num_images"), DEFAULT_NUM_IMAGES, "képek száma"
        ),
        thinking_level=DEFAULT_THINKING_LEVEL,
        enable_web_search=_validated_flag(
            web_search, DEFAULT_ENABLE_WEB_SEARCH, "webes keresés"
        ),
        safety_tolerance=_validated_choice(
            values.get("safety_tolerance"),
            SAFETY_TOLERANCES,
            DEFAULT_SAFETY_TOLERANCE,
            "biztonsági tűréshatár",
        ),
        prefer_async=_validated_flag(values.get("async"), False, "aszinkron mód"),
    )


def _build_request(
    prompt: Any, references: tuple[ImageReference, ...], values: dict[str, Any]
) -> ImageRequest:
    options = _options_from(values)
    return ImageRequest(
        prompt=_validated_prompt(prompt),
        references=references,
        aspect_ratio=options.aspect_ratio,
        resolution=options.resolution,
        num_images=options.num_images,
        thinking_level=options.thinking_level,
        enable_web_search=options.enable_web_search,
        safety_tolerance=options.safety_tolerance,
        prefer_async=options.prefer_async,
    )


async def _parse_json_request(payload: Any) -> ImageRequest:
    if not isinstance(payload, dict):
        raise ImageRequestError("Érvénytelen képkérés.", 400, "INVALID_IMAGE_REQUEST")
    values: dict[str, Any] = {key: payload.get(key) for key in OPTION_KEYS}
    raw_references: list[Any] = []
    for key in REFERENCE_KEYS:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            raw_references.extend(value)
        else:
            raw_references.append(value)
    references = await _collect_references(raw_references)
    return _build_request(payload.get("prompt"), references, values)


async def _parse_form_request(request: Request) -> ImageRequest:
    try:
        form = await request.form()
    except Exception as exc:
        raise ImageRequestError(
            "A képfeltöltés nem olvasható.", 400, "INVALID_IMAGE_REQUEST"
        ) from exc
    if not isinstance(form, FormData):
        raise ImageRequestError(
            "A képfeltöltés nem olvasható.", 400, "INVALID_IMAGE_REQUEST"
        )
    try:
        values: dict[str, Any] = {}
        raw_references: list[Any] = []
        for key, value in form.multi_items():
            name = str(key)
            if name in REFERENCE_KEYS:
                raw_references.append(value)
                continue
            if name in OPTION_KEYS and name not in values:
                values[name] = value
        references = await _collect_references(raw_references)
        return _build_request(values.get("prompt"), references, values)
    finally:
        await form.close()


async def _parse_request(request: Request) -> ImageRequest:
    content_type = request.headers.get("content-type", "").lower()
    declared = request.headers.get("content-length")
    declared_size = int(declared) if declared is not None and declared.isdigit() else None
    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        if declared_size is not None and declared_size > MAX_FORM_BODY_BYTES:
            raise ImageRequestError("A képfeltöltés túl nagy.", 413, "REFERENCE_TOO_LARGE")
        return await _parse_form_request(request)
    if "application/json" in content_type:
        if declared_size is not None and declared_size > MAX_JSON_BODY_BYTES:
            raise ImageRequestError("A kérés túl nagy.", 413, "REQUEST_TOO_LARGE")
        raw = await request.body()
        if len(raw) > MAX_JSON_BODY_BYTES:
            raise ImageRequestError("A kérés túl nagy.", 413, "REQUEST_TOO_LARGE")
        if not raw:
            raise ImageRequestError("Hiányzó kéréstartalom.", 400, "INVALID_IMAGE_REQUEST")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ImageRequestError("Érvénytelen JSON.", 400, "INVALID_IMAGE_REQUEST") from exc
        return await _parse_json_request(payload)
    raise ImageRequestError(
        "A kérés tartalma JSON vagy multipart/form-data kell legyen.",
        415,
        "UNSUPPORTED_MEDIA_TYPE",
    )


async def _read_bounded(response: httpx.Response, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            break
    return b"".join(chunks)[:limit]


async def _download_bytes(
    client: httpx.AsyncClient, url: str, limit: int, timeout_seconds: float, purpose: str
) -> tuple[bytes, str]:
    await _assert_public_http_url(url)
    try:
        async with client.stream(
            "GET", url, timeout=_timeout(timeout_seconds), follow_redirects=True
        ) as response:
            if response.status_code >= 400:
                raise ImageRequestError(
                    f"A {purpose} letöltése nem sikerült (HTTP {response.status_code}).",
                    502,
                    "IMAGE_DOWNLOAD_FAILED",
                )
            declared = response.headers.get("content-type", "")
            data = await _read_bounded(response, limit + 1)
    except ImageRequestError:
        raise
    except httpx.HTTPError as exc:
        raise ImageRequestError(
            f"A {purpose} letöltése nem sikerült.", 502, "IMAGE_DOWNLOAD_FAILED"
        ) from exc
    except asyncio.CancelledError:
        raise
    if not data:
        raise ImageRequestError(f"A {purpose} üres.", 502, "IMAGE_DOWNLOAD_FAILED")
    if len(data) > limit:
        raise ImageRequestError(
            f"A {purpose} túl nagy a feldolgozáshoz.", 502, "IMAGE_DOWNLOAD_FAILED"
        )
    return data, declared


async def _materialize_references(
    client: httpx.AsyncClient, references: tuple[ImageReference, ...]
) -> tuple[ImageReference, ...]:
    if not references:
        return references
    if all(item.data is not None for item in references):
        return references
    if all(item.is_remote for item in references):
        for item in references:
            await _assert_public_http_url(str(item.url))
        return references
    materialized: list[ImageReference] = []
    for index, item in enumerate(references):
        if item.data is not None:
            materialized.append(item)
            continue
        data, declared = await _download_bytes(
            client,
            str(item.url),
            MAX_REFERENCE_BYTES,
            REFERENCE_DOWNLOAD_TIMEOUT_SECONDS,
            "referencia kép",
        )
        mime_type = _normalized_mime(declared, data)
        if mime_type not in ALLOWED_REFERENCE_TYPES:
            raise ImageRequestError(
                "A referencia kép formátuma nem támogatott (PNG, JPEG, WebP, HEIF, GIF).",
                415,
                "REFERENCE_UNSUPPORTED",
            )
        materialized.append(
            ImageReference(
                filename=_safe_filename(item.filename, mime_type, index),
                mime_type=mime_type,
                url=item.url,
                data=data,
            )
        )
    return tuple(materialized)


def _form_payload(
    image_request: ImageRequest,
) -> tuple[dict[str, str], list[tuple[str, tuple[str, bytes, str]]]]:
    data = {
        "prompt": image_request.prompt,
        "aspect_ratio": image_request.aspect_ratio,
        "resolution": image_request.resolution,
        "num_images": str(image_request.num_images),
        "thinking_level": image_request.thinking_level,
        "enable_web_search": "true" if image_request.enable_web_search else "false",
        "safety_tolerance": image_request.safety_tolerance,
    }
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    if image_request.references:
        if all(item.is_remote for item in image_request.references):
            data["image_urls"] = json.dumps(
                [str(item.url) for item in image_request.references]
            )
        else:
            for item in image_request.references:
                files.append(
                    ("image_urls", (item.filename, item.data or b"", item.mime_type))
                )
    return data, files


def _loads(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped or stripped == "[DONE]":
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _iter_sse(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    lines: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if lines:
                parsed = _loads("\n".join(lines))
                lines = []
                if parsed is not None:
                    yield parsed
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            lines.append(line[5:].lstrip())
    if lines:
        parsed = _loads("\n".join(lines))
        if parsed is not None:
            yield parsed


def _event_message(event: dict[str, Any]) -> str:
    for key in ("details", "message", "error", "detail", "reason"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return _clip(value)
        if isinstance(value, dict):
            nested = value.get("message") or value.get("detail") or value.get("error")
            if isinstance(nested, str) and nested.strip():
                return _clip(nested)
            if nested is not None:
                return _clip(json.dumps(nested, ensure_ascii=False))
            return _clip(json.dumps(value, ensure_ascii=False))
        if isinstance(value, list) and value:
            return _clip(json.dumps(value, ensure_ascii=False))
    return ""


def _status_message(status_code: int, body: str) -> str:
    detail = ""
    parsed = _loads(body)
    if parsed is not None:
        detail = _event_message(parsed)
    if not detail:
        detail = _clip(body)
    lowered = detail.lower()
    if status_code in (401, 403):
        if "insufficient" in lowered or "funds" in lowered or "balance" in lowered:
            return "A FlyMyAI egyenleg nem fedezi a képkészítést."
        if status_code == 401:
            return "A FlyMyAI kulcs nem fogadható el."
        return detail or "A FlyMyAI elutasította a kérést."
    if status_code in (404, 405, 501):
        return "A FlyMyAI képkészítő végpontja nem érhető el."
    if status_code == 413:
        return "A referencia képek túl nagyok a FlyMyAI számára."
    if status_code in (400, 415, 422):
        return detail or "A FlyMyAI elutasította a megadott paramétereket."
    if status_code == 429:
        return "A FlyMyAI jelenleg nem fogad több képkérést; próbáld újra később."
    if 500 <= status_code < 600:
        return detail or "A FlyMyAI szolgáltatás hibát jelzett; próbáld újra később."
    return detail or f"A FlyMyAI hibát jelzett (HTTP {status_code})."


def _output_images(output: dict[str, Any]) -> list[Any]:
    for key in ("images", "image", "output_images", "samples", "sample", "data", "result"):
        value = output.get(key)
        if isinstance(value, list) and value:
            return value
        if isinstance(value, str) and value.strip():
            return [value]
    return []


def _image_url(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        for key in ("url", "image_url", "output_url", "uri", "href", "src"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in ("b64_json", "base64", "data"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                if value.startswith("data:"):
                    return value
                return f"data:image/png;base64,{value}"
    return ""


def _token_usage(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        return _loads(value)
    return None


def _price(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}".rstrip("0").rstrip(".") or "0"
    text = _text_value(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return _clip(text, 40)
    return f"{number:.6f}".rstrip("0").rstrip(".") or "0"


async def _predict_sync(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    data: dict[str, str],
    files: list[tuple[str, tuple[str, bytes, str]]],
    request: Request,
) -> dict[str, Any]:
    output: dict[str, Any] | None = None
    async with client.stream(
        "POST", _predict_url(), data=data, files=files, headers=headers
    ) as response:
        if response.status_code >= 400:
            body = await _read_bounded(response, MAX_ERROR_BYTES)
            text = body.decode("utf-8", errors="replace")
            raise ImageRequestError(
                _status_message(response.status_code, text), 502, "IMAGE_UPSTREAM_FAILED"
            )
        content_type = response.headers.get("content-type", "").lower()
        if "text/event-stream" in content_type:
            async for event in _iter_sse(response):
                if await request.is_disconnected():
                    raise ImageRequestError(
                        "A képkérés megszakadt.", 499, "IMAGE_REQUEST_CANCELLED"
                    )
                status = event.get("status")
                if event.get("details") is not None or (
                    isinstance(status, int) and status >= 400
                ):
                    code = status if isinstance(status, int) else 502
                    message = _event_message(event) or _status_message(code, "")
                    raise ImageRequestError(message, 502, "IMAGE_UPSTREAM_FAILED")
                candidate = event.get("output_data")
                if isinstance(candidate, dict) and _output_images(candidate):
                    output = candidate
        else:
            body = await _read_bounded(response, MAX_RESPONSE_BYTES)
            parsed = _loads(body.decode("utf-8", errors="replace"))
            if parsed is None:
                raise ImageRequestError(
                    "A FlyMyAI válasza nem értelmezhető.", 502, "IMAGE_UPSTREAM_FAILED"
                )
            candidate = parsed.get("output_data")
            output = candidate if isinstance(candidate, dict) else parsed
    if output is None or not _output_images(output):
        raise ImageRequestError(
            "A FlyMyAI nem adott vissza képet.", 502, "IMAGE_UPSTREAM_FAILED"
        )
    return output


def _prediction_id(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("prediction_id", "infer_id", "request_id", "task_id", "id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
    details = payload.get("infer_details")
    if isinstance(details, dict):
        return _prediction_id(details)
    return ""


def _async_output(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    for key in ("output_data", "response", "result", "output"):
        value = payload.get(key)
        if isinstance(value, dict) and _output_images(value):
            return value
    entries = payload.get("inference_responses")
    if isinstance(entries, list):
        for entry in entries:
            candidate = _async_output(entry)
            if candidate is not None:
                return candidate
    if _output_images(payload):
        return payload
    return None


def _async_failure(payload: dict[str, Any]) -> tuple[int, str]:
    details = payload.get("infer_details")
    status = 0
    message = ""
    if isinstance(details, dict):
        raw_status = details.get("status")
        if isinstance(raw_status, int):
            status = raw_status
        message = _event_message(details)
    if not message:
        message = _event_message(payload)
    return status, message


async def _cancel_prediction(
    client: httpx.AsyncClient, headers: dict[str, str], prediction_id: str
) -> None:
    if not prediction_id:
        return
    try:
        await client.patch(
            _cancel_url(),
            json={"infer_id": prediction_id},
            headers=headers,
            timeout=_timeout(CANCEL_TIMEOUT_SECONDS),
        )
    except httpx.HTTPError as exc:
        logger.warning("images: a képkészítés leállítása nem sikerült: %r", exc)


async def _predict_async(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    data: dict[str, str],
    files: list[tuple[str, tuple[str, bytes, str]]],
    request: Request,
) -> dict[str, Any]:
    deadline = time.monotonic() + PREDICT_DEADLINE_SECONDS
    response = await client.post(
        _async_submit_url(),
        data=data,
        files=files,
        headers=headers,
        timeout=_timeout(CALL_TIMEOUT_SECONDS),
    )
    if response.status_code in (404, 405, 501):
        raise _UseSyncRoute(response.text[:MAX_ERROR_BYTES])
    if response.status_code >= 400:
        text = response.text[:MAX_ERROR_BYTES]
        raise ImageRequestError(
            _status_message(response.status_code, text), 502, "IMAGE_UPSTREAM_FAILED"
        )
    payload = _loads(response.text)
    if payload is None:
        raise ImageRequestError(
            "A FlyMyAI válasza nem értelmezhető.", 502, "IMAGE_UPSTREAM_FAILED"
        )
    prediction_id = _prediction_id(payload)
    if not prediction_id:
        output = _async_output(payload)
        if output is not None:
            return output
        raise ImageRequestError(
            "A FlyMyAI nem adott azonosítót a képkészítéshez.", 502, "IMAGE_UPSTREAM_FAILED"
        )
    try:
        while True:
            if await request.is_disconnected():
                raise ImageRequestError(
                    "A képkérés megszakadt.", 499, "IMAGE_REQUEST_CANCELLED"
                )
            if time.monotonic() >= deadline:
                raise ImageRequestError(
                    "A képkészítés időtúllépés miatt megszakadt.", 504, "IMAGE_UPSTREAM_TIMEOUT"
                )
            await asyncio.sleep(ASYNC_POLL_SECONDS)
            poll = await client.get(
                _async_result_url(),
                params={"request_id": prediction_id},
                headers=headers,
                timeout=_timeout(CALL_TIMEOUT_SECONDS),
            )
            if poll.status_code == 425:
                continue
            if poll.status_code >= 400:
                raise ImageRequestError(
                    _status_message(poll.status_code, poll.text[:MAX_ERROR_BYTES]),
                    502,
                    "IMAGE_UPSTREAM_FAILED",
                )
            body = _loads(poll.text)
            if body is None:
                continue
            status, message = _async_failure(body)
            if status >= 400:
                raise ImageRequestError(
                    message or _status_message(status, ""), 502, "IMAGE_UPSTREAM_FAILED"
                )
            output = _async_output(body)
            if output is not None:
                return output
    except BaseException:
        await _cancel_prediction(client, headers, prediction_id)
        raise


def _build_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=_timeout(PREDICT_DEADLINE_SECONDS),
        limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        follow_redirects=False,
        headers={"Accept": "text/event-stream, application/json"},
    )


async def _inline_remote_images(
    client: httpx.AsyncClient, entries: list[Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    items: list[dict[str, Any]] = []
    warnings: list[str] = []
    inlined_total = 0
    for index, entry in enumerate(entries[:MAX_OUTPUT_IMAGES]):
        source = _image_url(entry)
        item: dict[str, Any] = {
            "url": source,
            "remote_url": source if source.startswith("http") else "",
            "mime_type": "image/png",
            "size": 0,
            "inline": False,
            "width": None,
            "height": None,
        }
        if isinstance(entry, dict):
            for key in ("width", "height"):
                value = entry.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    item[key] = int(value)
            declared = entry.get("content_type") or entry.get("mime_type")
            if isinstance(declared, str) and declared.strip():
                normalized = _normalized_mime(declared, None)
                if normalized in ALLOWED_REFERENCE_TYPES:
                    item["mime_type"] = normalized
        if source.startswith("data:"):
            match = DATA_URL_PATTERN.match(source)
            if match is not None:
                try:
                    decoded = base64.b64decode(match.group(2), validate=True)
                except (binascii.Error, ValueError):
                    decoded = b""
                item["mime_type"] = _normalized_mime(match.group(1), decoded) or item["mime_type"]
                item["size"] = len(decoded)
                item["inline"] = True
            items.append(item)
            continue
        if not source.startswith("http"):
            warnings.append(f"A {index + 1}. eredmény nem jeleníthető meg.")
            continue
        if inlined_total >= INLINE_TOTAL_MAX_BYTES:
            warnings.append(
                f"A {index + 1}. kép nem került beágyazásra; a link {REMOTE_LINK_HOURS} óráig él."
            )
            items.append(item)
            continue
        try:
            data, declared = await _download_bytes(
                client,
                source,
                DOWNLOAD_IMAGE_MAX_BYTES,
                REFERENCE_DOWNLOAD_TIMEOUT_SECONDS,
                "elkészült kép",
            )
        except ImageRequestError as exc:
            logger.warning("images: a kész kép letöltése nem sikerült: %s", exc.message)
            warnings.append(
                f"A {index + 1}. kép letöltése nem sikerült; a link {REMOTE_LINK_HOURS} óráig él."
            )
            items.append(item)
            continue
        except asyncio.CancelledError:
            raise
        mime_type = _normalized_mime(declared, data)
        if mime_type in ALLOWED_REFERENCE_TYPES:
            item["mime_type"] = mime_type
        item["size"] = len(data)
        if len(data) > INLINE_IMAGE_MAX_BYTES or inlined_total + len(data) > INLINE_TOTAL_MAX_BYTES:
            warnings.append(
                f"A {index + 1}. kép túl nagy a beágyazáshoz; a link {REMOTE_LINK_HOURS} óráig él."
            )
            items.append(item)
            continue
        inlined_total += len(data)
        item["url"] = f"data:{item['mime_type']};base64,{base64.b64encode(data).decode('ascii')}"
        item["inline"] = True
        items.append(item)
    return items, warnings


async def _run_image_request(request: Request) -> dict[str, Any]:
    api_key = _api_key()
    image_request = await _parse_request(request)
    headers = {"X-API-KEY": api_key}
    route = "queue" if image_request.needs_queue else "predict"
    async with _build_client() as client:
        references = await _materialize_references(client, image_request.references)
        image_request = image_request.with_references(references)
        data, files = _form_payload(image_request)
        try:
            if image_request.needs_queue:
                try:
                    output = await asyncio.wait_for(
                        _predict_async(client, headers, data, files, request),
                        timeout=PREDICT_DEADLINE_SECONDS + CALL_TIMEOUT_SECONDS,
                    )
                except _UseSyncRoute as fallback:
                    logger.warning(
                        "images: a FlyMyAI sor nem elérhető, szinkron hívás: %s",
                        _clip(fallback.detail, 200),
                    )
                    route = "predict"
                    output = await asyncio.wait_for(
                        _predict_sync(client, headers, data, files, request),
                        timeout=PREDICT_DEADLINE_SECONDS,
                    )
            else:
                output = await asyncio.wait_for(
                    _predict_sync(client, headers, data, files, request),
                    timeout=PREDICT_DEADLINE_SECONDS,
                )
        except asyncio.TimeoutError as exc:
            raise ImageRequestError(
                "A képkészítés időtúllépés miatt megszakadt.", 504, "IMAGE_UPSTREAM_TIMEOUT"
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning("images: a FlyMyAI nem érhető el: %r", exc)
            raise ImageRequestError(
                "A FlyMyAI nem érhető el; próbáld újra később.", 502, "IMAGE_UPSTREAM_FAILED"
            ) from exc
        entries = _output_images(output)
        if not entries:
            raise ImageRequestError(
                "A FlyMyAI nem adott vissza képet.", 502, "IMAGE_UPSTREAM_FAILED"
            )
        images, warnings = await _inline_remote_images(client, entries)
    usable = [item for item in images if item.get("url")]
    if not usable:
        raise ImageRequestError(
            "A FlyMyAI nem adott vissza megjeleníthető képet.", 502, "IMAGE_UPSTREAM_FAILED"
        )
    payload: dict[str, Any] = {
        "status": "completed",
        "model": _model_name(),
        "route": route,
        "images": usable,
        "description": _clip(output.get("description"), MAX_DESCRIPTION_CHARS),
        "token_usage": _token_usage(output.get("token_usage")),
        "calculated_price": _price(output.get("calculated_price")),
        "request": {
            "mode": image_request.mode,
            "prompt_chars": len(image_request.prompt),
            "aspect_ratio": image_request.aspect_ratio,
            "resolution": image_request.resolution,
            "num_images": image_request.num_images,
            "thinking_level": image_request.thinking_level,
            "enable_web_search": image_request.enable_web_search,
            "safety_tolerance": image_request.safety_tolerance,
            "reference_count": len(image_request.references),
        },
        "warnings": warnings,
    }
    if any(item.get("remote_url") for item in usable):
        expires = datetime.now(timezone.utc) + timedelta(hours=REMOTE_LINK_HOURS)
        payload["remote_link_expires_at"] = expires.isoformat().replace("+00:00", "Z")
    return payload


def _options_payload() -> dict[str, Any]:
    return {
        "model": _model_name(),
        "aspect_ratios": list(ASPECT_RATIOS),
        "resolutions": [DEFAULT_RESOLUTION],
        "thinking_levels": [DEFAULT_THINKING_LEVEL],
        "safety_tolerances": list(SAFETY_TOLERANCES),
        "reference_types": list(REFERENCE_MIME_TYPES),
        "defaults": {
            "aspect_ratio": DEFAULT_ASPECT_RATIO,
            "resolution": DEFAULT_RESOLUTION,
            "num_images": DEFAULT_NUM_IMAGES,
            "thinking_level": DEFAULT_THINKING_LEVEL,
            "enable_web_search": DEFAULT_ENABLE_WEB_SEARCH,
            "safety_tolerance": DEFAULT_SAFETY_TOLERANCE,
        },
        "limits": {
            "max_reference_images": MAX_REFERENCE_IMAGES,
            "max_images": MAX_IMAGES_PER_REQUEST,
            "max_reference_bytes": MAX_REFERENCE_BYTES,
            "max_prompt_chars": MAX_PROMPT_CHARS,
            "async_num_images_threshold": ASYNC_NUM_IMAGES_THRESHOLD,
        },
        "remote_link_hours": REMOTE_LINK_HOURS,
        "endpoints": ["/api/images/generate", "/api/images/edits"],
    }


@router.get("/api/images/options")
async def get_image_options(request: Request) -> Any:
    try:
        rate_limiter.check("list", client_key(request))
        return _json(_options_payload())
    except RateLimitFault as exc:
        return _rate_limit_failure(exc)
    except ImageRequestError as exc:
        return _failure(exc.message, exc.status_code, exc.code)
    except Exception:
        logger.exception("images: a beállítások lekérése nem sikerült")
        return _failure("Belső szerverhiba.", 500, "INTERNAL_ERROR")


@router.post("/api/images/generate")
async def post_image_generate(request: Request) -> Any:
    return await _handle_image_request(request)


@router.post("/api/images/edits")
async def post_image_edits(request: Request) -> Any:
    return await _handle_image_request(request)


async def _handle_image_request(request: Request) -> Any:
    try:
        origin_error = verify_origin(request)
        if origin_error is not None:
            return _failure(origin_error, 403, "INVALID_ORIGIN")
        rate_limiter.check("images", client_key(request))
        payload = await _run_image_request(request)
        return _json(payload)
    except RateLimitFault as exc:
        return _rate_limit_failure(exc)
    except ImageRequestError as exc:
        if exc.code == "IMAGE_REQUEST_CANCELLED":
            logger.info("images: a képkérés megszakadt: %s", exc.message)
        else:
            logger.warning("images: %s (%s)", exc.message, exc.code)
        return _failure(exc.message, exc.status_code, exc.code)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("images: a képkérés váratlanul elakadt")
        return _failure("Belső szerverhiba.", 500, "INTERNAL_ERROR")
