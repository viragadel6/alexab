from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

INT_TEXT_PATTERN = re.compile(r"[+-]?[0-9]+")
DIGIT_CHUNK_SIZE = 500
LOG10_OF_TWO = math.log10(2)


def now() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware(value: datetime) -> datetime:
    try:
        offset = value.utcoffset()
    except (TypeError, ValueError, OverflowError):
        offset = None
    if offset is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _utc_offset(value: datetime) -> timedelta:
    try:
        offset = value.utcoffset()
    except (TypeError, ValueError, OverflowError):
        offset = None
    if offset is None:
        return timedelta(0)
    return offset


def _decimal_digit_count(magnitude: int) -> int:
    if magnitude <= 0:
        return 1
    count = max(1, int(magnitude.bit_length() * LOG10_OF_TWO))
    while 10**count <= magnitude:
        count += 1
    while count > 1 and 10 ** (count - 1) > magnitude:
        count -= 1
    return count


def _magnitude_digits(magnitude: int) -> str:
    if magnitude <= 0:
        return "0"
    chunk = 10**DIGIT_CHUNK_SIZE
    parts: list[int] = []
    remaining = magnitude
    while remaining:
        remaining, remainder = divmod(remaining, chunk)
        parts.append(remainder)
    leading = str(parts[-1])
    trailing = "".join(str(part).zfill(DIGIT_CHUNK_SIZE) for part in reversed(parts[:-1]))
    return leading + trailing


def _int_to_text(value: int, limit: int = 0) -> str:
    try:
        text = str(value)
    except ValueError:
        text = None
    if text is not None:
        if limit > 0 and len(text) > limit:
            return text[:limit]
        return text
    sign = "-" if value < 0 else ""
    magnitude = -value if value < 0 else value
    if limit > 0:
        available = limit - len(sign)
        if available <= 0:
            return sign[:limit]
        count = _decimal_digit_count(magnitude)
        if count > available:
            magnitude //= 10 ** (count - available)
    return sign + _magnitude_digits(magnitude)


def _key_text(key: Any) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, int) and not isinstance(key, bool):
        return _int_to_text(key)
    try:
        return str(key)
    except Exception:
        try:
            return repr(key)
        except Exception:
            return f"<{type(key).__name__}>"


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return iso(value)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "backslashreplace")
    if isinstance(value, memoryview):
        return value.tobytes().decode("utf-8", "backslashreplace")
    if isinstance(value, (set, frozenset)):
        return list(value)
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        result = isoformat()
        if isinstance(result, str):
            return result
    raise TypeError(f"A(z) {type(value).__name__} típus nem szerializálható JSON formátumba.")


def iso(value: datetime | None = None) -> str:
    moment = value if value is not None else now()
    if not isinstance(moment, datetime):
        raise TypeError("Az iso() csak datetime értéket fogad el.")
    return _as_aware(moment).isoformat()


def parse_iso(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _as_aware(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        base = text[:-1]
        if not base or "Z" in base or "z" in base:
            return None
        text = f"{base}+00:00"
    elif "Z" in text or "z" in text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except (ValueError, TypeError, OverflowError):
        return None
    return _as_aware(parsed)


def age_ms(value: Any, reference: datetime | None = None) -> float | None:
    parsed = parse_iso(value)
    if parsed is None:
        return None
    if reference is None:
        base = now()
    elif isinstance(reference, datetime):
        base = _as_aware(reference)
    else:
        raise TypeError("Az age_ms() referencia értéke csak datetime lehet.")
    wall_delta = base.replace(tzinfo=None) - parsed.replace(tzinfo=None)
    offset_delta = _utc_offset(base) - _utc_offset(parsed)
    return (wall_delta - offset_delta).total_seconds() * 1000


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


def as_str(value: Any, limit: int = 0) -> str:
    if limit < 0:
        raise ValueError("Az as_str() limit értéke nem lehet negatív.")
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, int):
        return _int_to_text(value, limit)
    elif isinstance(value, float):
        text = str(value)
    else:
        return ""
    if limit > 0 and len(text) > limit:
        return text[:limit]
    return text


def as_int(
    value: Any,
    default: int = 0,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("Az as_int() minimum értéke nem lehet nagyobb a maximum értékénél.")
    converted = False
    number = default
    if isinstance(value, bool):
        number = int(value)
        converted = True
    elif isinstance(value, int):
        number = value
        converted = True
    elif isinstance(value, float):
        if math.isfinite(value):
            number = int(value)
            converted = True
    elif isinstance(value, str):
        stripped = value.strip()
        if INT_TEXT_PATTERN.fullmatch(stripped):
            try:
                number = int(stripped)
                converted = True
            except (ValueError, TypeError):
                number = default
    if not converted:
        return default
    if maximum is not None and number > maximum:
        number = maximum
    if minimum is not None and number < minimum:
        number = minimum
    return number


def as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int):
        try:
            converted = float(value)
        except (OverflowError, ValueError):
            return default
        return converted if math.isfinite(converted) else default
    if isinstance(value, float):
        return value if math.isfinite(value) else default
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            converted = float(stripped)
        except (ValueError, OverflowError):
            return default
        return converted if math.isfinite(converted) else default
    return default


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, float):
        if not math.isfinite(value):
            return default
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    return default


def as_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        result[_key_text(key)] = item
    return result


def as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def utf8_bytes(value: Any) -> int:
    if isinstance(value, str):
        try:
            return len(value.encode("utf-8", "surrogatepass"))
        except (UnicodeEncodeError, LookupError):
            return 0
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, memoryview):
        return value.nbytes
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=_json_default)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return 0
    try:
        return len(encoded.encode("utf-8", "surrogatepass"))
    except (UnicodeEncodeError, LookupError):
        return 0