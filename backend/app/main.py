from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
from collections.abc import AsyncIterator
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime
from importlib import metadata
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi import HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.routing import BaseRoute, Match, Route, WebSocketRoute
from starlette.types import Receive, Scope, Send

from app.config import get_settings
from app.database import close_databases
from app.middleware.error_handler import register_error_handlers
from app.middleware.security import RequestContextMiddleware
from app.routers import browser as browser_router
from app.routers import chat as chat_router
from app.routers import chat_jobs as chat_jobs_router
from app.routers import conversations as conversations_router
from app.routers import images as images_router
from app.routers import operator as operator_router
from app.routers import public_worker as public_worker_router
from app.routers import system as system_router
from app.services.supervisor import start_supervisor, stop_supervisor, supervisor_status

logger = logging.getLogger("app")

_PUBLIC_STATIC_EXTENSIONS = frozenset(
    {
        ".aac",
        ".avif",
        ".bmp",
        ".css",
        ".eot",
        ".flac",
        ".gif",
        ".htm",
        ".html",
        ".ico",
        ".jpeg",
        ".jpg",
        ".js",
        ".m4a",
        ".mjs",
        ".mp3",
        ".mp4",
        ".ogg",
        ".ogv",
        ".opus",
        ".otf",
        ".png",
        ".svg",
        ".ttf",
        ".wasm",
        ".wav",
        ".webm",
        ".webmanifest",
        ".webp",
        ".woff",
        ".woff2",
    }
)

_PUBLIC_STATIC_FILENAMES = frozenset(
    {
        "asset-manifest.json",
        "manifest.json",
        "robots.txt",
    }
)


@dataclass(frozen=True)
class _ShutdownPolicy:
    grace_seconds: float
    supervisor_grace_seconds: float
    server_timeout_seconds: float | None


@dataclass(frozen=True)
class _TaskOutcome:
    error: BaseException | None = None
    cancellation: asyncio.CancelledError | None = None


def configure_logging(settings: Any | None = None) -> None:
    settings = get_settings() if settings is None else settings
    configured_level = getattr(settings, "log_level", None)

    if configured_level is None:
        configured_level = "INFO"

    if isinstance(configured_level, bool):
        raise ValueError("log_level must be a logging level name or a nonnegative integer.")

    if isinstance(configured_level, int):
        if configured_level < 0:
            raise ValueError("log_level must not be negative.")
        level = configured_level
    elif isinstance(configured_level, str):
        level_name = configured_level.strip().upper()
        level = logging.getLevelName(level_name)
        if not isinstance(level, int):
            raise ValueError(f"Unsupported log_level: {level_name!r}.")
    else:
        raise ValueError("log_level must be a logging level name or a nonnegative integer.")

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logger.setLevel(level)
    logger.disabled = False
    logger.propagate = True
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _string_values(value: Any, setting_name: str) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = list(value)
    elif isinstance(value, (set, frozenset)):
        values = list(value)
    else:
        raise ValueError(
            f"{setting_name} must be a comma-separated string or a collection of strings."
        )

    result: list[str] = []
    seen: set[str] = set()

    for item in values:
        if not isinstance(item, str):
            raise ValueError(f"Every value in {setting_name} must be a string.")
        normalized = item.strip()
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)

    return result


def _boolean_setting(value: Any, setting_name: str) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, int) and value in (0, 1):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False

    raise ValueError(f"{setting_name} must be a boolean.")


def _normalize_origin(origin: str) -> str:
    if origin == "*":
        return origin

    if "*" in origin or any(character.isspace() for character in origin):
        raise ValueError("CORS origins must be explicit origins without wildcard patterns.")

    try:
        parsed = urlsplit(origin)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("A configured CORS origin is not a valid URL origin.") from exc

    scheme = parsed.scheme.lower()

    if (
        scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "CORS origins must contain only an HTTP or HTTPS scheme, hostname, and optional port."
        )

    try:
        hostname = hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ValueError("A configured CORS origin contains an invalid hostname.") from exc

    if ":" in hostname:
        hostname = f"[{hostname}]"

    default_port = 80 if scheme == "http" else 443
    port_suffix = "" if port is None or port == default_port else f":{port}"

    return f"{scheme}://{hostname}{port_suffix}"


def _cors_configuration(settings: Any) -> tuple[list[str], bool]:
    configured = _string_values(
        getattr(settings, "app_origins", None),
        "app_origins",
    )

    if not configured:
        configured = _string_values(
            getattr(settings, "app_origin", None),
            "app_origin",
        )

    origins = list(dict.fromkeys(_normalize_origin(origin) for origin in configured))

    credentials_value = getattr(settings, "cors_allow_credentials", None)
    allow_credentials = (
        True
        if credentials_value is None
        else _boolean_setting(credentials_value, "cors_allow_credentials")
    )

    if "*" in origins and allow_credentials:
        raise ValueError(
            "Wildcard CORS origins cannot be combined with credentialed requests."
        )

    if "*" in origins:
        origins = ["*"]

    return origins, allow_credentials


def _project_directories() -> tuple[Path, Path]:
    module_directory = Path(__file__).resolve().parent
    application_directory = module_directory.parent
    repository_directory = application_directory.parent
    return application_directory, repository_directory


def _source_application_version() -> str | None:
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            return None

    for directory in _project_directories():
        project_file = directory / "pyproject.toml"

        try:
            if not project_file.is_file():
                continue
            with project_file.open("rb") as stream:
                project_data = tomllib.load(stream)
        except (OSError, ValueError) as exc:
            logger.warning(
                "application_version_metadata_unreadable path=%s error=%s",
                project_file,
                exc,
                exc_info=True,
            )
            continue

        project = project_data.get("project")
        tool = project_data.get("tool")
        poetry = tool.get("poetry") if isinstance(tool, Mapping) else None

        for section in (project, poetry):
            if not isinstance(section, Mapping):
                continue

            version = section.get("version")
            if isinstance(version, str) and version.strip():
                return version.strip()

            distribution_name = section.get("name")
            if isinstance(distribution_name, str) and distribution_name.strip():
                try:
                    installed_version = metadata.version(distribution_name.strip())
                except metadata.PackageNotFoundError:
                    installed_version = None

                if installed_version:
                    return installed_version

    return None


def _application_version(settings: Any, override: str | None = None) -> str:
    if override is not None:
        if not isinstance(override, str) or not override.strip():
            raise ValueError("The application version must be a nonempty string.")
        return override.strip()

    for setting_name in ("app_version", "api_version", "version"):
        configured = getattr(settings, setting_name, None)
        if configured is None:
            continue
        if not isinstance(configured, str):
            raise ValueError(f"{setting_name} must be a string.")
        if configured.strip():
            return configured.strip()

    distribution_name = getattr(settings, "distribution_name", None)

    if distribution_name is None:
        distribution_name = "chat-pal-backend"

    if not isinstance(distribution_name, str) or not distribution_name.strip():
        raise ValueError("distribution_name must be a nonempty string.")

    try:
        installed_version = metadata.version(distribution_name.strip())
    except metadata.PackageNotFoundError:
        installed_version = None

    if installed_version:
        return installed_version

    package = sys.modules.get(__package__ or "app")
    package_version = getattr(package, "__version__", None)

    if isinstance(package_version, str) and package_version.strip():
        return package_version.strip()

    source_version = _source_application_version()
    if source_version is not None:
        return source_version

    raise RuntimeError(
        "The application version is unavailable. Configure app_version or api_version, "
        "pass version to create_app(), or install the application's distribution metadata."
    )


def resolve_static_directory(settings: Any | None = None) -> Path | None:
    settings = get_settings() if settings is None else settings
    application_directory, repository_directory = _project_directories()
    configured = getattr(settings, "static_dir", None)
    candidates: list[Path] = []

    if configured is not None:
        try:
            if isinstance(configured, str):
                configured = configured.strip()

            if configured != "":
                if not isinstance(configured, (str, os.PathLike)):
                    raise TypeError("static_dir must be a string or a filesystem path.")

                configured_path = Path(configured)
                if not configured_path.is_absolute():
                    configured_path = application_directory / configured_path
                candidates.append(configured_path)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            logger.warning(
                "configured_static_directory_invalid error=%s",
                exc,
                exc_info=True,
            )

    candidates.extend(
        [
            repository_directory / "public",
            application_directory / "public",
        ]
    )

    seen: set[Path] = set()

    for candidate in candidates:
        try:
            resolved = candidate.resolve()

            if resolved in seen:
                continue
            seen.add(resolved)

            if not resolved.is_dir():
                continue

            entrypoint = resolved / "chat.html"
            if not entrypoint.is_file():
                continue

            resolved_entrypoint = entrypoint.resolve(strict=True)
            resolved_entrypoint.relative_to(resolved)

            logger.info("static_directory_selected path=%s", resolved)
            return resolved
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            logger.warning(
                "static_directory_candidate_rejected path=%s error=%s",
                candidate,
                exc,
                exc_info=True,
            )

    logger.warning(
        "static_directory_not_found frontend_enabled=false candidates=%s",
        [str(candidate) for candidate in candidates],
    )
    return None


def _public_static_path(value: str) -> str:
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or ":" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("Static filenames must be safe relative URL paths.")

    parts = value.split("/")

    if any(
        not part
        or part in {".", ".."}
        or part.startswith(".")
        or part == "__pycache__"
        for part in parts
    ):
        raise ValueError("Static filenames cannot contain hidden or traversal components.")

    if parts[0].casefold() == "api":
        raise ValueError("The API namespace cannot be used for static files.")

    return "/".join(parts)


def _static_allowlist(settings: Any, directory: Path) -> frozenset[str]:
    configured = getattr(settings, "static_allowed_files", None)
    allowed: set[str] = set()

    if configured is None:
        try:
            for candidate in directory.rglob("*"):
                if candidate.is_symlink() or not candidate.is_file():
                    continue

                relative = candidate.relative_to(directory).as_posix()

                try:
                    relative = _public_static_path(relative)
                except ValueError:
                    continue

                if (
                    candidate.suffix.lower() not in _PUBLIC_STATIC_EXTENSIONS
                    and relative not in _PUBLIC_STATIC_FILENAMES
                ):
                    continue

                resolved = candidate.resolve(strict=True)

                try:
                    resolved.relative_to(directory)
                except ValueError:
                    continue

                allowed.add(relative)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError("The public static asset allowlist could not be built.") from exc

        logger.info(
            "static_allowlist_generated allowed_file_count=%s",
            len(allowed),
        )
    else:
        configured_files = _string_values(configured, "static_allowed_files")

        for filename in configured_files:
            relative = _public_static_path(filename)
            candidate = directory.joinpath(*relative.split("/"))

            try:
                if candidate.is_symlink():
                    raise ValueError("Symbolic links cannot be explicitly exposed as static files.")

                resolved = candidate.resolve(strict=True)
                resolved.relative_to(directory)

                if not resolved.is_file():
                    raise ValueError("A configured static asset is not a regular file.")
            except (OSError, ValueError, RuntimeError) as exc:
                raise ValueError(
                    f"Invalid static_allowed_files entry: {relative!r}."
                ) from exc

            allowed.add(relative)

    if "chat.html" not in allowed:
        raise ValueError("The static asset allowlist must include a regular chat.html file.")

    return frozenset(allowed)


class _PublicStaticFiles(StaticFiles):
    def __init__(self, directory: Path, allowed_files: frozenset[str]) -> None:
        super().__init__(
            directory=str(directory),
            html=False,
            check_dir=True,
            follow_symlink=False,
        )
        self.allowed_files = allowed_files

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return

        await super().__call__(scope, receive, send)

    async def get_response(self, path: str, scope: Scope) -> Response:
        normalized = path.replace(os.sep, "/")

        try:
            normalized = _public_static_path(normalized)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Not Found") from exc

        if normalized not in self.allowed_files:
            raise HTTPException(status_code=404, detail="Not Found")

        response = await super().get_response(normalized, scope)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")

        if Path(normalized).suffix.lower() in {".html", ".htm"}:
            response.headers.setdefault("Cache-Control", "no-cache")

        return response


def _finite_seconds(
    value: Any,
    setting_name: str,
    *,
    allow_zero: bool = False,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{setting_name} must be a finite number of seconds.")

    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{setting_name} must be a finite number of seconds."
        ) from exc

    if not math.isfinite(seconds):
        raise ValueError(f"{setting_name} must be finite.")

    if seconds < 0 or (seconds == 0 and not allow_zero):
        requirement = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{setting_name} must be {requirement}.")

    return seconds


def _shutdown_policy(
    settings: Any,
    grace_override: float | None = None,
    server_timeout_override: float | None = None,
) -> _ShutdownPolicy:
    grace_value = (
        grace_override
        if grace_override is not None
        else getattr(settings, "shutdown_grace_seconds", None)
    )
    if grace_value is None:
        grace_value = 15.0

    requested_grace = _finite_seconds(grace_value, "shutdown_grace_seconds")
    grace_seconds = requested_grace

    server_value = server_timeout_override
    if server_value is None:
        server_value = getattr(settings, "server_shutdown_timeout_seconds", None)
    if server_value is None:
        server_value = getattr(settings, "asgi_shutdown_timeout_seconds", None)

    server_timeout: float | None = None

    if server_value is not None:
        server_timeout = _finite_seconds(
            server_value,
            "server_shutdown_timeout_seconds",
        )
        margin_value = getattr(settings, "shutdown_timeout_margin_seconds", None)
        if margin_value is None:
            margin_value = min(1.0, server_timeout / 10.0)

        margin = _finite_seconds(
            margin_value,
            "shutdown_timeout_margin_seconds",
            allow_zero=True,
        )

        if margin >= server_timeout:
            raise ValueError(
                "shutdown_timeout_margin_seconds must be smaller than the server shutdown timeout."
            )

        grace_seconds = min(requested_grace, server_timeout - margin)

        if grace_seconds < requested_grace:
            logger.info(
                "shutdown_grace_capped requested_seconds=%s effective_seconds=%s",
                requested_grace,
                grace_seconds,
            )
    else:
        logger.warning(
            "server_shutdown_timeout_unspecified "
            "configure_server_shutdown_timeout_seconds_to_match_the_ASGI_server=true"
        )

    database_value = getattr(settings, "database_shutdown_grace_seconds", None)
    if database_value is None:
        database_value = min(5.0, grace_seconds / 3.0)

    database_grace = _finite_seconds(
        database_value,
        "database_shutdown_grace_seconds",
        allow_zero=True,
    )

    if database_grace >= grace_seconds:
        raise ValueError(
            "database_shutdown_grace_seconds must be smaller than the effective shutdown grace."
        )

    return _ShutdownPolicy(
        grace_seconds=grace_seconds,
        supervisor_grace_seconds=grace_seconds - database_grace,
        server_timeout_seconds=server_timeout,
    )


async def _drain_task(task: asyncio.Task[Any]) -> _TaskOutcome:
    cancellation: asyncio.CancelledError | None = None

    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                break

            if cancellation is None:
                cancellation = exc
                logger.warning(
                    "shutdown_cancellation_deferred task=%s",
                    task.get_name(),
                )
        except BaseException:
            if not task.done():
                raise
            break

    try:
        task.result()
    except BaseException as exc:
        return _TaskOutcome(error=exc, cancellation=cancellation)

    return _TaskOutcome(cancellation=cancellation)


async def _run_cleanup_phase(
    operation: Callable[[], Awaitable[Any]],
    operation_name: str,
    grace_seconds: float,
) -> _TaskOutcome:
    async def invoke() -> None:
        await operation()

    task = asyncio.create_task(invoke(), name=operation_name)
    cancellation: asyncio.CancelledError | None = None

    try:
        completed, _pending = await asyncio.wait(
            {task},
            timeout=max(0.0, grace_seconds),
        )

        if not completed:
            logger.warning(
                "%s_grace_exceeded grace_seconds=%s awaiting_cleanup_completion=true",
                operation_name,
                grace_seconds,
            )
    except asyncio.CancelledError as exc:
        cancellation = exc
    finally:
        outcome = await _drain_task(task)

    return _TaskOutcome(
        error=outcome.error,
        cancellation=(
            cancellation
            if cancellation is not None
            else outcome.cancellation
        ),
    )


async def _shutdown_resources(
    application: FastAPI,
    policy: _ShutdownPolicy,
) -> None:
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    deadline = started_at + policy.grace_seconds
    outcomes: list[tuple[str, _TaskOutcome]] = []

    operations: tuple[
        tuple[str, Callable[[], Awaitable[Any]]],
        tuple[str, Callable[[], Awaitable[Any]]],
    ] = (
        ("supervisor_stop", stop_supervisor),
        ("database_close", close_databases),
    )

    try:
        for operation_name, operation in operations:
            grace_seconds = (
                policy.supervisor_grace_seconds
                if operation_name == "supervisor_stop"
                else max(0.0, deadline - loop.time())
            )

            try:
                outcome = await _run_cleanup_phase(
                    operation,
                    operation_name,
                    grace_seconds,
                )
            except BaseException as exc:
                outcome = _TaskOutcome(error=exc)

            outcomes.append((operation_name, outcome))

            if outcome.error is not None:
                error = outcome.error
                logger.error(
                    "%s_failed",
                    operation_name,
                    exc_info=(type(error), error, error.__traceback__),
                )
    finally:
        elapsed = loop.time() - started_at
        complete = len(outcomes) == len(operations) and all(
            outcome.error is None for _operation_name, outcome in outcomes
        )
        application.state.shutdown_complete = complete

        if (
            policy.server_timeout_seconds is not None
            and elapsed >= policy.server_timeout_seconds
        ):
            logger.error(
                "application_shutdown_exceeded_server_budget "
                "elapsed_seconds=%s server_timeout_seconds=%s",
                elapsed,
                policy.server_timeout_seconds,
            )

        logger.info(
            "application_stopped cleanup_complete=%s elapsed_seconds=%s",
            complete,
            elapsed,
        )

    errors = [
        outcome.error
        for _operation_name, outcome in outcomes
        if outcome.error is not None
    ]

    for error in errors:
        if not isinstance(error, (Exception, asyncio.CancelledError)):
            raise error

    for _operation_name, outcome in outcomes:
        if outcome.cancellation is not None:
            raise outcome.cancellation

    for error in errors:
        if isinstance(error, asyncio.CancelledError):
            raise error

    if errors:
        raise RuntimeError(
            f"Application shutdown failed in {len(errors)} cleanup operation(s)."
        ) from errors[0]


@asynccontextmanager
async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
    policy: _ShutdownPolicy = _application.state.shutdown_policy
    primary_error: BaseException | None = None

    _application.state.shutting_down = False
    _application.state.startup_complete = False
    _application.state.shutdown_complete = False

    logger.info("application_starting")

    try:
        try:
            await start_supervisor()
        except asyncio.CancelledError:
            logger.warning("supervisor_start_cancelled", exc_info=True)
            raise
        except Exception:
            logger.exception("supervisor_start_failed")
            raise

        _application.state.startup_complete = True
        logger.info("application_started")
        yield
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        _application.state.shutting_down = True
        _application.state.startup_complete = False

        shutdown_task = asyncio.create_task(
            _shutdown_resources(_application, policy),
            name="application_shutdown",
        )
        outcome = await _drain_task(shutdown_task)

        if outcome.error is not None:
            error = outcome.error

            if primary_error is not None:
                logger.error(
                    "application_shutdown_failed_during_existing_error",
                    exc_info=(type(error), error, error.__traceback__),
                )
            elif outcome.cancellation is None:
                raise error

        if outcome.cancellation is not None:
            raise outcome.cancellation


def _status_flag(status: Mapping[str, Any], name: str) -> bool | None:
    value = status.get(name)

    if value is not None and not isinstance(value, bool):
        raise TypeError(f"Supervisor status field {name!r} must be a boolean or null.")

    return value


def _status_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}

    if not isinstance(value, Mapping):
        raise TypeError(f"Supervisor status field {name!r} must be a mapping.")

    return value


def _supervisor_health() -> tuple[bool, dict[str, Any]]:
    status = supervisor_status()

    if not isinstance(status, Mapping):
        raise TypeError("supervisor_status() must return a mapping.")

    enabled = _status_flag(status, "enabled")
    running = _status_flag(status, "running")
    leader = _status_flag(status, "leader")
    breakers = _status_mapping(status.get("breakers"), "breakers")
    metrics = _status_mapping(status.get("metrics"), "metrics")
    open_breakers: list[str] = []

    for name, breaker in breakers.items():
        if not isinstance(name, str) or not name:
            raise TypeError("Supervisor breaker names must be nonempty strings.")

        if not isinstance(breaker, Mapping):
            raise TypeError("Supervisor breaker status values must be mappings.")

        state = breaker.get("state")
        if not isinstance(state, str) or not state.strip():
            raise TypeError("Supervisor breaker states must be nonempty strings.")

        if state.strip().lower() == "open":
            open_breakers.append(name)

    consecutive_failures = status.get("consecutive_failures")

    if consecutive_failures is not None and (
        isinstance(consecutive_failures, bool)
        or not isinstance(consecutive_failures, int)
        or consecutive_failures < 0
    ):
        raise TypeError("consecutive_failures must be a nonnegative integer or null.")

    last_sweep_at = metrics.get("last_sweep_at")

    if last_sweep_at is not None:
        if isinstance(last_sweep_at, bool) or not isinstance(
            last_sweep_at,
            (str, int, float, date, datetime),
        ):
            raise TypeError("last_sweep_at must be a timestamp or null.")

        if isinstance(last_sweep_at, (int, float)) and not math.isfinite(last_sweep_at):
            raise ValueError("last_sweep_at must be finite.")

    stalled = enabled is True and running is not True and leader is not False
    healthy = enabled is not None and not open_breakers and not stalled

    public_status = {
        "available": True,
        "enabled": enabled,
        "running": running,
        "leader": leader,
        "consecutive_failures": consecutive_failures,
        "open_breakers": sorted(open_breakers),
        "last_sweep_at": jsonable_encoder(last_sweep_at),
    }

    return healthy, public_status


class _ApiFallback:
    def __init__(self, routes: Sequence[BaseRoute]) -> None:
        self.routes = tuple(routes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return

        if scope["type"] != "http":
            raise RuntimeError("The API fallback only supports HTTP and WebSocket scopes.")

        allowed_methods: set[str] = set()

        for route in self.routes:
            match, _child_scope = route.matches(scope)

            if match is Match.PARTIAL:
                methods = getattr(route, "methods", None)
                if methods:
                    allowed_methods.update(methods)

        if allowed_methods:
            raise HTTPException(
                status_code=405,
                detail="Method Not Allowed",
                headers={"Allow": ", ".join(sorted(allowed_methods))},
            )

        raise HTTPException(status_code=404, detail="Not Found")


def _validate_reserved_routes(application: FastAPI) -> None:
    for reserved_path in ("/", "/api/health"):
        matching_routes = [
            route
            for route in application.router.routes
            if isinstance(route, Route)
            and route.path == reserved_path
            and (route.methods is None or "GET" in route.methods)
        ]

        if len(matching_routes) != 1:
            raise RuntimeError(
                f"The reserved GET route {reserved_path!r} must be defined exactly once. "
                "Remove the conflicting route from the included router."
            )


def create_app(
    *,
    settings: Any | None = None,
    version: str | None = None,
    shutdown_grace_seconds: float | None = None,
    server_shutdown_timeout_seconds: float | None = None,
) -> FastAPI:
    settings = get_settings() if settings is None else settings
    configure_logging(settings)

    origins, allow_credentials = _cors_configuration(settings)
    application_version = _application_version(settings, version)
    policy = _shutdown_policy(
        settings,
        grace_override=shutdown_grace_seconds,
        server_timeout_override=server_shutdown_timeout_seconds,
    )

    static_directory = resolve_static_directory(settings)
    static_files: _PublicStaticFiles | None = None

    if static_directory is not None:
        static_files = _PublicStaticFiles(
            directory=static_directory,
            allowed_files=_static_allowlist(settings, static_directory),
        )

    app = FastAPI(
        title="Chat Pal Backend",
        version=application_version,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
        lifespan=lifespan,
    )

    app.state.shutdown_policy = policy
    app.state.shutting_down = False
    app.state.startup_complete = False
    app.state.shutdown_complete = False
    app.state.frontend_available = static_files is not None

    register_error_handlers(app)

    @app.get("/api/health", name="health")
    async def health() -> JSONResponse:
        try:
            healthy, public_status = _supervisor_health()
            healthy = healthy and not app.state.shutting_down

            return JSONResponse(
                status_code=200 if healthy else 503,
                content={
                    "status": "ok" if healthy else "degraded",
                    "service": "chat-pal-backend",
                    "supervisor": public_status,
                },
                headers={"Cache-Control": "no-store"},
            )
        except Exception:
            logger.exception("supervisor_status_failed")

        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "service": "chat-pal-backend",
                "supervisor": {
                    "available": False,
                    "enabled": None,
                    "running": None,
                    "leader": None,
                    "consecutive_failures": None,
                    "open_breakers": [],
                    "last_sweep_at": None,
                },
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/", include_in_schema=False, name="frontend_index")
    async def index(request: Request) -> Response:
        if static_files is None:
            raise HTTPException(
                status_code=404,
                detail="The frontend is not configured.",
            )

        target = request.url_for("static", path="chat.html")

        return RedirectResponse(
            url=target.path,
            status_code=307,
            headers={"Cache-Control": "no-store"},
        )

    app.include_router(chat_router.router)
    app.include_router(chat_jobs_router.router)
    app.include_router(conversations_router.router)
    app.include_router(operator_router.router)
    app.include_router(browser_router.router)
    app.include_router(images_router.router)
    app.include_router(public_worker_router.router)
    app.include_router(system_router.router)

    _validate_reserved_routes(app)

    api_fallback = _ApiFallback(app.router.routes)

    app.router.routes.extend(
        [
            Route(
                "/api",
                endpoint=api_fallback,
                name="api_not_found_root",
                include_in_schema=False,
            ),
            Route(
                "/api/{path:path}",
                endpoint=api_fallback,
                name="api_not_found",
                include_in_schema=False,
            ),
            WebSocketRoute(
                "/api",
                endpoint=api_fallback,
                name="api_websocket_not_found_root",
            ),
            WebSocketRoute(
                "/api/{path:path}",
                endpoint=api_fallback,
                name="api_websocket_not_found",
            ),
        ]
    )

    if static_files is not None:
        app.mount("/", static_files, name="static")

    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
        )

    app.add_middleware(RequestContextMiddleware)

    return app


class _LazyApplication:
    def __init__(self, factory: Callable[[], FastAPI]) -> None:
        self._factory = factory
        self._application: FastAPI | None = None
        self._lock = Lock()

    def _get_application(self) -> FastAPI:
        application = self._application

        if application is not None:
            return application

        with self._lock:
            application = self._application

            if application is None:
                application = self._factory()
                self._application = application

            return application

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            application = self._get_application()
        except Exception:
            logger.exception("application_initialization_failed")

            if scope["type"] == "lifespan":
                message = await receive()

                if message["type"] == "lifespan.startup":
                    await send(
                        {
                            "type": "lifespan.startup.failed",
                            "message": (
                                "Application initialization failed. "
                                "Consult the server logs for details."
                            ),
                        }
                    )
                    return

            raise

        await application(scope, receive, send)


app = _LazyApplication(create_app)
