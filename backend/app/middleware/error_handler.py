from __future__ import annotations

import logging
import traceback
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from app.errors import AgentFault, RateLimitFault

logger = logging.getLogger("app.errors")


async def agent_fault_handler(_request: Request, exc: AgentFault) -> Response:
    headers: dict[str, str] = {"Cache-Control": "no-store"}
    if isinstance(exc, RateLimitFault):
        headers["Retry-After"] = str(exc.retry_after)
    status_code = 400 if exc.status_code == 499 else exc.status_code
    return JSONResponse(
        status_code=status_code, content={"error": exc.to_payload()}, headers=headers
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> Response:
    logger.error(
        "unhandled_error",
        extra={
            "method": request.method,
            "path": request.url.path,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        },
    )
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": "Belső kiszolgálói hiba."}},
        headers={"Cache-Control": "no-store"},
    )


async def http_exception_handler(_request: Request, exc: Any) -> Response:
    status_code = getattr(exc, "status_code", 500)
    detail = getattr(exc, "detail", None)
    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(status_code=status_code, content=detail, headers={"Cache-Control": "no-store"})
    message = detail if isinstance(detail, str) else "Belső kiszolgálói hiba."
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": "HTTP_ERROR", "message": message}},
        headers={"Cache-Control": "no-store"},
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AgentFault, agent_fault_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
    try:
        from fastapi.exceptions import HTTPException

        app.add_exception_handler(HTTPException, http_exception_handler)
    except Exception:
        pass
