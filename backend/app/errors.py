from __future__ import annotations

from typing import Any


class AgentFault(Exception):
    def __init__(self, code: str, message: str, status_code: int = 500, details: Any = None):
        super().__init__(message)
        self.name = "AgentFault"
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            payload["details"] = self.details
        return payload


class ConfigurationFault(AgentFault):
    def __init__(self, message: str, code: str = "CONFIGURATION"):
        super().__init__(code, message, 503)


class DeploymentFault(AgentFault):
    def __init__(self, message: str):
        super().__init__("DEPLOYMENT_UNSUPPORTED", message, 503)


class RequestFault(AgentFault):
    def __init__(self, message: str, code: str = "INVALID_REQUEST", status_code: int = 400):
        super().__init__(code, message, status_code)


class PayloadTooLargeFault(AgentFault):
    def __init__(self, message: str = "A kérés túl nagy."):
        super().__init__("REQUEST_TOO_LARGE", message, 413)


class UnsupportedMediaTypeFault(AgentFault):
    def __init__(self, message: str = "application/json szükséges."):
        super().__init__("UNSUPPORTED_MEDIA_TYPE", message, 415)


class AuthenticationFault(AgentFault):
    def __init__(self, message: str = "Bejelentkezés szükséges."):
        super().__init__("UNAUTHORIZED", message, 401)


class ForbiddenFault(AgentFault):
    def __init__(self, message: str = "Tiltott művelet."):
        super().__init__("FORBIDDEN", message, 403)


class NotFoundFault(AgentFault):
    def __init__(self, message: str = "Nem található."):
        super().__init__("NOT_FOUND", message, 404)


class ConflictFault(AgentFault):
    def __init__(self, message: str):
        super().__init__("CONFLICT", message, 409)


class RateLimitFault(AgentFault):
    def __init__(self, message: str = "Túl sok kérés. Próbáld újra később.", retry_after: int = 60):
        super().__init__("RATE_LIMITED", message, 429)
        self.retry_after = retry_after


class ProviderFault(AgentFault):
    def __init__(self, code: str, message: str, status_code: int = 502):
        super().__init__(code, message, status_code)


class ProtocolFault(AgentFault):
    def __init__(self, message: str, status_code: int = 502, code: str = "PROTOCOL_ERROR"):
        super().__init__(code, message, status_code)


class RuntimeLimitFault(AgentFault):
    def __init__(self, message: str = "A futásidő lejárt."):
        super().__init__("RUNTIME_LIMIT", message, 408)


class CancelledFault(AgentFault):
    def __init__(self, message: str = "A feladat megszakadt."):
        super().__init__("CANCELLED", message, 499)


class UnknownOutcomeFault(AgentFault):
    def __init__(self, message: str, status_code: int = 504):
        super().__init__("UNKNOWN_OUTCOME", message, status_code)


class ExecutionUnknownFault(AgentFault):
    def __init__(self, message: str = "A végrehajtás kimenetele ismeretlen."):
        super().__init__("EXECUTION_UNKNOWN", message, 502)


class ToolUncertainFault(AgentFault):
    def __init__(self, message: str = "Egy eszközművelet kimenetele ismeretlen vagy megszakadt."):
        super().__init__("TOOL_UNCERTAIN", message, 502)


class ContextLimitFault(AgentFault):
    def __init__(self, message: str = "A szükséges kontextus nem fér a modell költségkeretébe."):
        super().__init__("CONTEXT_LIMIT", message, 413)


class StepLimitFault(AgentFault):
    def __init__(self, message: str = "Az eszközhívási korlátot elértük."):
        super().__init__("STEP_LIMIT", message, 429)
