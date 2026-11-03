from __future__ import annotations

import sys

if sys.version_info < (3, 10) or sys.version_info >= (4, 0):
    raise RuntimeError("Python >=3.10,<4.0 is required.")

import asyncio
import ipaddress
import json
import re
from collections.abc import Awaitable, Callable, Mapping
from enum import Enum
from typing import Annotated, NoReturn
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic.version import VERSION as PYDANTIC_VERSION

_pydantic_version_match = re.match(r"^([0-9]+)\.([0-9]+)", PYDANTIC_VERSION)

if (
    _pydantic_version_match is None
    or int(_pydantic_version_match.group(1)) != 2
    or int(_pydantic_version_match.group(2)) < 7
):
    raise RuntimeError("Pydantic >=2.7,<3.0 is required.")

from pydantic import (
    AliasChoices,
    ConfigDict,
    JsonValue,
    SecretStr,
    StrictStr,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)

ChatRole = Literal["system", "user", "assistant", "tool"]

NonEmptyString = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1),
]

NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]

GitRevision = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=40,
        max_length=40,
        pattern=r"^[0-9a-fA-F]{40}$",
    ),
]

Sha256HexDigest = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-fA-F]{64}$",
    ),
]

PinnedNixVersion = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=(
            r"^(?:0|[1-9][0-9]*)"
            r"\.(?:0|[1-9][0-9]*)"
            r"(?:\.(?:0|[1-9][0-9]*))?$"
        ),
    ),
]


class _ValidatedModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        revalidate_instances="always",
        hide_input_in_errors=True,
        allow_inf_nan=False,
    )


class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)

    def __format__(self, format_spec: str) -> str:
        return format(self.value, format_spec)


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"Non-standard JSON constant {value!r} is not permitted.")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise ValueError("JSON objects cannot contain duplicate property names.")
        result[key] = value

    return result


def _load_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (ValueError, RecursionError) as exc:
        raise ValueError("A valid JSON object is required.") from exc

    if not isinstance(value, dict):
        raise ValueError("JSON must encode an object.")

    return value


class ToolStatus(_ValueEnum):
    COMPLETED = "completed"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


class TerminalState(_ValueEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RUNTIME_LIMIT = "runtime_limit"
    STEP_LIMIT = "step_limit"
    CONTEXT_LIMIT = "context_limit"


class AgentEventType(_ValueEnum):
    PROGRESS = "progress"
    DELTA = "delta"
    AGENT_EVENT = "agent_event"
    ERROR = "error"
    TERMINAL = "terminal"
    DONE = "done"


class AgentErrorCode(_ValueEnum):
    INVALID_INPUT = "invalid_input"
    MODEL_ERROR = "model_error"
    TOOL_ERROR = "tool_error"
    INVALID_RESPONSE = "invalid_response"
    TRANSPORT_ERROR = "transport_error"
    INTERNAL_ERROR = "internal_error"


class ToolErrorCode(_ValueEnum):
    INVALID_ARGUMENTS = "invalid_arguments"
    EXECUTION_ERROR = "execution_error"
    INVALID_OUTPUT = "invalid_output"
    NOT_FOUND = "not_found"


class FinishReason(_ValueEnum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    FUNCTION_CALL = "function_call"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"


class ToolFunction(_ValidatedModel):
    name: NonEmptyString
    arguments: StrictStr

    @field_validator("arguments")
    @classmethod
    def validate_arguments_json(cls, value: str) -> str:
        _load_json_object(value)
        return value


class ToolCall(_ValidatedModel):
    id: NonEmptyString
    type: Literal["function"] = "function"
    function: ToolFunction

    @model_validator(mode="before")
    @classmethod
    def normalize_function_fields(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value

        data = dict(value)
        flat_fields = {"function_name", "function_arguments"}

        if flat_fields.intersection(data):
            if "function" in data:
                raise ValueError(
                    "Provide either function or the flat function fields, not both."
                )

            if not flat_fields.issubset(data):
                raise ValueError(
                    "function_name and function_arguments must be provided together."
                )

            data["function"] = {
                "name": data.pop("function_name"),
                "arguments": data.pop("function_arguments"),
            }

        return data

    @property
    def function_name(self) -> str:
        return self.function.name

    @property
    def function_arguments(self) -> str:
        return self.function.arguments

    def to_provider_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _ensure_unique_tool_call_ids(calls: list[ToolCall]) -> None:
    identifiers = [call.id for call in calls]

    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Tool-call identifiers must be unique within a response.")


class ChatMessage(_ValidatedModel):
    role: ChatRole
    content: StrictStr | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_message_fields(self) -> ChatMessage:
        if self.role != "assistant" and self.tool_calls is not None:
            raise ValueError("Only assistant messages may contain tool_calls.")

        if self.role == "tool":
            if self.tool_call_id is None:
                raise ValueError("Tool messages require tool_call_id.")

            if self.content is None:
                raise ValueError("Tool messages require string content.")
        elif self.tool_call_id is not None:
            raise ValueError("Only tool messages may contain tool_call_id.")

        if self.role in ("system", "user"):
            if self.content is None or not self.content.strip():
                raise ValueError(
                    "System and user messages require nonempty content."
                )

        if self.role == "assistant":
            has_content = self.content is not None and bool(self.content.strip())

            if not has_content and not self.tool_calls:
                raise ValueError(
                    "Assistant messages require content or at least one tool call."
                )

        if self.tool_calls is not None:
            _ensure_unique_tool_call_ids(self.tool_calls)

        return self

    def to_provider_payload(self) -> dict[str, Any]:
        content = self.content

        if (
            self.role == "assistant"
            and self.tool_calls
            and (content is None or not content.strip())
        ):
            content = None

        payload: dict[str, Any] = {
            "role": self.role,
            "content": content,
        }

        if self.tool_calls is not None:
            payload["tool_calls"] = [
                call.to_provider_payload() for call in self.tool_calls
            ]

        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id

        return payload


class AgentEventData(_ValidatedModel):
    name: NonEmptyString
    data: dict[str, JsonValue] = Field(default_factory=dict)


class AgentEvent(_ValidatedModel):
    type: AgentEventType
    task_id: NonEmptyString
    step: NonNegativeInt = 0
    message: NonEmptyString | None = None
    elapsed_since_task_start_ms: NonNegativeInt | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "elapsed_since_task_start_ms",
            "elapsed_ms",
        ),
        serialization_alias="elapsed_ms",
    )
    content: StrictStr | None = None
    reasoning_content: StrictStr | None = None
    event: AgentEventData | None = None
    code: AgentErrorCode | None = None
    outcome: TerminalState | None = None
    reason: NonEmptyString | None = None
    tool_call_count: NonNegativeInt | None = Field(
        default=None,
        validation_alias=AliasChoices("tool_call_count", "tool_calls"),
    )
    total_task_duration_ms: NonNegativeInt | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "total_task_duration_ms",
            "duration_ms",
        ),
        serialization_alias="duration_ms",
    )

    @property
    def elapsed_ms(self) -> int | None:
        return self.elapsed_since_task_start_ms

    @property
    def duration_ms(self) -> int | None:
        return self.total_task_duration_ms

    @model_validator(mode="after")
    def validate_event_fields(self) -> AgentEvent:
        if self.type is AgentEventType.PROGRESS:
            if self.message is None:
                raise ValueError("Progress events require a nonempty message.")

        elif self.type is AgentEventType.DELTA:
            if not self.content and not self.reasoning_content:
                raise ValueError(
                    "Delta events require nonempty content or reasoning_content."
                )

        elif self.type is AgentEventType.AGENT_EVENT:
            if self.event is None:
                raise ValueError("Agent events require structured event data.")

        elif self.type is AgentEventType.ERROR:
            if self.code is None or self.message is None:
                raise ValueError(
                    "Error events require an error code and a nonempty message."
                )

        elif self.type is AgentEventType.TERMINAL:
            if self.outcome is None:
                raise ValueError("Terminal events require an outcome.")

        if self.type is not AgentEventType.DELTA:
            if self.content is not None or self.reasoning_content is not None:
                raise ValueError(
                    "Only delta events may contain content or reasoning_content."
                )

        if self.event is not None and self.type is not AgentEventType.AGENT_EVENT:
            raise ValueError("Only agent_event events may contain event data.")

        if self.code is not None and self.type is not AgentEventType.ERROR:
            raise ValueError("Only error events may contain an error code.")

        terminal_event_types = (
            AgentEventType.TERMINAL,
            AgentEventType.DONE,
        )

        if self.outcome is not None and self.type not in terminal_event_types:
            raise ValueError("Only terminal and done events may contain an outcome.")

        if self.total_task_duration_ms is not None:
            if self.type not in terminal_event_types:
                raise ValueError(
                    "Total task duration is only valid for terminal and done events."
                )

        if self.reason is not None:
            if self.type not in (
                AgentEventType.ERROR,
                AgentEventType.TERMINAL,
                AgentEventType.DONE,
            ):
                raise ValueError(
                    "Only error, terminal, and done events may contain a reason."
                )

            if self.type in terminal_event_types and self.outcome is None:
                raise ValueError("A terminal reason requires an outcome.")

        if (
            self.outcome is not None
            and self.outcome is not TerminalState.COMPLETED
            and self.reason is None
        ):
            raise ValueError("Non-completed outcomes require a nonempty reason.")

        if (
            self.total_task_duration_ms is not None
            and self.elapsed_since_task_start_ms is not None
            and self.total_task_duration_ms > self.elapsed_since_task_start_ms
        ):
            raise ValueError(
                "Total task duration cannot exceed elapsed time since task start."
            )

        return self

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
        )


class AgentSink:
    def __init__(
        self,
        write_event_json: Callable[[str], Awaitable[None]],
    ) -> None:
        if not callable(write_event_json):
            raise TypeError("write_event_json must be an asynchronous callable.")

        self._write_event_json = write_event_json
        self._delivery_lock = asyncio.Lock()

    async def send(self, event: AgentEvent) -> None:
        if not isinstance(event, AgentEvent):
            raise TypeError("event must be an AgentEvent.")

        validated_event = AgentEvent.model_validate(event)
        text = json.dumps(
            validated_event.to_payload(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

        async with self._delivery_lock:
            await self._write_event_json(text)

    async def enqueue(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a serialized AgentEvent JSON object.")

        event = AgentEvent.model_validate(_load_json_object(text))
        await self.send(event)


class ToolError(_ValidatedModel):
    code: ToolErrorCode
    message: NonEmptyString
    details: dict[str, JsonValue] = Field(default_factory=dict)


class ToolOutput(_ValidatedModel):
    status: ToolStatus
    result: dict[str, JsonValue] = Field(default_factory=dict)
    error: ToolError | None = None
    interruption_reason: NonEmptyString | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_error_details(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value

        data = dict(value)
        status = data.get("status")
        if status not in (ToolStatus.ERROR, ToolStatus.ERROR.value) or data.get("error") is not None:
            return data

        result = data.get("result")
        result_data = result if isinstance(result, Mapping) else {}
        nested_error = result_data.get("error")
        nested_data = nested_error if isinstance(nested_error, Mapping) else {}
        source_code = nested_data.get("code", result_data.get("code"))
        source_message = nested_data.get("message", result_data.get("message"))
        if not isinstance(source_message, str) or not source_message.strip():
            source_message = result_data.get("stderr")
        if not isinstance(source_message, str) or not source_message.strip():
            exit_code = result_data.get("exit_code")
            if isinstance(exit_code, int) and not isinstance(exit_code, bool):
                source_message = (
                    f"Az eszköz végrehajtása sikertelen (kilépési kód: {exit_code})."
                )
            else:
                source_message = "Az eszköz végrehajtása sikertelen."

        if source_code in ("INVALID_ARGUMENT", "INVALID_ARGUMENTS"):
            error_code = ToolErrorCode.INVALID_ARGUMENTS
        elif source_code in ("UNKNOWN_TOOL", "NOT_FOUND"):
            error_code = ToolErrorCode.NOT_FOUND
        elif source_code in ("INVALID_OUTPUT", "PROVIDER_PROTOCOL"):
            error_code = ToolErrorCode.INVALID_OUTPUT
        else:
            error_code = ToolErrorCode.EXECUTION_ERROR

        details: dict[str, JsonValue] = {}
        if source_code is not None:
            details["source_code"] = str(source_code)[:128]
        for key in ("exit_code", "port"):
            detail_value = result_data.get(key)
            if isinstance(detail_value, (str, int, float, bool)) or detail_value is None:
                if detail_value is not None:
                    details[key] = detail_value

        data["error"] = {
            "code": error_code,
            "message": source_message.strip()[:2000],
            "details": details,
        }
        return data

    @model_validator(mode="after")
    def validate_status_details(self) -> ToolOutput:
        if self.status is ToolStatus.ERROR:
            if self.error is None:
                raise ValueError("Error tool outputs require structured error details.")
        elif self.error is not None:
            raise ValueError("Only error tool outputs may contain error details.")

        if self.status is ToolStatus.INTERRUPTED:
            if self.interruption_reason is None:
                raise ValueError(
                    "Interrupted tool outputs require an interruption reason."
                )
        elif self.interruption_reason is not None:
            raise ValueError(
                "Only interrupted tool outputs may contain an interruption reason."
            )

        return self


class PromptTokenDetails(_ValidatedModel):
    audio_tokens: NonNegativeInt | None = None
    cached_tokens: NonNegativeInt | None = None


class CompletionTokenDetails(_ValidatedModel):
    accepted_prediction_tokens: NonNegativeInt | None = None
    audio_tokens: NonNegativeInt | None = None
    reasoning_tokens: NonNegativeInt | None = None
    rejected_prediction_tokens: NonNegativeInt | None = None


class TokenUsage(_ValidatedModel):
    prompt_tokens: NonNegativeInt
    completion_tokens: NonNegativeInt
    total_tokens: NonNegativeInt
    prompt_tokens_details: PromptTokenDetails | None = None
    completion_tokens_details: CompletionTokenDetails | None = None

    @model_validator(mode="after")
    def validate_token_totals(self) -> TokenUsage:
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError(
                "total_tokens must equal prompt_tokens plus completion_tokens."
            )

        prompt_details = self.prompt_tokens_details

        if prompt_details is not None:
            for count in (
                prompt_details.audio_tokens,
                prompt_details.cached_tokens,
            ):
                if count is not None and count > self.prompt_tokens:
                    raise ValueError(
                        "A prompt token detail cannot exceed prompt_tokens."
                    )

        completion_details = self.completion_tokens_details

        if completion_details is not None:
            for count in (
                completion_details.accepted_prediction_tokens,
                completion_details.audio_tokens,
                completion_details.reasoning_tokens,
                completion_details.rejected_prediction_tokens,
            ):
                if count is not None and count > self.completion_tokens:
                    raise ValueError(
                        "A completion token detail cannot exceed completion_tokens."
                    )

        return self


class ModelResponse(_ValidatedModel):
    content: StrictStr = ""
    calls: list[ToolCall] = Field(default_factory=list)
    finish: FinishReason | None = None
    usage: TokenUsage | None = None

    @model_validator(mode="after")
    def validate_response_fields(self) -> ModelResponse:
        _ensure_unique_tool_call_ids(self.calls)

        if self.finish in (
            FinishReason.TOOL_CALLS,
            FinishReason.FUNCTION_CALL,
        ):
            if not self.calls:
                raise ValueError(
                    "A tool-call finish reason requires at least one tool call."
                )

        return self


_dns_label_pattern = re.compile(
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
)


def _normalize_host(value: str, *, allow_ip_address: bool) -> str:
    if not value or value != value.strip():
        raise ValueError("Host entries must be nonempty without surrounding whitespace.")

    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        address = None

    if address is None:
        try:
            host = value.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise ValueError("Host entries must be valid DNS names or IP addresses.") from exc

        if host.endswith("."):
            host = host[:-1]

        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None

        if address is None:
            if (
                not host
                or len(host) > 253
                or any(
                    _dns_label_pattern.fullmatch(label) is None
                    for label in host.split(".")
                )
            ):
                raise ValueError(
                    "Host entries must be bare DNS names without schemes, ports, or paths."
                )

            return host

    if not allow_ip_address:
        raise ValueError("Domain entries must be DNS names, not IP addresses.")

    return address.compressed


class RuntimeConfig(_ValidatedModel):
    model_key: SecretStr = Field(repr=False, exclude=True)
    insta_key: SecretStr | None = Field(
        default=None,
        repr=False,
        exclude=True,
    )
    browser_hosts: frozenset[StrictStr] = Field(default_factory=frozenset)
    allowed_domains: frozenset[StrictStr] = Field(default_factory=frozenset)
    nixpkgs_rev: GitRevision
    nix_version: PinnedNixVersion
    nix_installer_sha256: Sha256HexDigest
    deployment_max_request_ms: PositiveInt | None = None

    @field_validator("model_key", "insta_key")
    @classmethod
    def validate_secret(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None

        secret = value.get_secret_value()

        if not secret or any(character.isspace() for character in secret):
            raise ValueError("Configured keys must be nonempty and contain no whitespace.")

        return value

    @field_validator("browser_hosts")
    @classmethod
    def normalize_browser_hosts(cls, value: frozenset[str]) -> frozenset[str]:
        return frozenset(
            _normalize_host(host, allow_ip_address=True)
            for host in value
        )

    @field_validator("allowed_domains")
    @classmethod
    def normalize_allowed_domains(cls, value: frozenset[str]) -> frozenset[str]:
        return frozenset(
            _normalize_host(domain, allow_ip_address=False)
            for domain in value
        )

    @field_validator("nixpkgs_rev", "nix_installer_sha256")
    @classmethod
    def normalize_hexadecimal_values(cls, value: str) -> str:
        return value.lower()

    @field_serializer("browser_hosts", "allowed_domains", when_used="json")
    def serialize_host_collections(self, value: frozenset[str]) -> list[str]:
        return sorted(value)

    @model_validator(mode="after")
    def validate_host_configuration(self) -> RuntimeConfig:
        if self.browser_hosts and self.allowed_domains:
            raise ValueError(
                "browser_hosts and allowed_domains are alternative configuration "
                "inputs; only one may be populated."
            )

        return self
