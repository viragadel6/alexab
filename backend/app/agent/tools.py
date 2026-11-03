from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.errors import RequestFault

MAX_SCRIPT_CHARS = 100_000
MAX_FILE_BYTES = 131_072
MAX_PLAN_LENGTH = 12_000
MAX_TOOL_NAME_CHARS = 200
MAX_TOOL_ARGUMENT_CHARS = 1_048_576
MAX_REASON_QUESTION = 8_000
MAX_REASON_DEPTH = 3
MAX_REASON_BRANCHES = 3
MAX_BROWSER_TEXT_CHARS = 10_000
DEFAULT_EXECUTION_TIMEOUT_SECONDS = 180
MAX_EXECUTION_TIMEOUT_SECONDS = 1_800
MAX_PACKAGE_COUNT = 24
PACKAGE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.+-]{0,99}$")

TOOL_NAMES = (
    "run_shell",
    "run_python",
    "install_packages",
    "write_file",
    "read_file",
    "expose_port",
    "browse_url",
    "update_plan",
    "cognitive_deliberate",
    "reason",
    "update_state",
    "delegate_task",
    "consult_advisor",
    "multi_model_consensus",
    "cloud_browser_start",
    "cloud_browser_navigate",
    "cloud_browser_inspect",
    "cloud_browser_click",
    "cloud_browser_type",
    "cloud_browser_press_key",
    "cloud_browser_stop",
)

SANDBOX_TOOLS = frozenset(
    {
        "run_shell",
        "run_python",
        "install_packages",
        "write_file",
        "read_file",
        "expose_port",
    }
)

BROWSER_USE_TOOLS = frozenset(
    {
        "browse_url",
        "cloud_browser_start",
        "cloud_browser_navigate",
        "cloud_browser_inspect",
        "cloud_browser_click",
        "cloud_browser_type",
        "cloud_browser_press_key",
        "cloud_browser_stop",
    }
)

@dataclass(frozen=True)
class StringField:
    description: str
    required: bool = True
    min_length: int = 1
    max_length: int = 4096
    max_bytes: int | None = None
    nul_forbidden: bool = False
    choices: tuple[str, ...] = ()

    kind = "string"

    def validate(self, value: Any, key: str) -> str:
        if not isinstance(value, str):
            raise RequestFault(f"Invalid argument: {key}", code="INVALID_ARGUMENT")
        if len(value) < self.min_length or len(value) > self.max_length:
            raise RequestFault(f"Invalid argument: {key}", code="INVALID_ARGUMENT")
        if self.min_length > 0 and not value.strip():
            raise RequestFault(f"Invalid argument: {key}", code="INVALID_ARGUMENT")
        if self.nul_forbidden and "\x00" in value:
            raise RequestFault(f"Invalid argument: {key}", code="INVALID_ARGUMENT")
        if self.choices and value not in self.choices:
            raise RequestFault(f"Invalid argument: {key}", code="INVALID_ARGUMENT")
        if self.max_bytes is not None and len(value.encode("utf-8")) > self.max_bytes:
            raise RequestFault(f"Invalid argument: {key}", code="INVALID_ARGUMENT")
        return value


@dataclass(frozen=True)
class IntegerField:
    description: str
    required: bool = True
    minimum: int = 0
    maximum: int = 0
    default_value: int | None = None

    kind = "integer"

    def validate(self, value: Any, key: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise RequestFault(f"Invalid integer: {key}", code="INVALID_ARGUMENT")
        if value < self.minimum or value > self.maximum:
            raise RequestFault(f"Invalid integer: {key}", code="INVALID_ARGUMENT")
        return value


@dataclass(frozen=True)
class StringArrayField:
    description: str
    required: bool = True
    min_items: int = 1
    max_items: int = 1
    item_pattern: re.Pattern[str] = field(default=re.compile(r"^.*$"))

    kind = "string_array"

    def validate(self, value: Any, key: str) -> list[str]:
        if not isinstance(value, list):
            raise RequestFault(f"Invalid list: {key}", code="INVALID_ARGUMENT")
        if len(value) < self.min_items or len(value) > self.max_items:
            raise RequestFault(f"Invalid list: {key}", code="INVALID_ARGUMENT")
        for item in value:
            if not isinstance(item, str) or not self.item_pattern.match(item):
                raise RequestFault(f"Invalid list: {key}", code="INVALID_ARGUMENT")
        return list(value)


FieldSpec = StringField | IntegerField | StringArrayField


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    fields: dict[str, FieldSpec]
    require_any: bool = False


def _string(
    description: str,
    max_length: int,
    required: bool = True,
    nul_forbidden: bool = False,
    choices: tuple[str, ...] = (),
) -> StringField:
    return StringField(
        description=description,
        required=required,
        min_length=1 if required else 0,
        max_length=max_length,
        nul_forbidden=nul_forbidden,
        choices=choices,
    )


def _integer(
    description: str,
    minimum: int,
    maximum: int,
    required: bool = True,
    default_value: int | None = None,
) -> IntegerField:
    return IntegerField(
        description=description,
        required=required,
        minimum=minimum,
        maximum=maximum,
        default_value=default_value,
    )


TOOL_SPECS: dict[str, ToolSpec] = {
    "run_shell": ToolSpec(
        name="run_shell",
        description=(
            "Run a command in the isolated Linux VM assigned to the task. Files and installed "
            "tools persist for the lifetime of the VM; commands are executed as standalone "
            "shell processes."
        ),
        fields={
            "command": _string("Bash command", MAX_SCRIPT_CHARS),
            "timeout": _integer(
                "Execution timeout in seconds, default 180, maximum 1800",
                1,
                MAX_EXECUTION_TIMEOUT_SECONDS,
                required=False,
                default_value=DEFAULT_EXECUTION_TIMEOUT_SECONDS,
            ),
        },
    ),
    "run_python": ToolSpec(
        name="run_python",
        description=(
            "Run a Python 3 program in the Linux VM assigned to the task; interpreter state does "
            "not persist between calls."
        ),
        fields={
            "code": _string("Python 3 source code", MAX_SCRIPT_CHARS),
            "timeout": _integer(
                "Execution timeout in seconds, default 180, maximum 1800",
                1,
                MAX_EXECUTION_TIMEOUT_SECONDS,
                required=False,
                default_value=DEFAULT_EXECUTION_TIMEOUT_SECONDS,
            ),
        },
    ),
    "install_packages": ToolSpec(
        name="install_packages",
        description=(
            "Install Nix packages in the VM. Package availability depends on the Nix channel, "
            "network connectivity and installation success."
        ),
        fields={
            "packages": StringArrayField(
                description="Valid nixpkgs attribute names",
                min_items=1,
                max_items=MAX_PACKAGE_COUNT,
                item_pattern=PACKAGE_PATTERN,
            )
        },
    ),
    "write_file": ToolSpec(
        name="write_file",
        description=(
            "Atomically write a UTF-8 text file and verify its SHA-256 checksum inside the VM; "
            "up to 128 KiB (131072 bytes) in UTF-8 encoding."
        ),
        fields={
            "path": _string("File path", 4096, nul_forbidden=True),
            "content": StringField(
                description="UTF-8 file content, may be empty, maximum 131072 bytes",
                required=True,
                min_length=0,
                max_length=MAX_FILE_BYTES,
                max_bytes=MAX_FILE_BYTES,
            ),
        },
    ),
    "read_file": ToolSpec(
        name="read_file",
        description=(
            "Page through a file at the byte level; the reply contains Base64-encoded data, "
            "bytes_read, next_offset and eof fields. One page is 6000 bytes."
        ),
        fields={
            "path": _string("File path", 4096, nul_forbidden=True),
            "offset": _integer(
                "Non-negative, safe integer byte offset",
                0,
                2**53 - 1,
                required=False,
                default_value=0,
            ),
        },
    ),
    "expose_port": ToolSpec(
        name="expose_port",
        description=(
            "Verify a TCP service running inside the VM and obtain an InstaVM port URL reachable "
            "for the duration of the task. The VM is destroyed when the task ends."
        ),
        fields={"port": _integer("TCP port", 1, 65535)},
    ),
    "browse_url": ToolSpec(
        name="browse_url",
        description=(
            "Open an allowed HTTPS URL in a temporary Browser Use Cloud browser and return the "
            "page title, visible text, and links."
        ),
        fields={"url": _string("Allowed HTTPS URL", 4096)},
    ),
    "update_plan": ToolSpec(
        name="update_plan",
        description=(
            "Record the current plan and milestones into the running task state."
        ),
        fields={"plan": _string("Full current plan", MAX_PLAN_LENGTH)},
    ),
    "cognitive_deliberate": ToolSpec(
        name="cognitive_deliberate",
        description=(
            "Run the strategic cognitive planner for the current task and create subgoals, "
            "next steps, and verifiable completion conditions. Use only when the task is "
            "multi-step, ambiguous, novel, or consequential. Do not use for greetings, "
            "ordinary conversation, or straightforward one-step answers."
        ),
        fields={
            "focus": _string(
                "Optional detail about the planning question or risk to focus on",
                4000,
                required=False,
            )
        },
    ),
    "reason": ToolSpec(
        name="reason",
        description=(
            "Isolated recursive reasoning: decompose a complex sub-problem into a tree, examine "
            "sub-questions per branch independently and summarize them, without modifying the "
            "global task state. Use only when the task needs deeper analysis; do not use for "
            "greetings, ordinary conversation, or straightforward one-step answers."
        ),
        fields={
            "question": _string("The complex question to investigate", MAX_REASON_QUESTION),
            "depth": _integer(
                "Recursion depth 1-3", 1, MAX_REASON_DEPTH, required=False, default_value=2
            ),
            "branches": _integer(
                "Number of branches per level 1-3",
                1,
                MAX_REASON_BRANCHES,
                required=False,
                default_value=2,
            ),
        },
    ),
    "update_state": ToolSpec(
        name="update_state",
        description=(
            "Update the working memory (Σ): record a fact, add or resolve a blocker, mark a "
            "sub-goal as complete by identifier. You must supply at least one non-empty field."
        ),
        require_any=True,
        fields={
            "fact": _string("New verified fact", 400, required=False),
            "blocker": _string("New blocker", 400, required=False),
            "resolve_blocker": _string("Substring of a resolved blocker", 400, required=False),
            "complete_subgoal": _string("Identifier of completed sub-goal", 100, required=False),
        },
    ),
    "delegate_task": ToolSpec(
        name="delegate_task",
        description=(
            "Delegate a sub-task to a fast Gemini sub-agent that can execute code, search Google "
            "and read URLs autonomously, then returns a verified result."
        ),
        fields={
            "task": _string("The precisely worded sub-task", 32_000),
            "context": _string("Background required for the sub-task", 32_000, required=False),
        },
    ),
    "consult_advisor": ToolSpec(
        name="consult_advisor",
        description=(
            "Consult a stronger Gemini model for advice: analyzes the situation, flags errors and "
            "risks, and suggests concrete next steps."
        ),
        fields={
            "question": _string("The question posed to the advisor", 16_000),
            "context": _string("Current state, attempts and outcomes so far", 32_000, required=False),
        },
    ),
    "multi_model_consensus": ToolSpec(
        name="multi_model_consensus",
        description=(
            "Have multiple Gemini models answer the same question independently, then let a "
            "strong model compare the answers, providing consensus, divergences and confidence."
        ),
        fields={
            "question": _string("The question to be analyzed", 16_000),
            "context": _string("Background information", 32_000, required=False),
        },
    ),
    "cloud_browser_start": ToolSpec(
        name="cloud_browser_start",
        description=(
            "Launch a managed Chrome in Browser Use Cloud with proxy and recording, for up to "
            "4 hours. Returns the CDP websocket address and the live view URL."
        ),
        fields={
            "country_code": _string("Proxy country code, default us", 8, required=False),
            "timeout_minutes": _integer(
                "Runtime in minutes", 1, 240, required=False, default_value=240
            ),
        },
    ),
    "cloud_browser_stop": ToolSpec(
        name="cloud_browser_stop",
        description=(
            "Stop the cloud browser and retrieve the recording URL. Shut down every browser you "
            "started when you're done."
        ),
        fields={"browser_id": _string("Identifier returned by cloud_browser_start", 200)},
    ),
    "cloud_browser_navigate": ToolSpec(
        name="cloud_browser_navigate",
        description="Navigate the active Browser Use Cloud browser to an allowed HTTPS URL.",
        fields={
            "browser_id": _string("Identifier returned by cloud_browser_start", 200),
            "url": _string("Allowed HTTPS URL", 4096),
        },
    ),
    "cloud_browser_inspect": ToolSpec(
        name="cloud_browser_inspect",
        description="Read the active browser page title, visible text, and links.",
        fields={"browser_id": _string("Identifier returned by cloud_browser_start", 200)},
    ),
    "cloud_browser_click": ToolSpec(
        name="cloud_browser_click",
        description="Click the active browser page at the specified viewport coordinates.",
        fields={
            "browser_id": _string("Identifier returned by cloud_browser_start", 200),
            "x": _integer("Horizontal viewport coordinate", 0, 10_000),
            "y": _integer("Vertical viewport coordinate", 0, 10_000),
        },
    ),
    "cloud_browser_type": ToolSpec(
        name="cloud_browser_type",
        description="Type text into the currently focused field in the active browser.",
        fields={
            "browser_id": _string("Identifier returned by cloud_browser_start", 200),
            "text": _string("Text to type", MAX_BROWSER_TEXT_CHARS),
        },
    ),
    "cloud_browser_press_key": ToolSpec(
        name="cloud_browser_press_key",
        description="Press a supported key in the active browser.",
        fields={
            "browser_id": _string("Identifier returned by cloud_browser_start", 200),
            "key": _string(
                "Keyboard key",
                16,
                choices=(
                    "Enter",
                    "Tab",
                    "Escape",
                    "Backspace",
                    "Delete",
                    "ArrowUp",
                    "ArrowDown",
                    "ArrowLeft",
                    "ArrowRight",
                    "Home",
                    "End",
                    "PageUp",
                    "PageDown",
                    "Space",
                ),
            ),
        },
    ),
}


def is_tool_name(name: str) -> bool:
    return name in TOOL_SPECS


def validate_arguments(name: str, raw: Any) -> dict[str, Any]:
    if not is_tool_name(name):
        raise RequestFault("Unknown tool.", code="UNKNOWN_TOOL")
    if not isinstance(raw, dict):
        raise RequestFault(
            "Tool arguments must be a JSON object.",
            code="INVALID_ARGUMENT",
        )
    spec = TOOL_SPECS[name]
    validated: dict[str, Any] = {}
    for key in raw:
        if key not in spec.fields:
            raise RequestFault(f"Unknown argument: {key}", code="INVALID_ARGUMENT")
    for key, field_spec in spec.fields.items():
        if key not in raw:
            if field_spec.required:
                raise RequestFault(f"Missing argument: {key}", code="INVALID_ARGUMENT")
            if isinstance(field_spec, IntegerField) and field_spec.default_value is not None:
                validated[key] = field_spec.default_value
            continue
        validated[key] = field_spec.validate(raw[key], key)
    if spec.require_any:
        has_value = False
        for value in validated.values():
            if isinstance(value, str) and value.strip():
                has_value = True
                break
            if isinstance(value, list) and value:
                has_value = True
                break
            if isinstance(value, int):
                has_value = True
                break
        if not has_value:
            raise RequestFault(
                "At least one non-empty argument is required.",
                code="INVALID_ARGUMENT",
            )
    return validated


def provider_tools(names: list[str]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for name in names:
        spec = TOOL_SPECS[name]
        properties: dict[str, Any] = {}
        required: list[str] = []
        for key, field_spec in spec.fields.items():
            if field_spec.required:
                required.append(key)
            if isinstance(field_spec, StringField):
                properties[key] = {
                    "type": "string",
                    "minLength": field_spec.min_length,
                    "maxLength": field_spec.max_length,
                    "description": field_spec.description,
                }
                if field_spec.choices:
                    properties[key]["enum"] = list(field_spec.choices)
            elif isinstance(field_spec, IntegerField):
                properties[key] = {
                    "type": "integer",
                    "minimum": field_spec.minimum,
                    "maximum": field_spec.maximum,
                    "description": field_spec.description,
                }
            elif isinstance(field_spec, StringArrayField):
                properties[key] = {
                    "type": "array",
                    "minItems": field_spec.min_items,
                    "maxItems": field_spec.max_items,
                    "items": {"type": "string", "pattern": field_spec.item_pattern.pattern},
                    "description": field_spec.description,
                }
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
        if spec.require_any:
            parameters["minProperties"] = 1
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec.description,
                    "parameters": parameters,
                },
            }
        )
    return tools


def enabled_tool_names(
    insta_key: str | None,
    browser_use_key: str | None = None,
) -> list[str]:
    names: list[str] = []
    browser_use_enabled = bool(browser_use_key and browser_use_key.strip())
    for name in TOOL_NAMES:
        if name in SANDBOX_TOOLS and insta_key is None:
            continue
        if name in BROWSER_USE_TOOLS and not browser_use_enabled:
            continue
        names.append(name)
    return names


def tool_action_text(name: str) -> str:
    return TOOL_ACTIONS.get(name, "Executing the next action.")


TOOL_ACTIONS: dict[str, str] = {
    "run_shell": "Running a command on the virtual machine.",
    "run_python": "Executing Python on the virtual machine.",
    "browse_url": "Opening and reviewing the web page.",
    "cloud_browser_start": "Starting the cloud browser.",
    "cloud_browser_navigate": "Navigating the cloud browser.",
    "cloud_browser_inspect": "Inspecting the cloud browser page.",
    "cloud_browser_click": "Clicking in the cloud browser.",
    "cloud_browser_type": "Typing in the cloud browser.",
    "cloud_browser_press_key": "Pressing a key in the cloud browser.",
    "cloud_browser_stop": "Stopping the cloud browser and retrieving the recording.",
    "delegate_task": "A helper model is working on the sub-task.",
    "consult_advisor": "Requesting the opinion of a helper model about the blockage.",
    "multi_model_consensus": "Comparing the findings of multiple models.",
    "write_file": "Saving the finished file.",
    "read_file": "Reviewing the file contents.",
    "cognitive_deliberate": "Planning the task using the cognitive system.",
    "reason": "Deliberating the next decision in detail.",
    "update_state": "Updating the status of sub-goals.",
    "update_plan": "Updating the work plan.",
    "install_packages": "Installing the required programs.",
    "expose_port": "Making the completed service accessible.",
}
