from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from app.agent.tools import MAX_EXECUTION_TIMEOUT_SECONDS
from app.errors import (
    AgentFault,
    ConfigurationFault,
    ProtocolFault,
    ProviderFault,
    RequestFault,
    RuntimeLimitFault,
    UnknownOutcomeFault,
)
from app.models import RuntimeConfig, ToolOutput, ToolStatus

MAX_FILE_CHUNK_BYTES = 65_536
MAX_FILE_PAGE_BYTES = 6_000
MAX_RESULT_BYTES = 12_000
EXECUTE_WAIT_MAX_SECONDS = 600
CLEANUP_TIMEOUT_SECONDS = 12
VM_LIFETIME_SECONDS = 22_500
MAX_PROVIDER_BYTES = 2_097_152


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int


@dataclass
class BackgroundResult:
    stdout: str
    stderr: str
    stdout_base64_preview: str
    stderr_base64_preview: str
    stdout_bytes: int
    stderr_bytes: int
    stdout_truncated: bool
    stderr_truncated: bool
    stdout_file: str
    stderr_file: str
    exit_code: int
    timed_out: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_base64_preview": self.stdout_base64_preview,
            "stderr_base64_preview": self.stderr_base64_preview,
            "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "stdout_file": self.stdout_file,
            "stderr_file": self.stderr_file,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
        }


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def _private_address(host: str) -> bool:
    normalized = host.lower()
    if (
        normalized == "localhost"
        or normalized.endswith(".localhost")
        or normalized.endswith(".local")
        or normalized.endswith(".internal")
        or normalized == "metadata.google.internal"
    ):
        return True
    try:
        infos = socket.getaddrinfo(normalized, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False
    for info in infos:
        address = info[4][0]
        if _ip_is_private(address):
            return True
    return False


def _ip_is_private(address: str) -> bool:
    try:
        packed = socket.inet_pton(socket.AF_INET, address)
    except OSError:
        try:
            packed = socket.inet_pton(socket.AF_INET6, address)
        except OSError:
            return True
        return _ipv6_is_private(packed)
    octets = list(packed)
    a, b, c = octets[0], octets[1], octets[2]
    if a == 0 or a == 10 or a == 127:
        return True
    if a == 169 and b == 254:
        return True
    if a == 172 and 16 <= b <= 31:
        return True
    if a == 100 and 64 <= b <= 127:
        return True
    if a == 192 and (b == 0 and (c == 0 or c == 2) or b == 168 or (b == 88 and c == 99)):
        return True
    if a == 198 and (b in (18, 19) or (b == 51 and c == 100)):
        return True
    if a == 203 and b == 0 and c == 113:
        return True
    if a >= 224:
        return True
    return False


def _ipv6_is_private(packed: bytes) -> bool:
    value = int.from_bytes(packed, "big")
    if value == 0 or value == 1:
        return True
    if (value >> 125) != 1:
        return True
    if (value >> 112) == 0x2002:
        return True
    if (value >> 105) == 0x100080:
        return True
    if (value >> 96) == 0x20010DB8:
        return True
    if (value >> 80) == 0x200100020000:
        return True
    if (value >> 100) == 0x2001002:
        return True
    if (value >> 108) == 0x3FFF0:
        return True
    return False


async def validate_browser_url(
    url: str,
    allowed_hosts: set[str],
    allow_public_hosts: bool = False,
) -> None:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise RequestFault("Hibás URL.", code="INVALID_ARGUMENT")
    if parsed.scheme != "https":
        raise RequestFault("Ez a böngésző URL nem engedélyezett.", code="NETWORK_DENIED", status_code=403)
    if parsed.username or parsed.password:
        raise RequestFault("Ez a böngésző URL nem engedélyezett.", code="NETWORK_DENIED", status_code=403)
    if parsed.port and parsed.port != 443:
        raise RequestFault("Ez a böngésző URL nem engedélyezett.", code="NETWORK_DENIED", status_code=403)
    hostname = (parsed.hostname or "").lower()
    if hostname not in allowed_hosts and not (allow_public_hosts and not allowed_hosts):
        raise RequestFault("Ez a böngésző URL nem engedélyezett.", code="NETWORK_DENIED", status_code=403)
    if _private_address(hostname):
        raise RequestFault("Privát vagy nem feloldható böngésző cím.", code="NETWORK_DENIED", status_code=403)
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(hostname, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ProviderFault(
            "DNS_RESOLUTION_FAILED",
            f"A böngésző gazdagépe nem oldható fel ({exc.strerror or 'UNKNOWN'}).",
            502,
        ) from exc
    if not infos:
        raise RequestFault(
            "Privát vagy nem feloldható böngésző cím.",
            code="NETWORK_DENIED",
            status_code=403,
        )
    for info in infos:
        if _ip_is_private(info[4][0]):
            raise RequestFault(
                "Privát vagy nem feloldható böngésző cím.",
                code="NETWORK_DENIED",
                status_code=403,
            )


class Sandbox:
    def __init__(
        self,
        key: str,
        config: RuntimeConfig,
        parent_cancel: asyncio.Event,
        deadline: float,
        base_url: str,
    ):
        self._key = key
        self._config = config
        self._parent_cancel = parent_cancel
        self._deadline = deadline
        self._base_url = base_url.rstrip("/")
        self._session_id: str | None = None
        self._browser_id: str | None = None
        self._owned_sessions: set[str] = set()
        self._uncertain_sessions: set[str] = set()
        self._provision_task: asyncio.Task[str] | None = None
        self._disposed = False
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
        self._lock = asyncio.Lock()
        self._created_at = time.monotonic()

    async def close(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        if self._provision_task is not None:
            self._provision_task.cancel()
            try:
                await self._provision_task
            except (asyncio.CancelledError, Exception):
                pass
        failures: list[str] = []
        browser = self._browser_id
        if browser:
            try:
                await self._api(
                    f"/v1/browser/sessions/{quote(browser, safe='')}",
                    "DELETE",
                    None,
                    timeout=CLEANUP_TIMEOUT_SECONDS,
                )
            except Exception:
                failures.append("browser")
        for session in list(self._owned_sessions):
            try:
                await self._api("/kill", "POST", {"session_id": session}, timeout=CLEANUP_TIMEOUT_SECONDS)
            except Exception:
                failures.append(f"vm:{session}")
        for session in list(self._uncertain_sessions):
            try:
                await self._api("/kill", "POST", {"session_id": session}, timeout=CLEANUP_TIMEOUT_SECONDS)
            except Exception:
                pass
        await self._client.aclose()
        if failures:
            raise AgentFault(
                "CLEANUP_FAILED",
                f"Nem ellenőrzött erőforrás-felszabadítás: {', '.join(failures)}",
            )

    async def _api(
        self,
        path: str,
        method: str,
        body: Any,
        timeout: float | None = None,
    ) -> Any:
        self._raise_if_cancelled()
        headers = {"X-API-Key": self._key}
        if body is not None:
            headers["Content-Type"] = "application/json"
        try:
            response = await self._client.request(
                method,
                f"{self._base_url}{path}",
                headers=headers,
                json=body,
                timeout=httpx.Timeout(timeout, connect=10.0) if timeout else None,
            )
        except httpx.TimeoutException as exc:
            raise UnknownOutcomeFault("A sandbox műveletének kimenetele ismeretlen.") from exc
        except httpx.HTTPError as exc:
            raise UnknownOutcomeFault(
                "A sandbox műveletének kimenetele hálózati hiba miatt ismeretlen.", 502
            ) from exc
        if response.status_code >= 400:
            status = response.status_code
            if status in (401, 403):
                raise ProviderFault("PROVIDER_AUTH", "A sandbox hitelesítése vagy jogosultsága sikertelen.")
            if status == 429:
                raise ProviderFault("PROVIDER_RATE_LIMIT", "A sandbox szolgáltató korlátozta a kéréseket.", 503)
            if 400 <= status < 500:
                raise ProviderFault("PROVIDER_REJECTED", f"A sandbox elutasította a kérést: HTTP {status}")
            raise ProviderFault("PROVIDER_HTTP", f"Sandbox HTTP {status}")
        if response.status_code == 204:
            return {}
        raw = response.content
        if len(raw) > MAX_PROVIDER_BYTES:
            raise ProtocolFault("A sandbox válasza meghaladja a bájthatárt.")
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolFault("A sandbox nem JSON választ küldött.") from exc

    def _raise_if_cancelled(self) -> None:
        if self._parent_cancel.is_set():
            raise AgentFault("CANCELLED", "A feladat megszakadt.", 499)

    async def session(self) -> str:
        if self._disposed:
            raise AgentFault("SESSION_TERMINATED", "A VM már lezárult.", 409)
        if self._session_id is not None:
            return self._session_id
        async with self._lock:
            if self._session_id is not None:
                return self._session_id
            if self._provision_task is None:
                self._provision_task = asyncio.create_task(self._provision())
            task = self._provision_task
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done() and not task.cancelled() and task.exception() is not None:
                raise task.exception() from None
            raise
        except Exception:
            async with self._lock:
                if self._provision_task is task:
                    self._provision_task = None
            raise

    async def _provision(self) -> str:
        session_data = await self._api(
            "/v1/sessions/session",
            "POST",
            {
                "vm_lifetime_seconds": VM_LIFETIME_SECONDS,
                "prewarm": False,
            },
            timeout=30,
        )
        session_record = self._output_record(session_data)
        session_id = self._property_string(session_record, "session_id")
        self._uncertain_sessions.add(session_id)
        data = await self._api(
            "/v1/vms?wait=true",
            "POST",
            {
                "session_id": session_id,
                "egress_policy": {
                    "allow_http": False,
                    "allow_https": True,
                    "allow_package_managers": True,
                    "allowed_domains": list(self._config.allowed_domains),
                    "allowed_cidrs": [],
                },
            },
            timeout=90,
        )
        payload = self._output_record(data)
        provisioned_id = self._property_string(payload, "session_id")
        self._owned_sessions.add(provisioned_id)
        self._uncertain_sessions.discard(session_id)
        if provisioned_id != session_id:
            self._uncertain_sessions.add(session_id)
            raise ProtocolFault(
                "A VM munkamenet-azonosítója váratlanul megváltozott.",
                code="PROVIDER_PROTOCOL",
            )
        if payload.get("status") not in ("running", "active"):
            raise ProtocolFault(
                "A VM nem aktív a létrehozás után.", code="PROVIDER_PROTOCOL"
            )
        self._session_id = session_id
        return session_id

    @staticmethod
    def _output_record(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ProtocolFault("A végrehajtási válasz nem objektum.")
        return raw

    @staticmethod
    def _property_string(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ProtocolFault(
                f"Hiányzó vagy hibás szöveges mező: {key}",
                code="INVALID_PROVIDER_RESPONSE",
            )
        return value

    @staticmethod
    def _output_text(raw: dict[str, Any]) -> tuple[str, str]:
        if raw.get("success") is False or raw.get("error") not in (None,):
            raise ProviderFault("PROVIDER_EXECUTION", "Az InstaVM végrehajtási hibát jelentett.")
        exit_code = raw.get("exit_code")
        if isinstance(exit_code, int) and exit_code != 0:
            raise ProviderFault("PROVIDER_EXECUTION", "Az InstaVM hibás kilépési kódot jelentett.")
        stdout = raw.get("stdout")
        if not isinstance(stdout, str):
            stdout = raw.get("output")
        if not isinstance(stdout, str):
            raise ProtocolFault("A végrehajtási válasz nem tartalmaz ellenőrizhető kimenetet.")
        stderr = raw.get("stderr")
        return stdout, stderr if isinstance(stderr, str) else ""

    async def _raw(self, command: str, max_wait_seconds: int) -> tuple[str, str]:
        session_id = await self.session()
        wait = min(EXECUTE_WAIT_MAX_SECONDS, max(1, int(max_wait_seconds)))
        response = await self._api(
            "/execute",
            "POST",
            {
                "session_id": session_id,
                "command": command,
                "language": "bash",
                "timeout": wait,
            },
            timeout=wait + 10,
        )
        return self._output_text(self._output_record(response))

    async def checked(self, command: str, max_wait_seconds: int = 60) -> ExecutionResult:
        marker = f"__CHAT_EXIT_{uuid.uuid4().hex}__"
        wrapper = f"set +e\n(\n{command}\n)\nresult=$?\nprintf '\\n{marker}%s\\n' \"$result\"\nexit 0"
        stdout, stderr = await self._raw(wrapper, max_wait_seconds)
        index = stdout.rfind(f"\n{marker}")
        if index < 0:
            raise UnknownOutcomeFault("Nem érkezett végrehajtási zárójelzés; a kimenetel ismeretlen.", 502)
        tail = stdout[index + len(marker) + 1 :].strip()
        if not tail.isdigit():
            raise ProtocolFault("Érvénytelen végrehajtási kilépési kód.")
        exit_code = int(tail)
        if exit_code > 2**31 - 1:
            raise ProtocolFault("Hibás kilépési kód.")
        return ExecutionResult(stdout=stdout[:index], stderr=stderr, exit_code=exit_code)

    async def _append_bytes(self, target: str, data: bytes) -> ExecutionResult | None:
        destination = shell_quote(target)
        for offset in range(0, len(data), MAX_FILE_CHUNK_BYTES):
            chunk = base64.b64encode(data[offset : offset + MAX_FILE_CHUNK_BYTES]).decode("ascii")
            written = await self.checked(
                f"printf %s {shell_quote(chunk)} | base64 -d >> {destination}", 30
            )
            if written.exit_code != 0:
                return written
        return None

    async def background(
        self, script: str, interpreter: str, timeout_seconds: int
    ) -> BackgroundResult:
        job_id = uuid.uuid4().hex
        directory = f"/tmp/chat-job-{job_id}"
        quoted = shell_quote(directory)
        init = await self.checked(f"umask 077; mkdir -m 700 {quoted} && : > {quoted}/job", 30)
        if init.exit_code != 0:
            raise ProviderFault("EXECUTION_FAILED", "A folyamatkönyvtár létrehozása sikertelen.")
        transfer = await self._append_bytes(f"{directory}/job", script.encode("utf-8"))
        if transfer is not None:
            raise ProviderFault("EXECUTION_FAILED", "A távoli program átvitele sikertelen.")
        if interpreter == "python":
            invocation = "bash -lc 'exec python3 \"$0\"' \"$1/job\""
        else:
            invocation = "bash -l \"$1/job\""
        runner = (
            "set -e\n"
            f"chmod 600 {quoted}/job\n"
            f"cat > {quoted}/runner.sh <<'CHAT_RUNNER'\n"
            "set +e\n"
            f"timeout --signal=TERM --kill-after=5s {timeout_seconds}s {invocation} > \"$1/stdout\" 2> \"$1/stderr\"\n"
            "rc=$?\n"
            "printf '%s\\n' \"$rc\" > \"$1/exit.tmp\"\n"
            "mv \"$1/exit.tmp\" \"$1/exit\"\n"
            "CHAT_RUNNER\n"
            f"chmod 700 {quoted}/runner.sh\n"
            f"setsid nohup bash {quoted}/runner.sh {quoted} </dev/null >/dev/null 2>&1 &\n"
            f"printf '%s\\n' \"$!\" > {quoted}/pid"
        )
        launch = await self.checked(runner, 30)
        if launch.exit_code != 0:
            raise ProviderFault("EXECUTION_FAILED", "A folyamat indítása sikertelen.")
        limit = min(self._deadline, asyncio.get_running_loop().time() + timeout_seconds + 10)
        settled = False
        try:
            while True:
                self._raise_if_cancelled()
                sample = await self.checked(
                    f"if test -f {quoted}/exit; then printf 'FINISHED\\n'; cat {quoted}/exit; else printf 'RUNNING\\n'; fi",
                    20,
                )
                if sample.exit_code != 0:
                    raise UnknownOutcomeFault("A háttérfolyamat státusza nem olvasható.", 502)
                if sample.stdout.startswith("FINISHED\n"):
                    value = sample.stdout[9:].strip()
                    if not value.isdigit():
                        raise ProtocolFault("Hibás folyamatkilépési kód.")
                    status = int(value)
                    capture = (
                        "import json,sys,os,base64\n"
                        "paths=sys.argv[1:]\n"
                        "def read(path):\n"
                        " with open(path,'rb') as f:\n"
                        "  data=f.read(2001)\n"
                        " size=os.path.getsize(path)\n"
                        " if len(data)>2000: data=data[:2000]\n"
                        " try: text=data.decode('utf-8')\n"
                        " except UnicodeDecodeError: text=''\n"
                        " return text,base64.b64encode(data).decode('ascii'),size,size>len(data)\n"
                        "a,ab,asz,at=read(paths[0]);b,bb,bsz,bt=read(paths[1])\n"
                        "print(json.dumps({'stdout':a,'stderr':b,'stdout_base64_preview':ab,"
                        "'stderr_base64_preview':bb,'stdout_bytes':asz,'stderr_bytes':bsz,"
                        "'stdout_truncated':at,'stderr_truncated':bt}))"
                    )
                    logs = await self.checked(
                        f"python3 -c {shell_quote(capture)} {quoted}/stdout {quoted}/stderr", 25
                    )
                    if logs.exit_code != 0:
                        raise UnknownOutcomeFault("A folyamatnapló nem olvasható.", 502)
                    try:
                        details = json.loads(logs.stdout)
                    except json.JSONDecodeError as exc:
                        raise ProtocolFault("Hibás folyamatnapló.") from exc
                    required_types = {
                        "stdout": str,
                        "stderr": str,
                        "stdout_base64_preview": str,
                        "stderr_base64_preview": str,
                        "stdout_bytes": int,
                        "stderr_bytes": int,
                        "stdout_truncated": bool,
                        "stderr_truncated": bool,
                    }
                    if not isinstance(details, dict):
                        raise ProtocolFault("Hiányos folyamatnapló.")
                    for key, expected in required_types.items():
                        value = details.get(key)
                        if expected is bool:
                            if not isinstance(value, bool):
                                raise ProtocolFault("Hiányos folyamatnapló.")
                        elif expected is int:
                            if isinstance(value, bool) or not isinstance(value, int):
                                raise ProtocolFault("Hiányos folyamatnapló.")
                        elif not isinstance(value, str):
                            raise ProtocolFault("Hiányos folyamatnapló.")
                    settled = True
                    return BackgroundResult(
                        stdout=details["stdout"],
                        stderr=details["stderr"],
                        stdout_base64_preview=details["stdout_base64_preview"],
                        stderr_base64_preview=details["stderr_base64_preview"],
                        stdout_bytes=details["stdout_bytes"],
                        stderr_bytes=details["stderr_bytes"],
                        stdout_truncated=details["stdout_truncated"],
                        stderr_truncated=details["stderr_truncated"],
                        stdout_file=f"{directory}/stdout",
                        stderr_file=f"{directory}/stderr",
                        exit_code=status,
                        timed_out=status in (124, 137),
                    )
                if asyncio.get_running_loop().time() >= self._deadline:
                    raise RuntimeLimitFault()
                if asyncio.get_running_loop().time() >= limit:
                    raise UnknownOutcomeFault(
                        "A távoli folyamat nem zárult le ellenőrizhetően.", 504
                    )
                await asyncio.sleep(1.0)
        finally:
            if not settled and self._session_id:
                try:
                    await self._api(
                        "/execute",
                        "POST",
                        {
                            "session_id": self._session_id,
                            "command": (
                                f"if test -f {quoted}/pid; then pid=$(cat {quoted}/pid); "
                                'if kill -0 "$pid" 2>/dev/null; then kill -TERM -- "-$pid" 2>/dev/null '
                                '|| kill -TERM "$pid" 2>/dev/null || true; fi; fi'
                            ),
                            "language": "bash",
                            "timeout": 10,
                        },
                        timeout=CLEANUP_TIMEOUT_SECONDS,
                    )
                except Exception:
                    pass

    async def exec(self, script: str, language: str, timeout_seconds: int) -> ToolOutput:
        result = await self.background(script, language, timeout_seconds)
        status = ToolStatus.COMPLETED if result.exit_code == 0 else ToolStatus.ERROR
        return ToolOutput(status=status, result=result.to_payload())

    async def install(self, packages: list[str]) -> ToolOutput:
        revision = self._config.nixpkgs_rev
        if not re.fullmatch(r"[0-9a-f]{40}", (revision or "").lower()):
            raise ConfigurationFault("Nix telepítéséhez rögzített CHAT_NIXPKGS_REV szükséges.")
        version = self._config.nix_version
        if not re.fullmatch(r"\d+\.\d+\.\d+", version or ""):
            raise ConfigurationFault("Nix telepítéséhez rögzített CHAT_NIX_VERSION szükséges.")
        checksum = self._config.nix_installer_sha256
        if not re.fullmatch(r"[0-9a-f]{64}", (checksum or "").lower()):
            raise ConfigurationFault("Nix telepítéséhez kötelező a CHAT_NIX_INSTALLER_SHA256 érték.")
        installer_url = f"https://releases.nixos.org/nix/nix-{version}/install"
        refs = " ".join(shell_quote(f"github:NixOS/nixpkgs/{revision}#{item}") for item in packages)
        names = " ".join(shell_quote(item) for item in packages)
        nix_flags = "--extra-experimental-features 'nix-command flakes'"
        verifier_source = (
            "import json,sys\n"
            "with open(sys.argv[1],'r',encoding='utf-8') as handle:\n"
            " data=json.load(handle)\n"
            "elements=data.get('elements',[]) if isinstance(data,dict) else []\n"
            "items=list(elements.values()) if isinstance(elements,dict) else list(elements)\n"
            "installed=set()\n"
            "for item in items:\n"
            " if not isinstance(item,dict):\n"
            "  continue\n"
            " path=item.get('attrPath')\n"
            " if not isinstance(path,str):\n"
            "  continue\n"
            " parts=path.split('.',2)\n"
            " if len(parts)==3 and parts[0] in ('legacyPackages','packages'):\n"
            "  installed.add(parts[2])\n"
            "missing=[name for name in sys.argv[2:] if name not in installed]\n"
            "if missing:\n"
            " sys.stderr.write('Hiányzó csomagok: '+' '.join(missing)+'\\n')\n"
            " sys.exit(1)\n"
            "print('Ellenőrzött csomagok: '+' '.join(sys.argv[2:]))"
        )
        script = (
            "set -euo pipefail\n"
            "load_nix() {\n"
            '  if test -f "$HOME/.nix-profile/etc/profile.d/nix.sh"; then\n'
            "    set +u\n"
            '    . "$HOME/.nix-profile/etc/profile.d/nix.sh"\n'
            "    set -u\n"
            "  fi\n"
            "}\n"
            "load_nix\n"
            "if ! command -v nix >/dev/null 2>&1; then\n"
            '  sudo install -d -m 755 -o "$(id -u)" -g "$(id -g)" /nix\n'
            f"  curl --fail --location --show-error --silent {shell_quote(installer_url)} "
            "--output /tmp/chat-nix-installer.sh\n"
            f"  printf '%s  %s\\n' {shell_quote(checksum.lower())} /tmp/chat-nix-installer.sh "
            "| sha256sum --check --status\n"
            "  sh /tmp/chat-nix-installer.sh --no-daemon\n"
            "  load_nix\n"
            "fi\n"
            f"nix {nix_flags} profile add {refs}\n"
            "listing=$(mktemp)\n"
            "verifier=$(mktemp)\n"
            "trap 'rm -f \"$listing\" \"$verifier\"' EXIT\n"
            f"nix {nix_flags} profile list --json > \"$listing\"\n"
            "cat > \"$verifier\" <<'CHAT_VERIFY'\n"
            f"{verifier_source}\n"
            "CHAT_VERIFY\n"
            f"python3 \"$verifier\" \"$listing\" {names}"
        )
        return await self.exec(script, "bash", MAX_EXECUTION_TIMEOUT_SECONDS)

    async def write_bytes(self, path: str, data: bytes) -> ToolOutput:
        destination = shell_quote(path)
        temporary_path = f"{path}.chat-{uuid.uuid4()}.tmp"
        temporary = shell_quote(temporary_path)
        init = await self.checked(
            f"mkdir -p -- \"$(dirname -- {destination})\" && : > {temporary}", 30
        )
        if init.exit_code != 0:
            return ToolOutput(status=ToolStatus.ERROR, result={"exit_code": init.exit_code, "stdout": init.stdout, "stderr": init.stderr})
        try:
            failed = await self._append_bytes(temporary_path, data)
            if failed is not None:
                return ToolOutput(
                    status=ToolStatus.ERROR,
                    result={"exit_code": failed.exit_code, "stdout": failed.stdout, "stderr": failed.stderr},
                )
            expected = hashlib.sha256(data).hexdigest()
            check = await self.checked(
                f'test "$(wc -c < {temporary})" -eq {len(data)} && '
                f"test \"$(sha256sum -- {temporary} | cut -d ' ' -f 1)\" = {shell_quote(expected)} && "
                f"mv -f -- {temporary} {destination}",
                30,
            )
            if check.exit_code != 0:
                return ToolOutput(
                    status=ToolStatus.ERROR,
                    result={"exit_code": check.exit_code, "stdout": check.stdout, "stderr": check.stderr},
                )
            return ToolOutput(
                status=ToolStatus.COMPLETED,
                result={"bytes_written": len(data), "sha256": expected, "path": path},
            )
        finally:
            try:
                await self.checked(f"rm -f -- {temporary}", 20)
            except Exception:
                pass

    async def write_file(self, path: str, content: str) -> ToolOutput:
        data = content.encode("utf-8")
        if len(data) > 131_072:
            raise RequestFault("A fájl bájtban túl nagy.")
        return await self.write_bytes(path, data)

    async def read_file(self, path: str, offset: int) -> ToolOutput:
        python = (
            "import base64,json,sys,os\n"
            "p=sys.argv[1]\n"
            "o=int(sys.argv[2])\n"
            "with open(p,'rb') as f:\n"
            " f.seek(o)\n"
            f" b=f.read({MAX_FILE_PAGE_BYTES})\n"
            " size=f.seek(0,2)\n"
            "print(json.dumps({'data_base64':base64.b64encode(b).decode('ascii'),"
            "'bytes_read':len(b),'next_offset':o+len(b),'eof':o+len(b)>=size,'total_bytes':size}))"
        )
        response = await self.checked(f"python3 -c {shell_quote(python)} {shell_quote(path)} {offset}", 30)
        if response.exit_code != 0:
            return ToolOutput(
                status=ToolStatus.ERROR,
                result={"exit_code": response.exit_code, "stdout": response.stdout, "stderr": response.stderr},
            )
        try:
            parsed = json.loads(response.stdout)
        except json.JSONDecodeError as exc:
            raise ProtocolFault("Hibás fájllapozási válasz.") from exc
        if (
            not isinstance(parsed, dict)
            or not isinstance(parsed.get("data_base64"), str)
            or not isinstance(parsed.get("bytes_read"), int)
            or not isinstance(parsed.get("next_offset"), int)
            or not isinstance(parsed.get("eof"), bool)
            or not isinstance(parsed.get("total_bytes"), int)
            or parsed["bytes_read"] > MAX_FILE_PAGE_BYTES
            or parsed["next_offset"] != offset + parsed["bytes_read"]
        ):
            raise ProtocolFault("Érvénytelen fájllapozási válasz.")
        return ToolOutput(status=ToolStatus.COMPLETED, result=parsed)

    async def expose_port(self, port: int) -> ToolOutput:
        probe = await self.checked(
            "python3 -c "
            + shell_quote(
                "import socket,sys;s=socket.socket();s.settimeout(3);"
                "s.connect(('127.0.0.1',int(sys.argv[1])));s.close()"
            )
            + f" {port}",
            15,
        )
        if probe.exit_code != 0:
            return ToolOutput(
                status=ToolStatus.ERROR,
                result={"message": "A TCP port nem elérhető a VM-ben.", "port": port},
            )
        session_id = await self.session()
        raw = self._output_record(
            await self._api(f"/v1/sessions/app-url/{quote(session_id, safe='')}?port={port}", "GET", None)
        )
        url = raw.get("url")
        if not isinstance(url, str):
            url = raw.get("app_url")
        if not isinstance(url, str) or not url.lower().startswith("https://"):
            raise ProtocolFault(
                "A szolgáltató nem küldött HTTPS port URL-t.", code="PROVIDER_PROTOCOL"
            )
        if not re.match(r"^https://[^/\s]+", url):
            raise ProtocolFault(
                "A szolgáltató nem küldött érvényes HTTPS port URL-t.",
                code="PROVIDER_PROTOCOL",
            )
        return ToolOutput(
            status=ToolStatus.COMPLETED,
            result={
                "url": url,
                "port": port,
                "retention": "A VM és az URL a feladat befejezéséig élhet; a végén a VM leáll.",
                "session_age_ms": int((time.monotonic() - self._created_at) * 1000),
            },
        )

    async def browse(self, url: str) -> ToolOutput:
        await validate_browser_url(url, self._config.browser_hosts)
        if not self._browser_id:
            session = self._output_record(
                await self._api(
                    "/v1/browser/sessions/",
                    "POST",
                    {"viewport_width": 1280, "viewport_height": 900},
                )
            )
            self._browser_id = self._property_string(session, "session_id")
        navigation = await self._api(
            "/v1/browser/interactions/navigate",
            "POST",
            {"url": url, "wait_timeout": 30000, "session_id": self._browser_id},
            timeout=40,
        )
        if not isinstance(navigation, dict) or navigation.get("error") or navigation.get("success") is False:
            raise ProviderFault("BROWSER_NAVIGATION", "A böngésző navigációja sikertelen.")
        result = await self._api(
            "/v1/browser/interactions/content",
            "POST",
            {"session_id": self._browser_id, "include_interactive": False, "include_anchors": True},
            timeout=40,
        )
        if not isinstance(result, dict):
            raise ProviderFault("BROWSER_CONTENT", "Hibás böngésző tartalomválasz.")
        if result.get("error") or result.get("success") is False:
            raise ProviderFault("BROWSER_CONTENT", "A böngésző nem tudta lekérni az oldal tartalmát.")
        if not isinstance(result.get("text"), str) and not isinstance(result.get("content"), str):
            raise ProviderFault(
                "BROWSER_CONTENT", "A böngésző nem adott vissza olvasható szöveget."
            )
        serialized = json.dumps(result, ensure_ascii=False)
        encoded = serialized.encode("utf-8")
        if len(encoded) > MAX_RESULT_BYTES - 2_000:
            artifact_path = f"/tmp/chat-browser-{uuid.uuid4()}.json"
            saved = await self.write_bytes(artifact_path, encoded)
            if saved.status != ToolStatus.COMPLETED:
                return saved
            return ToolOutput(
                status=ToolStatus.COMPLETED,
                result={
                    "artifact_path": artifact_path,
                    "retrieval": "read_file",
                    "preview": serialized[:1000],
                    "original_bytes": len(encoded),
                },
            )
        return ToolOutput(status=ToolStatus.COMPLETED, result=result)


def sanitize_result(output: ToolOutput) -> tuple[ToolStatus, str]:
    value = json.dumps(
        {"status": output.status.value, "result": output.result}, ensure_ascii=False
    )
    size = len(value.encode("utf-8"))
    if size <= MAX_RESULT_BYTES:
        return output.status, value
    return output.status, json.dumps(
        {
            "status": output.status.value,
            "truncated": True,
            "original_bytes": size,
            "message": (
                "A teljes eredmény nagyobb, mint a modell megjelenítési költségkerete. "
                "A fájlokhoz a read_file lapozását használd."
            ),
            "preview": value[:2000],
        },
        ensure_ascii=False,
    )

