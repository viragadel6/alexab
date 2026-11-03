from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote, urlparse

import httpx

from app.config import get_settings


class DatabaseError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: int,
        code: str | None = None,
        details: Any = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details


def is_new_api_key(value: str) -> bool:
    return value.startswith("sb_publishable_") or value.startswith("sb_secret_")


def encode_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return str(value)


def encode_in_value(values: Iterable[Any]) -> str:
    parts: list[str] = []
    for item in values:
        raw = encode_value(item)
        if any(char in raw for char in (",", '"', "(", ")")):
            raw = '"' + raw.replace('"', '\\"') + '"'
        parts.append(raw)
    return "(" + ",".join(parts) + ")"


class Filter:
    __slots__ = ("column", "operator", "value")

    def __init__(self, column: str, operator: str, value: Any):
        self.column = column
        self.operator = operator
        self.value = value

    def to_params(self) -> dict[str, str]:
        if self.operator == "in":
            return {self.column: f"in.{encode_in_value(self.value)}"}
        if self.operator == "is":
            return {self.column: f"is.{'null' if self.value is None else 'not.null'}"}
        return {self.column: f"{self.operator}.{encode_value(self.value)}"}


class DatabaseResponse:
    __slots__ = ("data", "count", "status_code")

    def __init__(self, data: Any, count: int | None, status_code: int):
        self.data = data
        self.count = count
        self.status_code = status_code

    @property
    def rows(self) -> list[dict[str, Any]]:
        if isinstance(self.data, list):
            return [row for row in self.data if isinstance(row, dict)]
        return []

    def single(self) -> dict[str, Any]:
        rows = self.rows
        if not rows:
            raise DatabaseError("A PostgREST válasz nem tartalmaz sort.", 406)
        return rows[0]

    def maybe_single(self) -> dict[str, Any] | None:
        rows = self.rows
        if not rows:
            return None
        return rows[0]


class SupabaseDatabase:
    def __init__(self, url: str, key: str, timeout: float = 30.0):
        if not url:
            raise DatabaseError("A SUPABASE_URL nincs beállítva.", 500)
        if not key:
            raise DatabaseError("A Supabase kulcs nincs beállítva.", 500)
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise DatabaseError("A SUPABASE_URL érvénytelen.", 500)
        self._base_url = url.rstrip("/")
        self._rest_url = f"{self._base_url}/rest/v1"
        self._key = key
        self._headers = self._build_headers(key)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
            follow_redirects=False,
        )

    @staticmethod
    def _build_headers(key: str) -> dict[str, str]:
        headers = {
            "apikey": key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if not is_new_api_key(key):
            headers["Authorization"] = f"Bearer {key}"
        return headers

    async def close(self) -> None:
        await self._client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        body: Any = None,
        prefer: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> DatabaseResponse:
        request_headers = dict(self._headers)
        if prefer:
            request_headers["Prefer"] = prefer
        if headers:
            request_headers.update(headers)
        url = f"{self._rest_url}/{path.lstrip('/')}"
        request_timeout = httpx.Timeout(timeout, connect=10.0) if timeout else None
        try:
            response = await self._client.request(
                method,
                url,
                params=dict(params) if params else None,
                json=body,
                headers=request_headers,
                timeout=request_timeout,
            )
        except httpx.TimeoutException as exc:
            raise DatabaseError(f"Az adatbázis kérés időtúllépett: {exc}", 504) from exc
        except httpx.HTTPError as exc:
            raise DatabaseError(f"Az adatbázis kérés nem sikerült: {exc}", 502) from exc
        return self._parse(response)

    @staticmethod
    def _parse(response: httpx.Response) -> DatabaseResponse:
        text = response.text
        payload: Any = None
        if text:
            content_type = response.headers.get("content-type", "")
            if "json" in content_type:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = text
            else:
                payload = text
        if response.status_code >= 400:
            message = "Az adatbázis kérés sikertelen."
            code: str | None = None
            details: Any = None
            if isinstance(payload, dict):
                message = str(payload.get("message") or payload.get("error") or message)
                raw_code = payload.get("code")
                code = str(raw_code) if raw_code is not None else None
                details = payload.get("details")
            elif isinstance(payload, str) and payload:
                message = payload
            raise DatabaseError(message, response.status_code, code, details)
        count = _count_from_headers(response.headers)
        if payload is None:
            payload = []
        return DatabaseResponse(payload, count, response.status_code)

    def table(self, name: str) -> "Table":
        return Table(self, name)

    def rpc(self, function: str, payload: Mapping[str, Any] | None = None) -> "RpcBuilder":
        return RpcBuilder(self, function, payload)


def _count_from_headers(headers: Mapping[str, str]) -> int | None:
    content_range = headers.get("content-range")
    if not content_range or "/" not in content_range:
        return None
    total = content_range.rsplit("/", 1)[1].strip()
    if total.isdigit():
        return int(total)
    return None


class Table:
    def __init__(self, database: SupabaseDatabase, name: str):
        self._db = database
        self._name = name
        self._filters: list[Filter] = []
        self._columns: str | None = None
        self._order: list[tuple[str, bool, bool]] = []
        self._limit: int | None = None
        self._offset: int | None = None
        self._count_mode: str | None = None
        self._head = False
        self._method = "GET"
        self._body: Any = None
        self._prefer: str | None = None
        self._on_conflict: str | None = None

    def select(self, *columns: str, count: str | None = None, head: bool = False) -> "Table":
        self._method = "GET"
        self._columns = ",".join(columns) if columns else "*"
        self._count_mode = count
        self._head = head
        return self

    def insert(self, payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> "Table":
        self._method = "POST"
        self._body = payload
        self._prefer = "return=representation"
        return self

    def upsert(
        self,
        payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        on_conflict: str = "",
        ignore_duplicates: bool = False,
    ) -> "Table":
        self._method = "POST"
        self._body = payload
        self._on_conflict = on_conflict
        resolution = "ignoreDuplicates" if ignore_duplicates else "mergeDuplicates"
        self._prefer = f"resolution={resolution},return=representation"
        return self

    def update(self, payload: Mapping[str, Any]) -> "Table":
        self._method = "PATCH"
        self._body = payload
        self._prefer = "return=representation"
        return self

    def delete(self) -> "Table":
        self._method = "DELETE"
        self._prefer = "return=representation"
        return self

    def eq(self, column: str, value: Any) -> "Table":
        self._filters.append(Filter(column, "eq", value))
        return self

    def neq(self, column: str, value: Any) -> "Table":
        self._filters.append(Filter(column, "neq", value))
        return self

    def gt(self, column: str, value: Any) -> "Table":
        self._filters.append(Filter(column, "gt", value))
        return self

    def gte(self, column: str, value: Any) -> "Table":
        self._filters.append(Filter(column, "gte", value))
        return self

    def lt(self, column: str, value: Any) -> "Table":
        self._filters.append(Filter(column, "lt", value))
        return self

    def lte(self, column: str, value: Any) -> "Table":
        self._filters.append(Filter(column, "lte", value))
        return self

    def like(self, column: str, pattern: str) -> "Table":
        self._filters.append(Filter(column, "like", pattern))
        return self

    def ilike(self, column: str, pattern: str) -> "Table":
        self._filters.append(Filter(column, "ilike", pattern))
        return self

    def is_(self, column: str, value: Any) -> "Table":
        self._filters.append(Filter(column, "is", value))
        return self

    def in_(self, column: str, values: Iterable[Any]) -> "Table":
        self._filters.append(Filter(column, "in", list(values)))
        return self

    def order(self, column: str, desc: bool = False, nullsfirst: bool = False) -> "Table":
        self._order.append((column, desc, nullsfirst))
        return self

    def limit(self, count: int) -> "Table":
        self._limit = count
        return self

    def offset(self, count: int) -> "Table":
        self._offset = count
        return self

    def range(self, start: int, end: int) -> "Table":
        self._offset = start
        self._limit = max(0, end - start)
        return self

    def _build_params(self) -> dict[str, str]:
        params: dict[str, str] = {}
        if self._columns:
            params["select"] = self._columns
        for item in self._filters:
            params.update(item.to_params())
        if self._order:
            parts: list[str] = []
            for column, desc, nullsfirst in self._order:
                piece = f"{column}.{'desc' if desc else 'asc'}"
                if nullsfirst:
                    piece += ".nullsfirst"
                parts.append(piece)
            params["order"] = ",".join(parts)
        if self._limit is not None:
            params["limit"] = str(self._limit)
        if self._offset is not None:
            params["offset"] = str(self._offset)
        if self._on_conflict:
            params["on_conflict"] = self._on_conflict
        return params

    async def execute(self) -> DatabaseResponse:
        prefer = self._prefer
        if self._count_mode == "exact":
            prefer = (prefer + "," if prefer else "") + "count=exact"
        headers: dict[str, str] | None = None
        if self._head:
            headers = {"Accept": "application/json"}
        return await self._db.request(
            self._method,
            self._name,
            params=self._build_params(),
            body=self._body,
            prefer=prefer,
            headers=headers,
        )

    async def single(self) -> dict[str, Any]:
        response = await self.execute()
        return response.single()

    async def maybe_single(self) -> dict[str, Any] | None:
        response = await self.execute()
        return response.maybe_single()

    async def count_exact(self) -> int:
        response = await self._db.request(
            "GET",
            self._name,
            params=self._build_params(),
            prefer="count=exact",
            headers={"Accept": "application/json", "Range": "0-0"},
        )
        if response.count is not None:
            return response.count
        return len(response.rows)


class RpcBuilder:
    def __init__(
        self,
        database: SupabaseDatabase,
        function: str,
        payload: Mapping[str, Any] | None,
    ):
        self._db = database
        self._function = function
        self._payload = dict(payload) if payload else {}

    async def execute(self) -> DatabaseResponse:
        return await self._db.request(
            "POST",
            f"rpc/{self._function}",
            body=self._payload,
            headers={"Accept": "application/json"},
        )

    async def value(self) -> Any:
        response = await self.execute()
        return response.data


_admin: SupabaseDatabase | None = None
_anon: SupabaseDatabase | None = None


def get_admin_database() -> SupabaseDatabase:
    global _admin
    if _admin is None:
        settings = get_settings()
        _admin = SupabaseDatabase(settings.supabase_url, settings.supabase_service_role_key)
    return _admin


def get_anon_database() -> SupabaseDatabase:
    global _anon
    if _anon is None:
        settings = get_settings()
        key = settings.supabase_publishable_key or settings.supabase_anon_key
        _anon = SupabaseDatabase(settings.supabase_url, key)
    return _anon


async def close_databases() -> None:
    global _admin, _anon
    if _admin is not None:
        await _admin.close()
        _admin = None
    if _anon is not None:
        await _anon.close()
        _anon = None


def encode_path_segment(value: str) -> str:
    return quote(value, safe="")
