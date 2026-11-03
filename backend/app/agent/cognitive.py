from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app import database as database_module
from app.errors import AgentFault

VECTOR_DIMS = (
    "confidence",
    "progress",
    "risk",
    "uncertainty",
    "complexity",
    "novelty",
    "urgency",
    "tool_need",
    "info_need",
    "verification_need",
    "reversibility",
    "resource_cost",
    "alignment",
    "coherence",
    "momentum",
    "blocker_pressure",
)

GATE_WEIGHTS: dict[str, float] = {
    "confidence": 1.6,
    "progress": 0.8,
    "risk": -1.4,
    "uncertainty": -1.2,
    "complexity": -0.4,
    "novelty": -0.3,
    "urgency": 0.3,
    "tool_need": 0.2,
    "info_need": -0.9,
    "verification_need": -0.3,
    "reversibility": 0.7,
    "resource_cost": -0.4,
    "alignment": 1.0,
    "coherence": 0.9,
    "momentum": 0.5,
    "blocker_pressure": -1.3,
}

GATE_BIAS = 0.1

PHASES = (
    "osPerceive",
    "osDeliberate",
    "osAct",
    "osValidate",
    "osReflect",
    "osConsolidate",
)

TRANSITIONS: dict[str, tuple[str, ...]] = {
    "osPerceive": ("osDeliberate", "osAct", "osConsolidate"),
    "osDeliberate": ("osAct", "osConsolidate"),
    "osAct": ("osAct", "osValidate", "osReflect", "osPerceive", "osDeliberate", "osConsolidate"),
    "osValidate": ("osReflect", "osConsolidate"),
    "osReflect": ("osDeliberate", "osConsolidate"),
    "osConsolidate": (),
}

ALLOWED_COMPONENTS = (
    "planner",
    "executor",
    "tool",
    "environment",
    "verifier",
    "model",
)

EMBED_DIM = 256
REASON_MAX_DEPTH = 4
REASON_MAX_BRANCHES = 6
REASON_MAX_CALLS = 300

Think = Callable[[str, str], Awaitable[str]]
Probe = Callable[[str], Awaitable[bool]]

_db_write_lock = asyncio.Lock()
_subgoal_counter = 0


@dataclass
class Subgoal:
    id: str
    text: str
    done: bool = False


@dataclass
class GoalSpec:
    state_pointers: list[dict[str, Any]] = field(default_factory=list)
    required_files: list[str] = field(default_factory=list)
    file_contents: list[dict[str, str]] = field(default_factory=list)

    def copy(self) -> "GoalSpec":
        return GoalSpec(
            state_pointers=[dict(item) for item in self.state_pointers],
            required_files=list(self.required_files),
            file_contents=[dict(item) for item in self.file_contents],
        )


@dataclass
class Sigma:
    objective: str = ""
    strategy: str = ""
    subgoals: list[Subgoal] = field(default_factory=list)
    closed_subgoals: int = 0
    facts: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    vector: list[float] = field(default_factory=list)
    gate: float = 0.0
    decision: str = "act"
    goal: GoalSpec = field(default_factory=GoalSpec)

    def copy(self) -> "Sigma":
        return Sigma(
            objective=self.objective,
            strategy=self.strategy,
            subgoals=[Subgoal(item.id, item.text, item.done) for item in self.subgoals],
            closed_subgoals=self.closed_subgoals,
            facts=list(self.facts),
            blockers=list(self.blockers),
            next_steps=list(self.next_steps),
            vector=list(self.vector),
            gate=self.gate,
            decision=self.decision,
            goal=self.goal.copy(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "strategy": self.strategy,
            "subgoals": [
                {"id": item.id, "text": item.text, "done": item.done} for item in self.subgoals
            ],
            "closed_subgoals": self.closed_subgoals,
            "facts": list(self.facts),
            "blockers": list(self.blockers),
            "next_steps": list(self.next_steps),
            "vector": list(self.vector),
            "gate": self.gate,
            "decision": self.decision,
            "goal": {
                "state_pointers": [dict(item) for item in self.goal.state_pointers],
                "required_files": list(self.goal.required_files),
                "file_contents": [dict(item) for item in self.goal.file_contents],
            },
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "Sigma":
        goal = payload.get("goal") if isinstance(payload.get("goal"), dict) else {}
        return Sigma(
            objective=str(payload.get("objective") or ""),
            strategy=str(payload.get("strategy") or ""),
            subgoals=[
                Subgoal(
                    str(item.get("id") or ""),
                    str(item.get("text") or ""),
                    bool(item.get("done")),
                )
                for item in payload.get("subgoals") or []
                if isinstance(item, dict)
            ],
            closed_subgoals=int(payload.get("closed_subgoals") or 0),
            facts=[str(item) for item in payload.get("facts") or []],
            blockers=[str(item) for item in payload.get("blockers") or []],
            next_steps=[str(item) for item in payload.get("next_steps") or []],
            vector=[float(item) for item in payload.get("vector") or []],
            gate=float(payload.get("gate") or 0.0),
            decision=str(payload.get("decision") or "act"),
            goal=GoalSpec(
                state_pointers=[dict(item) for item in goal.get("state_pointers") or []],
                required_files=[str(item) for item in goal.get("required_files") or []],
                file_contents=[dict(item) for item in goal.get("file_contents") or []],
            ),
        )


@dataclass
class Reflection:
    failure_point: str
    component: str
    root_cause: str
    pivot_action: str
    patch: str
    failed_action: str
    corrective_action: str


@dataclass
class ReasonNode:
    question: str
    answer: str
    children: list["ReasonNode"]


@dataclass
class Skill:
    id: str
    name: str
    dsl: str
    terms: str
    vector: list[float]
    successes: int
    failures: int


def clip(value: str, max_length: int) -> str:
    if max_length <= 0:
        return ""
    if len(value) <= max_length:
        return value
    end = max_length
    if 0xD800 <= ord(value[end - 1]) <= 0xDBFF:
        end -= 1
    return value[:end]


def is_record(value: Any) -> bool:
    return isinstance(value, dict)


def strip_sensitive(value: Any) -> Any:
    if isinstance(value, list):
        return [strip_sensitive(item) for item in value]
    if not is_record(value):
        return value
    out: dict[str, Any] = {}
    for key, inner in value.items():
        if key not in ("messages", "chain_of_thought"):
            out[key] = strip_sensitive(inner)
    return out


def gate(vector: list[float]) -> tuple[float, str]:
    z = GATE_BIAS
    for index, dim in enumerate(VECTOR_DIMS):
        value = vector[index] if index < len(vector) else 0.5
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            value = 0.5
        z += GATE_WEIGHTS[dim] * float(value)
    g = 1 / (1 + math.exp(-z))
    if g >= 0.5:
        decision = "act"
    elif g >= 0.3:
        decision = "probe"
    else:
        decision = "replan"
    return g, decision


def empty_sigma(objective: str) -> Sigma:
    vector = [0.5] * len(VECTOR_DIMS)
    g, decision = gate(vector)
    return Sigma(
        objective=clip(objective, 4000),
        vector=vector,
        gate=g,
        decision=decision,
    )


def maintain(sigma: Sigma) -> Sigma:
    preserved_goal = sigma.goal.copy()
    closed = len([item for item in sigma.subgoals if item.done])
    clean = strip_sensitive(sigma.to_dict())
    assert isinstance(clean, dict)
    result = Sigma.from_dict(clean)
    result.goal = preserved_goal
    result.subgoals = [
        Subgoal(clip(item.id, 40), clip(item.text, 400), False)
        for item in sigma.subgoals
        if not item.done
    ][:12]
    result.closed_subgoals = sigma.closed_subgoals + closed
    seen: set[str] = set()
    facts: list[str] = []
    for fact in result.facts:
        value = clip(str(fact), 400)
        if value not in seen:
            seen.add(value)
            facts.append(value)
    result.facts = facts[-20:]
    seen.clear()
    blockers: list[str] = []
    for blocker in result.blockers:
        value = clip(str(blocker), 400)
        if value not in seen:
            seen.add(value)
            blockers.append(value)
    result.blockers = blockers[-8:]
    result.next_steps = [clip(str(item), 400) for item in result.next_steps[:8]]
    result.strategy = clip(str(result.strategy), 2000)
    return result


def scan_balanced_json(source: str) -> Any:
    length = len(source)
    for index in range(length):
        char = source[index]
        if char not in "{[":
            continue
        stack = ["}" if char == "{" else "]"]
        in_string = False
        escape = False
        for position in range(index + 1, length):
            current = source[position]
            if in_string:
                if escape:
                    escape = False
                elif current == "\\":
                    escape = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                stack.append("}")
            elif current == "[":
                stack.append("]")
            elif current in "}]":
                if stack and stack[-1] == current:
                    stack.pop()
                    if not stack:
                        try:
                            return json.loads(source[index : position + 1])
                        except json.JSONDecodeError:
                            break
                else:
                    break
    return None


def extract_json(text: str) -> Any:
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE):
        candidate = match.group(1)
        if candidate:
            parsed = scan_balanced_json(candidate)
            if parsed is not None:
                return parsed
    return scan_balanced_json(text)


def collect_strings(value: Any, limit: int) -> list[str]:
    if limit <= 0 or not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()][:limit]


async def deliberate(think: Think, sigma: Sigma, trace: str, guidance: str) -> Sigma:
    global _subgoal_counter
    system = "\n".join(
        [
            "Te a System 2 stratégiai tervező vagy. Kizárólag egyetlen JSON objektumot adj vissza, magyarázat nélkül.",
            'Séma: {"strategy": string, "subgoals": [{"id": string, "text": string, "done": boolean}], "next_steps": [string], "vector": [16 szám 0 és 1 között], "goal": {"state_pointers": [{"pointer": JSON-pointer a munkamemóriába, "equals": érték}], "required_files": [abszolút útvonal], "file_contents": [{"path": string, "contains": string}]}}',
            f"A vector dimenziói sorrendben: {', '.join(VECTOR_DIMS)}.",
            "A next_steps legfeljebb 5 konkrét, azonnal végrehajtható mikrolépés a System 1 számára.",
            "A goal csak olyan ellenőrizhető feltételeket tartalmazzon, amelyek a feladat valódi befejezését bizonyítják; ha nincs fájlkimenet, hagyd üresen a fájllistákat.",
            "Egyszerű beszélgetési üzenetnél vagy egylépéses kérdésnél a subgoals és a goal minden listája legyen üres.",
        ]
    )
    user = (
        f"MUNKAMEMÓRIA:\n{json.dumps(sigma.to_dict(), ensure_ascii=False)}"
        f"\n\nTANULT ÚTMUTATÁS:\n{guidance or 'nincs'}"
        f"\n\nFRISS NYOMVONAL:\n{clip(trace, 8000)}"
    )
    raw_response = await think(system, user)
    parsed = extract_json(raw_response)
    if not is_record(parsed):
        raise AgentFault("COGNITIVE_INVALID_JSON", "A stratégiai tervező (deliberate) érvénytelen JSON választ adott.", 502)
    result = sigma.copy()
    if isinstance(parsed.get("strategy"), str):
        result.strategy = parsed["strategy"]
    if isinstance(parsed.get("subgoals"), list):
        existing_by_id = {item.id: item for item in sigma.subgoals}
        existing_by_text = {item.text: item for item in sigma.subgoals}
        used_ids = {item.id for item in sigma.subgoals}
        taken: set[str] = set()
        collected: list[Subgoal] = []
        for entry in parsed["subgoals"]:
            if not is_record(entry):
                continue
            text = entry.get("text")
            text = text.strip() if isinstance(text, str) else ""
            if not text:
                continue
            raw_id = entry.get("id")
            raw_id = raw_id.strip() if isinstance(raw_id, str) else ""
            existing = existing_by_id.get(raw_id) if raw_id else None
            if existing is None:
                existing = existing_by_text.get(text)
            if existing is not None:
                if existing.id in taken:
                    continue
                taken.add(existing.id)
                collected.append(Subgoal(existing.id, text, existing.done))
                continue
            identifier = raw_id
            if not identifier or identifier in used_ids:
                while True:
                    _subgoal_counter += 1
                    identifier = f"sg_{_subgoal_counter:08x}"
                    if identifier not in used_ids:
                        break
            used_ids.add(identifier)
            collected.append(Subgoal(identifier, text, False))
        result.subgoals = collected
    if isinstance(parsed.get("next_steps"), list):
        result.next_steps = collect_strings(parsed["next_steps"], 5)
    vector = parsed.get("vector")
    if isinstance(vector, list) and len(vector) == len(VECTOR_DIMS):
        result.vector = [
            min(1.0, max(0.0, float(item)))
            if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item)
            else 0.5
            for item in vector
        ]
    goal = parsed.get("goal")
    if is_record(goal):
        pointers = goal.get("state_pointers")
        required_files = goal.get("required_files")
        file_contents = goal.get("file_contents")
        result.goal = GoalSpec(
            state_pointers=[
                {"pointer": item["pointer"], "equals": item["equals"]}
                for item in (pointers or [])
                if is_record(item)
                and isinstance(item.get("pointer"), str)
                and "equals" in item
                and item.get("equals") is not None
            ][:10],
            required_files=collect_strings(required_files, 20),
            file_contents=[
                {"path": item["path"], "contains": item["contains"]}
                for item in (file_contents or [])
                if is_record(item)
                and isinstance(item.get("path"), str)
                and isinstance(item.get("contains"), str)
            ][:20],
        )
    g, decision = gate(result.vector)
    result.gate = g
    result.decision = decision
    return maintain(result)


def apply_state_update(sigma: Sigma, args: dict[str, Any]) -> Sigma:
    result = sigma.copy()
    fact = args.get("fact")
    if isinstance(fact, str) and fact.strip():
        result.facts.append(fact.strip())
    blocker = args.get("blocker")
    if isinstance(blocker, str) and blocker.strip():
        result.blockers.append(blocker.strip())
    resolve = args.get("resolve_blocker")
    if isinstance(resolve, str) and resolve.strip():
        key = resolve.strip().lower()
        result.blockers = [
            item for item in result.blockers if item.strip().lower() != key
        ]
    complete = args.get("complete_subgoal")
    if isinstance(complete, str) and complete.strip():
        key = complete.strip()
        for subgoal in result.subgoals:
            if subgoal.id == key or subgoal.text == key:
                subgoal.done = True
    return maintain(result)


def resolve_pointer(root: Any, path: str) -> Any:
    if path == "":
        return root
    if not path.startswith("/"):
        return None
    current = root
    for raw in path[1:].split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not re.match(r"^(0|[1-9][0-9]*)$", key):
                return None
            index = int(key)
            if index < 0 or index >= len(current):
                return None
            current = current[index]
        elif is_record(current):
            if key not in current:
                return None
            current = current[key]
        else:
            return None
    return current


def deep_equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if left is None or right is None:
        return left == right
    if isinstance(left, list):
        if len(left) != len(right):
            return False
        return all(deep_equal(a, b) for a, b in zip(left, right, strict=False))
    if is_record(left) and is_record(right):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(deep_equal(left[key], right[key]) for key in left)
    return left == right


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


async def safe_probe(probe: Probe, command: str) -> bool:
    try:
        return await probe(command)
    except Exception:
        return False


async def verify(sigma: Sigma, probe: Probe | None) -> dict[str, Any]:
    failures: list[str] = []
    payload = sigma.to_dict()
    for pointer in sigma.goal.state_pointers:
        if "equals" not in pointer:
            failures.append(f"Hiányzó elvárt érték (equals): {pointer.get('pointer')}")
            continue
        actual = resolve_pointer(payload, str(pointer.get("pointer") or ""))
        if actual is None or not deep_equal(actual, pointer.get("equals")):
            failures.append(f"Állapot nem egyezik: {pointer.get('pointer')}")
    needs_files = bool(sigma.goal.required_files or sigma.goal.file_contents)
    if needs_files and probe is None:
        failures.append("Fájlellenőrzés szükséges, de nincs elérhető VM.")
    if probe is not None:
        for path in sigma.goal.required_files:
            if not await safe_probe(probe, f"test -f {shell_quote(path)}"):
                failures.append(f"Hiányzó fájl: {path}")
        for entry in sigma.goal.file_contents:
            import base64

            encoded = base64.b64encode(entry["contains"].encode("utf-8")).decode("ascii")
            command = (
                "node -e 'const fs=require(\"fs\");"
                "try{const c=fs.readFileSync(process.argv[1],\"utf8\");"
                "const n=Buffer.from(process.argv[2],\"base64\").toString(\"utf8\");"
                "process.exit(c.includes(n)?0:1);}catch{process.exit(1);}' "
                f"{shell_quote(entry['path'])} {shell_quote(encoded)}"
            )
            if not await safe_probe(probe, command):
                failures.append(f"Fájltartalom nem egyezik: {entry['path']}")
    if sigma.blockers:
        failures.append(f"Nyitott blokkolók: {'; '.join(sigma.blockers)}")
    open_subgoals = [item for item in sigma.subgoals if not item.done]
    if open_subgoals:
        failures.append(f"Lezáratlan részcélok: {', '.join(item.id for item in open_subgoals)}")
    has_evidence = bool(
        sigma.goal.state_pointers
        or sigma.goal.required_files
        or sigma.goal.file_contents
        or sigma.closed_subgoals > 0
        or sigma.facts
    )
    if not has_evidence:
        failures.append("Nincs ellenőrizhető bizonyíték a feladat teljesítésére.")
    return {"ok": not failures, "failures": failures}


async def reflect(
    think: Think, objective: str, trace: str, failed_action: str
) -> Reflection:
    system = "\n".join(
        [
            "Hibaelemző vagy. A nyers nyomvonal alapján végezz gyökérok-elemzést. Kizárólag JSON-t adj vissza.",
            'Séma: {"failure_point": a kudarc pontos lépése, "component": felelős komponens (planner|executor|tool|environment|verifier|model), "root_cause": string, "pivot_action": egyetlen konkrét korrekciós lépés, "patch": javítócsomag leírása, "corrective_action": a sikertelen műveletet közvetlenül kiváltó helyes parancs vagy művelet}',
        ]
    )
    raw = await think(
        system,
        f"CÉL:\n{objective}\n\nSIKERTELEN MŰVELET:\n{clip(failed_action, 2000)}\n\nNYOMVONAL:\n{clip(trace, 10000)}",
    )
    parsed = extract_json(raw)
    if not is_record(parsed):
        raise AgentFault("COGNITIVE_INVALID_JSON", "A hibaelemző (reflect) érvénytelen JSON választ adott.", 502)

    def required_text(key: str) -> str:
        value = parsed.get(key)
        if not isinstance(value, str) or not value.strip():
            raise AgentFault(
                "COGNITIVE_MISSING_FIELD",
                f"A hibaelemzésből hiányzik a kötelező mező: {key}",
                502,
            )
        return clip(value.strip(), 2000)

    def optional_text(key: str) -> str:
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return clip(value.strip(), 2000)
        return ""

    component = required_text("component").lower()
    if component not in ALLOWED_COMPONENTS:
        raise AgentFault(
            "COGNITIVE_INVALID_COMPONENT",
            f"Érvénytelen komponens a hibaelemzésben: {component}",
            502,
        )
    pivot_action = required_text("pivot_action")
    patch = optional_text("patch")
    corrective_action = optional_text("corrective_action") or patch or pivot_action
    return Reflection(
        failure_point=required_text("failure_point"),
        component=component,
        root_cause=required_text("root_cause"),
        pivot_action=pivot_action,
        patch=patch,
        failed_action=clip(failed_action, 2000),
        corrective_action=corrective_action,
    )


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+|\d+|_+", text.lower(), re.UNICODE)


def fnv_hash(text: str) -> int:
    value = 0x811C9DC5
    for char in text:
        value ^= ord(char)
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def embed(text: str) -> list[float]:
    vector = [0.0] * EMBED_DIM
    tokens = tokenize(text)
    for index, token in enumerate(tokens):
        position = fnv_hash(token) % EMBED_DIM
        vector[position] += 1
        if index + 1 < len(tokens):
            next_position = fnv_hash(f"{token} {tokens[index + 1]}") % EMBED_DIM
            vector[next_position] += 0.5
        for offset in range(len(token) - 2):
            trigram = f"#{token[offset : offset + 3]}"
            trigram_position = fnv_hash(trigram) % EMBED_DIM
            vector[trigram_position] += 0.25
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def parse_vector(raw: Any) -> list[float]:
    if isinstance(raw, list):
        result: list[float] = []
        for item in raw:
            if isinstance(item, bool):
                result.append(0.0)
            elif isinstance(item, (int, float)):
                result.append(float(item) if math.isfinite(float(item)) else 0.0)
            elif isinstance(item, str):
                try:
                    result.append(float(item))
                except ValueError:
                    result.append(0.0)
            else:
                result.append(0.0)
        return result
    if isinstance(raw, str):
        trimmed = raw.strip()
        if not trimmed:
            return []
        try:
            parsed = json.loads(trimmed)
        except json.JSONDecodeError:
            inner = trimmed.strip("[](){}")
            if not inner:
                return []
            values: list[float] = []
            for part in inner.split(","):
                try:
                    values.append(float(part.strip()))
                except ValueError:
                    values.append(0.0)
            return values
        if isinstance(parsed, list):
            return parse_vector(parsed)
        return []
    return []


def cosine(left: Any, right: Any) -> float:
    a = parse_vector(left)
    b = parse_vector(right)
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for index in range(len(a)):
        x = a[index]
        y = b[index]
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    denominator = math.sqrt(norm_a) * math.sqrt(norm_b)
    return dot / denominator if denominator > 0 else 0.0


async def retrieve(objective: str, context: str) -> dict[str, Any]:
    database = database_module.get_admin_database()
    skills_response = await (
        database.table("agent_skills")
        .select("id", "name", "dsl", "terms", "vector", "successes", "failures")
        .eq("status", "active")
        .limit(500)
        .execute()
    )
    playbooks_response = await (
        database.table("agent_playbooks")
        .select("content")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    skills = skills_response.rows
    playbooks = playbooks_response.rows
    query = f"{objective}\n{context}"
    query_vector = embed(query)
    query_tokens = set(tokenize(query))
    ranked: list[tuple[Skill, float]] = []
    for row in skills:
        vector = parse_vector(row.get("vector"))
        terms = tokenize(f"{row.get('name', '')} {row.get('terms', '')}")
        if terms:
            matched = len([term for term in terms if term in query_tokens])
            lexical = matched / math.sqrt(len(terms) * max(1, len(query_tokens)))
        else:
            lexical = 0.0
        successes = int(row.get("successes") or 0)
        failures = int(row.get("failures") or 0)
        rate = (successes + 1) / (successes + failures + 2)
        score = 0.5 * cosine(query_vector, vector) + 0.3 * min(1.0, lexical) + 0.2 * rate
        if score > 0.25:
            skill = Skill(
                id=str(row.get("id")),
                name=str(row.get("name")),
                dsl=str(row.get("dsl")),
                terms=str(row.get("terms")),
                vector=vector,
                successes=successes,
                failures=failures,
            )
            ranked.append((skill, score))
    ranked.sort(key=lambda entry: entry[1], reverse=True)
    playbook = str(playbooks[0].get("content") or "") if playbooks else ""
    return {"skills": [skill for skill, _ in ranked[:3]], "playbook": playbook}


def ngrams(text: str) -> list[str]:
    tokens = tokenize(text)
    return [f"{tokens[index]} {tokens[index + 1]}" for index in range(len(tokens) - 1)]


def distribution(grams: list[str]) -> dict[str, float]:
    if not grams:
        return {}
    counts: dict[str, int] = {}
    for gram in grams:
        counts[gram] = counts.get(gram, 0) + 1
    return {key: value / len(grams) for key, value in counts.items()}


async def apply_policy_delta(ngram: str, delta: float) -> None:
    from datetime import UTC, datetime

    database = database_module.get_admin_database()
    for _ in range(8):
        current = await (
            database.table("agent_policy_weights")
            .select("ngram", "weight", "observations")
            .eq("ngram", ngram)
            .limit(1)
            .execute()
        )
        rows = current.rows
        now = datetime.now(UTC).isoformat()
        if not rows:
            try:
                await (
                    database.table("agent_policy_weights")
                    .insert(
                        {
                            "ngram": ngram,
                            "weight": delta,
                            "observations": 1,
                            "updated_at": now,
                        }
                    )
                    .execute()
                )
                return
            except Exception as exc:
                if getattr(exc, "code", None) == "23505":
                    continue
                raise
        row = rows[0]
        observations = int(row.get("observations") or 0)
        weight = float(row.get("weight") or 0.0)
        updated = await (
            database.table("agent_policy_weights")
            .update({"weight": weight + delta, "observations": observations + 1, "updated_at": now})
            .eq("ngram", ngram)
            .eq("observations", observations)
            .execute()
        )
        if updated.rows:
            return
    raise AgentFault(
        "COGNITIVE_POLICY_CONFLICT",
        f"A politikasúly frissítése ütközés miatt sikertelen: {ngram}",
        500,
    )


async def distill(pairs: list[dict[str, Any]]) -> None:
    deltas: dict[str, float] = {}
    epsilon = 1e-3
    learning_rate = 0.5
    for pair in pairs:
        failed = ngrams(str(pair.get("failed") or ""))
        corrective = ngrams(str(pair.get("corrective") or ""))
        if not failed or not corrective:
            continue
        advantage = pair.get("advantage")
        if not isinstance(advantage, (int, float)) or isinstance(advantage, bool) or not math.isfinite(advantage):
            advantage = 0.0
        advantage = min(1.0, max(0.0, float(advantage)))
        if advantage <= 0:
            continue
        p_distribution = distribution(failed)
        q_distribution = distribution(corrective)
        for gram in set(p_distribution) | set(q_distribution):
            q_value = q_distribution.get(gram, 0.0) + epsilon
            p_value = p_distribution.get(gram, 0.0) + epsilon
            delta = learning_rate * advantage * (q_value + p_value) * math.log(q_value / p_value)
            if abs(delta) > 1e-4:
                deltas[gram] = deltas.get(gram, 0.0) + delta
    if not deltas:
        return
    selected = sorted(deltas.items(), key=lambda entry: abs(entry[1]), reverse=True)[:400]
    async with _db_write_lock:
        batch_size = 25
        for offset in range(0, len(selected), batch_size):
            batch = selected[offset : offset + batch_size]
            await asyncio.gather(
                *(apply_policy_delta(gram, delta) for gram, delta in batch)
            )


async def policy_guidance() -> str:
    database = database_module.get_admin_database()
    favor_response = await (
        database.table("agent_policy_weights")
        .select("ngram", "weight")
        .gt("weight", 0.01)
        .order("weight", desc=True)
        .limit(8)
        .execute()
    )
    avoid_response = await (
        database.table("agent_policy_weights")
        .select("ngram", "weight")
        .lt("weight", -0.01)
        .order("weight", desc=False)
        .limit(8)
        .execute()
    )
    lines: list[str] = []
    favor = favor_response.rows
    avoid = avoid_response.rows
    if favor:
        lines.append("FAVOR: " + ", ".join(f'"{row.get("ngram")}"' for row in favor))
    if avoid:
        lines.append("AVOID: " + ", ".join(f'"{row.get("ngram")}"' for row in avoid))
    return "\n".join(lines)


def valid_dsl(dsl: str) -> bool:
    lines = [line.strip() for line in dsl.split("\n")]

    def has(keyword: str) -> bool:
        return any(
            line.startswith(f"{keyword} ") or line.startswith(f"{keyword}:") for line in lines
        )

    return all(
        has(keyword)
        for keyword in ("SKILL", "WHEN", "REQUIRE", "STEP", "VERIFY", "RECOVER")
    )


def skill_solves(skill_tokens: set[str], skill_vector: list[float], task: dict[str, Any]) -> bool:
    needed = list(
        {
            token
            for token in tokenize(f"{task.get('pivot_action', '')} {task.get('root_cause', '')}")
            if len(token) >= 4
        }
    )
    if not needed:
        return False
    covered = len([token for token in needed if token in skill_tokens]) / len(needed)
    if covered < 0.5:
        return False
    objective = f"{task.get('objective', '')}\n{task.get('root_cause', '')}"
    return cosine(embed(objective), skill_vector) >= 0.2


async def regression_gate(
    candidate: str, cluster_reflection_ids: set[str], cluster_task_ids: set[str], component: str
) -> dict[str, Any]:
    database = database_module.get_admin_database()
    active_response = await (
        database.table("agent_skills").select("dsl").eq("status", "active").limit(100).execute()
    )
    reflection_response = await (
        database.table("agent_reflections")
        .select("id", "task_id", "objective", "component", "root_cause", "pivot_action")
        .eq("component", component)
        .order("created_at", desc=True)
        .limit(200)
        .execute()
    )
    seen_objectives: set[str] = set()
    held_out: list[dict[str, Any]] = []
    for row in reflection_response.rows:
        if str(row.get("id")) in cluster_reflection_ids or str(row.get("task_id")) in cluster_task_ids:
            continue
        objective = str(row.get("objective") or "")
        if objective in seen_objectives:
            continue
        seen_objectives.add(objective)
        held_out.append(row)
        if len(held_out) >= 20:
            break
    if not held_out:
        return {"accept": False, "baseline": 0.0, "candidate": 0.0}
    base_skills = [
        (set(tokenize(str(row.get("dsl") or ""))), embed(str(row.get("dsl") or "")))
        for row in active_response.rows
    ]
    candidate_tokens = set(tokenize(candidate))
    candidate_vector = embed(candidate)
    baseline_solved = 0
    candidate_solved = 0
    for task in held_out:
        base_hit = any(
            skill_solves(tokens, vector, task) for tokens, vector in base_skills
        )
        candidate_hit = base_hit or skill_solves(candidate_tokens, candidate_vector, task)
        if base_hit:
            baseline_solved += 1
        if candidate_hit:
            candidate_solved += 1
    baseline = baseline_solved / len(held_out)
    candidate_rate = candidate_solved / len(held_out)
    return {
        "accept": candidate_rate > baseline,
        "baseline": baseline,
        "candidate": candidate_rate,
    }


async def synthesize_skills(think: Think, log: Callable[[str], Awaitable[None]]) -> None:
    database = database_module.get_admin_database()
    reflection_response = await (
        database.table("agent_reflections")
        .select("id", "task_id", "component", "root_cause", "pivot_action", "patch", "objective")
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )
    skills_response = await database.table("agent_skills").select("vector").limit(1000).execute()
    items = [
        {
            **row,
            "v": embed(f"{row.get('component', '')} {row.get('root_cause', '')}"),
        }
        for row in reflection_response.rows
    ]
    known_vectors: list[Any] = [row.get("vector") for row in skills_response.rows]
    used: set[int] = set()
    for index in range(len(items)):
        if index in used:
            continue
        seed = items[index]
        cluster = [index]
        for other_index in range(index + 1, len(items)):
            if other_index in used:
                continue
            other = items[other_index]
            if other.get("component") != seed.get("component"):
                continue
            if cosine(seed["v"], other["v"]) >= 0.55:
                cluster.append(other_index)
        if len(cluster) < 2:
            continue
        used.update(cluster)
        group = [items[position] for position in cluster]
        centroid = embed(
            " ".join(f"{item.get('component', '')} {item.get('root_cause', '')}" for item in group)
        )
        if any(cosine(vector, centroid) >= 0.8 for vector in known_vectors):
            continue
        system = "\n".join(
            [
                "Meta-ágens vagy: a visszatérő hibamintából újrahasznosítható képességet szintetizálsz SKILL DSL nyelven. Csak a DSL-t add vissza.",
                "Kötelező formátum, minden záradék külön sorban:",
                "SKILL <rövid_snake_case_név>",
                "WHEN <alkalmazási feltétel>",
                "REQUIRE <előfeltétel>",
                "STEP <lépés> (több STEP sor is lehet)",
                "VERIFY <ellenőrizhető sikerfeltétel>",
                "RECOVER <helyreállítási lépés hiba esetén>",
            ]
        )
        user_prompt = "\n---\n".join(
            f"KOMPONENS: {item.get('component')}\n"
            f"GYÖKÉROK: {item.get('root_cause')}\n"
            f"KORREKCIÓ: {item.get('pivot_action')}\n"
            f"JAVÍTÁS: {item.get('patch')}"
            for item in group
        )
        dsl = re.sub(r"```[a-z]*\n?|```", "", await think(system, user_prompt)).strip()
        if not valid_dsl(dsl):
            await log("A szintetizált képesség érvénytelen DSL-t adott, elvetve.")
            continue
        name_match = re.search(r"^[ \t]*SKILL[: \t]+(\S[^\n]*)$", dsl, re.MULTILINE)
        name = (
            name_match.group(1).strip()[:80]
            if name_match
            else f"skill_{abs(hash(dsl)) % 10**10:010x}"
        )
        gate_result = await regression_gate(
            dsl,
            {str(item.get("id")) for item in group},
            {str(item.get("task_id")) for item in group},
            str(seed.get("component")),
        )
        await (
            database.table("agent_skills")
            .insert(
                {
                    "name": name,
                    "dsl": dsl,
                    "terms": " ".join(tokenize(dsl)[:200]),
                    "vector": centroid,
                    "status": "active" if gate_result["accept"] else "rejected",
                    "baseline_rate": gate_result["baseline"],
                    "candidate_rate": gate_result["candidate"],
                }
            )
            .execute()
        )
        known_vectors.append(centroid)
        await log(
            f"Új képesség „{name}”: "
            f"{'aktiválva' if gate_result['accept'] else 'elutasítva'} "
            f"(alap {gate_result['baseline']:.2f}, jelölt {gate_result['candidate']:.2f})."
        )


async def evolve_playbook(think: Think, log: Callable[[str], Awaitable[None]]) -> None:
    database = database_module.get_admin_database()
    total = await database.table("agent_tasks").select("id", count="exact", head=True).count_exact()
    last_response = await (
        database.table("agent_playbooks")
        .select("task_count")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    last_count = 0
    if last_response.rows:
        last_count = int(last_response.rows[0].get("task_count") or 0)
    if total - last_count < 5:
        return
    reflection_response = await (
        database.table("agent_reflections")
        .select("component", "root_cause", "pivot_action")
        .order("created_at", desc=True)
        .limit(40)
        .execute()
    )
    skills_response = await (
        database.table("agent_skills").select("name", "dsl").eq("status", "active").limit(20).execute()
    )
    previous_response = await (
        database.table("agent_playbooks")
        .select("content")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    system = (
        "Tapasztalati tudáskonszolidátor vagy. Írj tömör, ember által olvasható magyar működési "
        "kézikönyvet két szakasszal: 'Bevált eljárások' és 'Elkerülendő anti-patternek'. "
        "Legfeljebb 40 pont."
    )
    reflections = "\n".join(
        f"- [{row.get('component')}] {row.get('root_cause')} => {row.get('pivot_action')}"
        for row in reflection_response.rows
    )
    active_skills = "\n\n".join(str(row.get("dsl") or "") for row in skills_response.rows)
    previous_content = str(previous_response.rows[0].get("content") or "nincs") if previous_response.rows else "nincs"
    content = (
        await think(
            system,
            f"ELŐZŐ KÉZIKÖNYV:\n{previous_content}\n\nREFLEXIÓK:\n{reflections}\n\nAKTÍV KÉPESSÉGEK:\n{active_skills}",
        )
    ).strip()
    if not content:
        return
    inserted = await (
        database.table("agent_playbooks")
        .insert({"content": clip(content, 20000), "task_count": total})
        .execute()
    )
    if not inserted.rows:
        return
    own_id = str(inserted.rows[0].get("id"))
    same_count = await (
        database.table("agent_playbooks")
        .select("id")
        .eq("task_count", total)
        .order("created_at", desc=False)
        .order("id", desc=False)
        .execute()
    )
    if same_count.rows and str(same_count.rows[0].get("id")) != own_id:
        await database.table("agent_playbooks").delete().eq("id", own_id).execute()
        return
    await log("A működési kézikönyv frissült.")


async def record_skill_outcome(skill_id: str, verified: bool) -> None:
    database = database_module.get_admin_database()
    for _ in range(8):
        current = await (
            database.table("agent_skills")
            .select("successes", "failures")
            .eq("id", skill_id)
            .limit(1)
            .execute()
        )
        if not current.rows:
            return
        row = current.rows[0]
        successes = int(row.get("successes") or 0)
        failures = int(row.get("failures") or 0)
        patch = (
            {"successes": successes + 1} if verified else {"failures": failures + 1}
        )
        updated = await (
            database.table("agent_skills")
            .update(patch)
            .eq("id", skill_id)
            .eq("successes", successes)
            .eq("failures", failures)
            .execute()
        )
        if updated.rows:
            return
    raise AgentFault(
        "COGNITIVE_SKILL_CONFLICT",
        f"A képesség eredményének rögzítése ütközés miatt sikertelen: {skill_id}",
        500,
    )


async def consolidate(
    think: Think,
    payload: dict[str, Any],
    log: Callable[[str], Awaitable[None]],
) -> None:
    database = database_module.get_admin_database()
    task_id = str(payload.get("taskId") or "")
    objective = str(payload.get("objective") or "")
    outcome = str(payload.get("outcome") or "")
    steps = int(payload.get("steps") or 0)
    verified = bool(payload.get("verified"))
    reflections = payload.get("reflections") or []
    used_skills = payload.get("usedSkills") or []
    await (
        database.table("agent_tasks")
        .upsert(
            {
                "id": task_id,
                "objective": clip(objective, 4000),
                "outcome": outcome,
                "steps": steps,
                "verified": verified,
            },
            on_conflict="id",
        )
        .execute()
    )
    if reflections:
        rows = []
        for reflection in reflections:
            rows.append(
                {
                    "task_id": task_id,
                    "objective": clip(objective, 4000),
                    "failure_point": reflection.get("failure_point", ""),
                    "component": reflection.get("component", ""),
                    "root_cause": reflection.get("root_cause", ""),
                    "pivot_action": reflection.get("pivot_action", ""),
                    "patch": reflection.get("patch", ""),
                    "failed_action": reflection.get("failed_action", ""),
                    "corrective_action": reflection.get("corrective_action", ""),
                }
            )
        await database.table("agent_reflections").insert(rows).execute()
        await distill(
            [
                {
                    "failed": reflection.get("failed_action", ""),
                    "corrective": reflection.get("corrective_action", ""),
                    "advantage": 1 if verified else 0.25,
                }
                for reflection in reflections
                if reflection.get("corrective_action")
            ]
        )
    for skill_id in dict.fromkeys(str(item) for item in used_skills):
        await record_skill_outcome(skill_id, verified)
    async with _db_write_lock:
        if reflections:
            await synthesize_skills(think, log)
        await evolve_playbook(think, log)


async def take_operator_messages(task_id: str) -> list[str]:
    database = database_module.get_admin_database()
    response = await (
        database.table("agent_operator_messages")
        .update({"consumed": True})
        .eq("task_id", task_id)
        .eq("consumed", False)
        .execute()
    )
    rows = sorted(response.rows, key=lambda row: str(row.get("created_at") or ""))
    return [str(row.get("content") or "") for row in rows]


async def post_operator_message(task_id: str, content: str) -> None:
    database = database_module.get_admin_database()
    await (
        database.table("agent_operator_messages")
        .insert({"task_id": task_id, "content": clip(content, 8000)})
        .execute()
    )


async def reason_node(think: Think, question: str, depth: int, branches: int) -> ReasonNode:
    if depth <= 0:
        answer = await think(
            "Izolált következtető vagy. Tömören, tényszerűen válaszolj a kérdésre; ne javasolj állapotmódosítást.",
            question,
        )
        return ReasonNode(question=question, answer=clip(answer, 3000), children=[])
    decomposition = await think(
        f"Bontsd a kérdést legfeljebb {branches} független, kisebb alkérdésre. "
        "Csak JSON tömböt adj vissza stringekkel. Ha a kérdés atomi, adj vissza üres tömböt.",
        question,
    )
    sub_questions = collect_strings(extract_json(decomposition), branches)
    children = [
        await reason_node(think, sub_question, depth - 1, branches)
        for sub_question in sub_questions
    ]
    if children:
        answer = await think(
            "Összegző vagy. Az alkérdésekre adott válaszokból adj tömör, következetes választ az eredeti kérdésre.",
            "EREDETI KÉRDÉS:\n"
            f"{question}\n\nALVÁLASZOK:\n"
            + "\n".join(f"- {child.question}: {child.answer}" for child in children),
        )
    else:
        answer = await think(
            "Izolált következtető vagy. Tömören, tényszerűen válaszolj a kérdésre; ne javasolj állapotmódosítást.",
            question,
        )
    return ReasonNode(question=question, answer=clip(answer, 3000), children=children)


def reason_node_to_dict(node: ReasonNode) -> dict[str, Any]:
    return {
        "question": node.question,
        "answer": node.answer,
        "children": [reason_node_to_dict(child) for child in node.children],
    }


async def reason_tree(think: Think, question: str, depth: int, branches: int) -> ReasonNode:
    if depth < 0 or depth > REASON_MAX_DEPTH:
        raise AgentFault(
            "INVALID_ARGUMENT",
            f"Érvénytelen mélység: 0 és {REASON_MAX_DEPTH} közötti egész szám szükséges.",
        )
    if branches < 1 or branches > REASON_MAX_BRANCHES:
        raise AgentFault(
            "INVALID_ARGUMENT",
            f"Érvénytelen elágazásszám: 1 és {REASON_MAX_BRANCHES} közötti egész szám szükséges.",
        )
    nodes = 0
    level = 1
    for _ in range(depth + 1):
        nodes += level
        level *= branches
    if nodes * 2 > REASON_MAX_CALLS:
        raise AgentFault(
            "INVALID_ARGUMENT",
            f"A kért fa legfeljebb {nodes * 2} modellhívást igényelhet, a korlát {REASON_MAX_CALLS}.",
        )
    return await reason_node(think, question, depth, branches)


class Lifecycle:
    def __init__(self, on_change: Callable[[str, str], Awaitable[None]]):
        self.phase = "osPerceive"
        self.trail: list[str] = ["osPerceive"]
        self._on_change = on_change

    async def to(self, next_phase: str) -> None:
        if next_phase not in TRANSITIONS.get(self.phase, ()):
            raise AgentFault(
                "INVALID_PHASE_TRANSITION",
                f"Tiltott fázisátmenet: {self.phase} -> {next_phase}",
            )
        previous = self.phase
        await self._on_change(previous, next_phase)
        self.phase = next_phase
        self.trail.append(next_phase)
