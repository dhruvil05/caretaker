"""
caretaker/dashboard/app.py
Local-only dashboard server (Starlette + Uvicorn, both already installed with fastmcp).

Security model — memories are personal, so:
  - Listens on 127.0.0.1 only; requests whose Host header isn't 127.0.0.1 /
    localhost are rejected (blocks DNS-rebinding attacks from web pages).
  - Every /api request needs the random per-run token in the X-Caretaker-Token
    header. Browsers can't add custom headers cross-site without a CORS
    preflight, which this server never approves, so other sites can neither
    read nor change data.
  - Strict Content-Security-Policy: the page loads only its own files, makes
    no outside requests, and memory text is rendered as text, never HTML.

Data model:
  - SQLite is the source of truth. The dashboard writes only to SQLite (never
    to ChromaDB, which isn't safe to share between processes while the MCP
    server runs). Archived memories are excluded from recall immediately;
    edited/restored memories are re-embedded by the next maintenance run.
"""

import json
import logging
import re
import secrets
import socket
import threading
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from caretaker import paths
from caretaker.dashboard import api_v05
from caretaker.storage import local_db

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
TOKEN_HEADER = "x-caretaker-token"
DEFAULT_PORT = 8765

MAX_TEXT_LEN = 20_000
SORTS = {
    "updated": "updated_at DESC",
    "created": "created_at DESC",
    "importance": "importance DESC, updated_at DESC",
    "recalled": "retrieval_count DESC, updated_at DESC",
    "last_used": "last_used IS NULL, last_used DESC",
}
FILTER_COLUMNS = {"type": "type", "temperature": "temperature", "status": "status", "agent": "source_agent",
                  "project": "project", "sensitivity": "sensitivity"}
MEMORY_FIELDS = (
    "id", "type", "subtype", "fact_type", "status", "temperature", "importance",
    "decay_score", "retrieval_count", "source_agent", "short", "full", "keywords",
    "superseded_by", "created_at", "updated_at", "last_used",
    "project", "sensitivity", "privacy_findings", "privacy_reviewed",
)

MEMORY_TYPES = {"PROJECT", "PREFERENCE", "PROBLEM", "DECISION", "LEARNING", "PERSONAL", "EMOTION", "CORRECTION"}

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Cache-Control": "no-store",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

GLOBAL_PROJECT = "__global__"   # query value meaning "memories with no project"


def _project_param(request: Request):
    """None = all projects; '' = global only; otherwise one project."""
    raw = request.query_params.get("project")
    if raw in (None, "", "*"):
        return None
    if raw == GLOBAL_PROJECT:
        return ""
    from caretaker.projects import normalise_project
    return normalise_project(raw)


def _scoped(sql: str, args, project):
    """Restrict a query on `memories` to one project via a CTE."""
    if project is None:
        return sql, list(args)
    return ("WITH scoped AS (SELECT * FROM memories WHERE project = ?) "
            + sql.replace("FROM memories", "FROM scoped"), [project, *args])


def _json(data, status: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status, headers=SECURITY_HEADERS)


def _error(message: str, status: int) -> JSONResponse:
    return _json({"error": message}, status)


def _public_memory(row: dict) -> dict:
    mem = {k: row.get(k) for k in MEMORY_FIELDS}
    kw = mem.get("keywords")
    if isinstance(kw, str):
        try:
            parsed = json.loads(kw)
            mem["keywords"] = parsed if isinstance(parsed, list) else [str(parsed)]
        except Exception:
            mem["keywords"] = [k.strip() for k in kw.split(",") if k.strip()]
    elif kw is None:
        mem["keywords"] = []
    try:
        mem["privacy_findings"] = json.loads(mem.get("privacy_findings") or "[]")
    except (TypeError, ValueError):
        mem["privacy_findings"] = []
    mem["sensitivity"] = mem.get("sensitivity") or "none"
    mem["project"] = mem.get("project") or ""
    return mem


def _dir_size(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except OSError:
        return 0


def _parse_maintenance_log(path: Path, limit: int = 10) -> list:
    """Parse <data_dir>/logs/maintenance.log into a list of runs (newest first)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    runs = []
    for block in re.split(r"\n={20,}\n", text):
        lines = [l for l in block.strip().splitlines() if l.strip()]
        if not lines or not re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", lines[0]):
            continue
        values = {}
        for line in lines[1:]:
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                if key and len(key) < 40:
                    values[key] = value.strip()
        runs.append({"time": lines[0].strip(), "values": values})
    return list(reversed(runs))[:limit]


def _next_maintenance(config: dict) -> Optional[str]:
    if config.get("scheduler_enabled") is False:
        return None
    try:
        hh, mm = str(config.get("maintenance_time", "02:00")).split(":")
        now = datetime.now(timezone.utc)
        nxt = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(days=1)
        return nxt.isoformat()
    except Exception:
        return None


# ── Endpoints ──────────────────────────────────────────────────────────────────

async def index(request: Request) -> Response:
    return FileResponse(STATIC_DIR / "index.html", headers=SECURITY_HEADERS)


async def static_file(request: Request) -> Response:
    name = request.path_params["name"]
    allowed = {"app.js": "text/javascript", "app.css": "text/css", "favicon.svg": "image/svg+xml"}
    if name not in allowed:
        return _error("Not found", 404)
    return FileResponse(STATIC_DIR / name, media_type=allowed[name], headers=SECURITY_HEADERS)


async def overview(request: Request) -> Response:
    config = paths.load_config()
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=29)

    project = _project_param(request)
    with local_db.get_connection() as conn:
        def run(sql, args=()):
            q, a = _scoped(sql, args, project)
            return conn.execute(q, a)

        def scalar(sql, args=()):
            return run(sql, args).fetchone()[0] or 0

        def grouped(sql):
            return [{"key": r[0] if r[0] is not None else "(none)", "count": r[1]}
                    for r in run(sql).fetchall()]

        total = scalar("SELECT COUNT(*) FROM memories")
        by_status = grouped("SELECT status, COUNT(*) FROM memories GROUP BY status ORDER BY 2 DESC")
        by_type = grouped("SELECT type, COUNT(*) FROM memories WHERE status='ACTIVE' GROUP BY type ORDER BY 2 DESC")
        by_temperature = grouped(
            "SELECT temperature, COUNT(*) FROM memories WHERE status='ACTIVE' GROUP BY temperature"
        )
        by_agent = grouped(
            "SELECT source_agent, COUNT(*) FROM memories WHERE status='ACTIVE' GROUP BY source_agent ORDER BY 2 DESC"
        )
        created = dict(run(
            "SELECT substr(created_at, 1, 10) AS d, COUNT(*) FROM memories "
            "WHERE substr(created_at, 1, 10) >= ? GROUP BY d", (start.isoformat(),)
        ).fetchall())
        recalls_total = scalar("SELECT SUM(retrieval_count) FROM memories")
        active = scalar("SELECT COUNT(*) FROM memories WHERE status='ACTIVE'")
        uncompressed = scalar(
            "SELECT COUNT(*) FROM memories WHERE status='ACTIVE' AND (short IS NULL OR short = '')"
        )
        compressed = active - uncompressed
        recalled = scalar("SELECT COUNT(*) FROM memories WHERE status='ACTIVE' AND retrieval_count > 0")
        hot = scalar("SELECT COUNT(*) FROM memories WHERE status='ACTIVE' AND temperature IN ('PRIORITY_HOT','HOT')")
        cooling = scalar("SELECT COUNT(*) FROM memories WHERE status='ACTIVE' AND temperature IN ('WARM','COLD')")
        outdated = scalar("SELECT COUNT(*) FROM memories WHERE status='OUTDATED'")
        archived = scalar("SELECT COUNT(*) FROM memories WHERE status='ARCHIVED'")
        top = [dict(r) for r in run(
            "SELECT id, type, short, full, retrieval_count, last_used FROM memories "
            "WHERE status='ACTIVE' AND retrieval_count > 0 ORDER BY retrieval_count DESC LIMIT 5"
        ).fetchall()]

    temp_order = ["PRIORITY_HOT", "HOT", "WARM", "COLD"]
    by_temperature.sort(key=lambda r: temp_order.index(r["key"]) if r["key"] in temp_order else 99)

    growth = [{"date": (start + timedelta(days=i)).isoformat(),
               "count": created.get((start + timedelta(days=i)).isoformat(), 0)} for i in range(30)]

    data_dir = paths.data_dir(config)
    runs = _parse_maintenance_log(paths.logs_dir(config) / "maintenance.log", limit=1)

    return _json({
        "totals": {
            "total": total, "active": active, "archived": archived, "outdated": outdated,
            "recalls": recalls_total, "awaiting_compression": uncompressed,
        },
        "by_status": by_status,
        "by_type": by_type,
        "by_temperature": by_temperature,
        "by_agent": by_agent,
        "growth": growth,
        "workflow": [
            {"stage": "Captured", "count": active + outdated + archived,
             "detail": "Saved from your chats"},
            {"stage": "Compressed", "count": compressed,
             "detail": f"{uncompressed:,} waiting for a short summary"},
            {"stage": "Recalled", "count": recalled,
             "detail": f"{recalls_total:,} recalls in total"},
            {"stage": "Hot", "count": hot, "detail": "Recalled first"},
            {"stage": "Cooling", "count": cooling, "detail": "Warm or cold, recalled less"},
            {"stage": "Archived", "count": archived, "detail": f"+{outdated:,} replaced by newer facts"},
        ],
        "top_recalled": [
            {"id": t["id"], "type": t["type"], "text": (t["short"] or t["full"] or "")[:160],
             "retrieval_count": t["retrieval_count"], "last_used": t["last_used"]} for t in top
        ],
        "storage": {
            "data_dir": str(data_dir),
            "config_file": str(paths.config_path()),
            "database_bytes": _dir_size(paths.db_path(config)),
            "vector_index_bytes": _dir_size(paths.chromadb_dir(config)),
        },
        "privacy": api_v05.data_flow(config),
        "maintenance": {
            "next_run": _next_maintenance(config),
            "last_run": runs[0]["time"] if runs else None,
        },
    })


async def list_memories(request: Request) -> Response:
    q = request.query_params
    where, args = [], []
    for param, column in FILTER_COLUMNS.items():
        value = q.get(param)
        if param == "project" and value == GLOBAL_PROJECT:
            value, match_empty = "", True
        else:
            match_empty = False
        if value or match_empty:
            where.append(f"{column} = ?")
            args.append(value)
    search = (q.get("q") or "").strip()
    if search:
        like = f"%{search}%"
        where.append("(full LIKE ? OR short LIKE ? OR keywords LIKE ? OR id LIKE ?)")
        args += [like, like, like, like]
    order = SORTS.get(q.get("sort") or "updated", SORTS["updated"])
    try:
        limit = max(1, min(200, int(q.get("limit", 50))))
        offset = max(0, int(q.get("offset", 0)))
    except ValueError:
        return _error("limit and offset must be numbers", 400)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    with local_db.get_connection() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM memories {where_sql}", args).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM memories {where_sql} ORDER BY {order} LIMIT ? OFFSET ?",
            args + [limit, offset],
        ).fetchall()
        facets = {
            name: [r[0] for r in conn.execute(
                f"SELECT DISTINCT {column} FROM memories WHERE {column} IS NOT NULL ORDER BY 1"
            ).fetchall()]
            for name, column in FILTER_COLUMNS.items()
        }
    return _json({
        "total": total, "limit": limit, "offset": offset,
        "items": [_public_memory(dict(r)) for r in rows],
        "facets": facets,
    })


async def get_memory(request: Request) -> Response:
    mem = local_db.get_memory_by_id(request.path_params["id"])
    if not mem:
        return _error("Memory not found", 404)
    return _json(_public_memory(mem))


async def update_memory(request: Request) -> Response:
    memory_id = request.path_params["id"]
    mem = local_db.get_memory_by_id(memory_id)
    if not mem:
        return _error("Memory not found", 404)
    try:
        body = await request.json()
    except Exception:
        return _error("Body must be JSON", 400)
    if not isinstance(body, dict):
        return _error("Body must be a JSON object", 400)

    updates = {}
    for field in ("short", "full"):
        if field in body:
            value = body[field]
            if not isinstance(value, str) or len(value) > MAX_TEXT_LEN:
                return _error(f"'{field}' must be text up to {MAX_TEXT_LEN} characters", 400)
            if field == "full" and not value.strip():
                return _error("'full' can't be empty", 400)
            updates[field] = value
    if "importance" in body:
        try:
            importance = float(body["importance"])
        except (TypeError, ValueError):
            return _error("'importance' must be a number", 400)
        if not 0.0 <= importance <= 1.0:
            return _error("'importance' must be between 0 and 1", 400)
        from caretaker.memory.temperature_engine import assign_temperature
        updates["importance"] = importance
        updates["temperature"] = assign_temperature(importance, mem.get("temperature", "WARM"))
    if "project" in body:
        from caretaker.projects import normalise_project
        if not isinstance(body["project"], str):
            return _error("'project' must be text", 400)
        updates["project"] = normalise_project(body["project"])
    if "type" in body:
        if body["type"] not in MEMORY_TYPES:
            return _error(f"'type' must be one of {sorted(MEMORY_TYPES)}", 400)
        updates["type"] = body["type"]
    unknown = set(body) - {"short", "full", "importance", "project", "type"}
    if unknown:
        return _error(f"Unknown fields: {sorted(unknown)}", 400)
    if not updates:
        return _error("Nothing to update", 400)

    privacy_note = ""
    if "short" in updates or "full" in updates:
        from caretaker.privacy import load_policy, protect, sensitivity_of
        policy = load_policy()
        findings = []
        for field in ("full", "short"):
            text = updates.get(field, mem.get(field))
            if text:
                result = protect(text, policy=policy)
                findings.extend(result.findings)
                if field in updates:
                    updates[field] = result.text
        updates["sensitivity"] = sensitivity_of(findings) if findings else "none"
        updates["privacy_findings"] = json.dumps(findings) if findings else None
        updates["privacy_reviewed"] = 0
        from caretaker.privacy import ProtectResult
        privacy_note = ProtectResult("", findings).describe()

    if not local_db.update_memory_fields(memory_id, updates):
        return _error("Database update failed", 500)
    body = _public_memory(local_db.get_memory_by_id(memory_id))
    body["privacy_note"] = privacy_note
    return _json(body)


async def archive_memory(request: Request) -> Response:
    memory_id = request.path_params["id"]
    if not local_db.get_memory_by_id(memory_id):
        return _error("Memory not found", 404)
    if not local_db.archive_memory(memory_id):
        return _error("Database update failed", 500)
    return _json(_public_memory(local_db.get_memory_by_id(memory_id)))


async def restore_memory(request: Request) -> Response:
    memory_id = request.path_params["id"]
    if not local_db.get_memory_by_id(memory_id):
        return _error("Memory not found", 404)
    if not local_db.restore_memory(memory_id):
        return _error("Database update failed", 500)
    return _json(_public_memory(local_db.get_memory_by_id(memory_id)))


async def activity(request: Request) -> Response:
    config = paths.load_config()
    project = _project_param(request)
    with local_db.get_connection() as conn:
        def run(sql, args=()):
            q, a = _scoped(sql, args, project)
            return conn.execute(q, a)

        saves = [dict(r) for r in run(
            "SELECT id, type, status, source_agent, short, full, created_at FROM memories "
            "ORDER BY created_at DESC LIMIT 25"
        ).fetchall()]
        used = [dict(r) for r in run(
            "SELECT id, type, status, short, full, retrieval_count, last_used FROM memories "
            "WHERE last_used IS NOT NULL ORDER BY last_used DESC LIMIT 25"
        ).fetchall()]
    for row in saves + used:
        row["text"] = (row.pop("short", None) or row.pop("full", None) or "")[:200]
        row.pop("full", None)
    return _json({
        "recent_saves": saves,
        "recently_recalled": used,
        "maintenance_runs": _parse_maintenance_log(paths.logs_dir(config) / "maintenance.log"),
    })


# ── App ────────────────────────────────────────────────────────────────────────

class _TokenMiddleware:
    """Require the per-run token on every /api request."""

    def __init__(self, app, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/api/"):
            headers = dict(scope.get("headers") or [])
            supplied = headers.get(TOKEN_HEADER.encode(), b"").decode("latin-1")
            if not secrets.compare_digest(supplied, self.token):
                await _error("Missing or invalid dashboard token", 401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app(token: str, allowed_hosts=("127.0.0.1", "localhost")) -> Starlette:
    routes = [
        Route("/", index),
        Route("/static/{name}", static_file),
        Route("/api/overview", overview),
        Route("/api/activity", activity),
        Route("/api/memories", list_memories),
        Route("/api/memories/{id}", get_memory, methods=["GET"]),
        Route("/api/memories/{id}", update_memory, methods=["PATCH"]),
        Route("/api/memories/{id}/archive", archive_memory, methods=["POST"]),
        Route("/api/memories/{id}/restore", restore_memory, methods=["POST"]),
        *api_v05.routes(),
    ]
    return Starlette(
        routes=routes,
        middleware=[
            Middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts)),
            Middleware(_TokenMiddleware, token=token),
        ],
    )


def _free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("No free port available")


def run_dashboard(port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    """Start the dashboard on 127.0.0.1 and (optionally) open it in the browser."""
    import uvicorn

    local_db.run_migrations()
    token = secrets.token_urlsafe(32)
    port = _free_port(port)
    url = f"http://127.0.0.1:{port}/#token={token}"

    print(f"\n  Caretaker dashboard: {url}", flush=True)
    print(f"  Data folder: {paths.data_dir()}", flush=True)
    print("  Only this computer can open it. Press Ctrl+C to stop.\n", flush=True)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(create_app(token), host="127.0.0.1", port=port, log_level="warning")
