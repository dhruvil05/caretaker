"""
caretaker/dashboard/api_v05.py
0.5 dashboard API: projects, "what will the agent see" preview, privacy page
and review queue, replaced/stale review with undo, bulk actions, setup page.

Same protection as the rest of the dashboard (token + Host check + CSP, in
app.py). Responses never contain secrets found by the privacy scanner: only
labels/counts, and the stored text — which has already been redacted.
"""

import json
import re
from datetime import datetime, timedelta, timezone

from starlette.requests import Request
from starlette.routing import Route

from caretaker import paths
from caretaker.privacy import data_flow
from caretaker.storage import local_db

STALE_DAYS = 60
MAX_BULK = 500


def _app():
    from caretaker.dashboard import app   # late import: app.py imports this module
    return app


def _json(data, status=200):
    return _app()._json(data, status)


def _error(message, status):
    return _app()._error(message, status)


async def _body(request: Request):
    try:
        body = await request.json()
    except Exception:
        return None
    return body if isinstance(body, dict) else None


# ── Projects ───────────────────────────────────────────────────────────────────

async def projects(request: Request):
    with local_db.get_connection() as conn:
        rows = conn.execute(
            "SELECT project, COUNT(*) AS n FROM memories WHERE status = 'ACTIVE' GROUP BY project ORDER BY n DESC"
        ).fetchall()
    return _json([{"project": r["project"] or "", "count": r["n"]} for r in rows])


async def summary(request: Request):
    """Small counts for the navigation: what needs the user's attention."""
    stale_before = (datetime.now(timezone.utc) - timedelta(days=STALE_DAYS)).isoformat()
    with local_db.get_connection() as conn:
        one = lambda sql, *a: conn.execute(sql, a).fetchone()[0]
        return _json({
            "privacy_review": one("SELECT COUNT(*) FROM memories WHERE sensitivity = 'flagged' "
                                  "AND privacy_reviewed = 0 AND status != 'ARCHIVED'"),
            "replaced": one("SELECT COUNT(*) FROM memories WHERE status = 'OUTDATED'"),
            "stale": one("SELECT COUNT(*) FROM memories WHERE status = 'ACTIVE' "
                         "AND COALESCE(last_used, created_at) < ?", stale_before),
            "data_flow": data_flow(paths.load_config()),
        })


# ── Preview: what an agent would receive ───────────────────────────────────────

async def preview(request: Request):
    message = (request.query_params.get("message") or "").strip()
    if not message:
        return _error("Type a message to preview", 400)
    if len(message) > 4000:
        return _error("Message is too long to preview", 400)
    agent = request.query_params.get("agent") or "claude"
    from caretaker.projects import normalise_project
    project = normalise_project(request.query_params.get("project") or "")

    from caretaker.mcp_server.agent_adapter import adapt
    from caretaker.mcp_server.injector import build_whisper
    from caretaker.privacy import protect
    from caretaker.retrieval.retrieval_engine import retrieve_context

    # Same pipeline as caretaker_get_context, but dry-run: nothing is counted as
    # recalled. The dashboard never opens the vector index (it isn't safe to share
    # with the running MCP server), so this preview uses keyword search.
    context = retrieve_context(message, agent, project=project, dry_run=True)
    text = protect(adapt(build_whisper(context), agent)).text
    picked = []
    for m in context.get("relevant", []):
        picked.append({
            "id": m.get("id"), "type": m.get("type"), "project": m.get("project") or "",
            "text": (m.get("short") or m.get("full") or "")[:200],
            "reason": "general background (nothing matched)" if m.get("_reason") == "background"
                      else f"keyword match (score {m.get('match_score', 0):.2f})",
        })
    import importlib.util
    semantic_installed = all(importlib.util.find_spec(n) for n in ("chromadb", "sentence_transformers"))
    return _json({
        "text": text,
        "approx_tokens": max(1, len(text) // 4),
        "budget": context.get("budget"), "level": context.get("level"),
        "memories": picked,
        "note": ("Agents also use semantic search, so they may get some different memories than this keyword-based preview."
                 if semantic_installed else ""),
    })


# ── Privacy ────────────────────────────────────────────────────────────────────

async def privacy_overview(request: Request):
    from caretaker.privacy import load_policy
    config = paths.load_config()
    policy = load_policy(config)
    with local_db.get_connection() as conn:
        counts = {r["sensitivity"]: r["n"] for r in conn.execute(
            "SELECT sensitivity, COUNT(*) AS n FROM memories WHERE status != 'ARCHIVED' GROUP BY sensitivity")}
        queue = [dict(r) for r in conn.execute(
            "SELECT * FROM memories WHERE sensitivity = 'flagged' AND privacy_reviewed = 0 "
            "AND status != 'ARCHIVED' ORDER BY updated_at DESC LIMIT 200")]
        redacted = [dict(r) for r in conn.execute(
            "SELECT * FROM memories WHERE sensitivity = 'redacted' ORDER BY updated_at DESC LIMIT 50")]
    return _json({
        "policy": {k: policy[k] for k in ("secret", "financial", "contact", "personal", "custom")},
        "custom_patterns": policy["custom_patterns"],
        "allowlist": policy["allowlist"],
        "counts": {"flagged": counts.get("flagged", 0), "redacted": counts.get("redacted", 0),
                   "none": counts.get("none", 0)},
        "queue": [_app()._public_memory(m) for m in queue],
        "redacted": [_app()._public_memory(m) for m in redacted],
        "leaves_device": data_flow(config),
        "data_dir": str(paths.data_dir(config)),
    })


async def privacy_scan(request: Request):
    body = await _body(request) or {}
    from caretaker.privacy.review import scan_memories
    report = scan_memories(apply=bool(body.get("apply")))
    # Count memories, not occurrences: a key in both the summary and the full
    # text of one memory is one affected memory.
    by_kind = {}
    for item in report["items"]:
        for key in {f"{f['label']}|{f['action']}" for f in item["findings"]}:
            by_kind[key] = by_kind.get(key, 0) + 1
    return _json({
        "scanned": report["scanned"], "with_findings": report["with_findings"], "changed": report["changed"],
        "will_change": sum(1 for i in report["items"] if i["will_change"]),
        "by_kind": [{"label": k.split("|")[0], "action": k.split("|")[1], "count": n} for k, n in sorted(by_kind.items())],
        "applied": bool(body.get("apply")),
    })


async def memory_privacy(request: Request):
    memory_id = request.path_params["id"]
    if not local_db.get_memory_by_id(memory_id):
        return _error("Memory not found", 404)
    body = await _body(request) or {}
    from caretaker.privacy.review import mark_not_sensitive, redact_memory
    action = body.get("action")
    if action == "not_sensitive":
        mark_not_sensitive(memory_id)
    elif action == "redact":
        redact_memory(memory_id)
    else:
        return _error("action must be 'not_sensitive' or 'redact'", 400)
    return _json(_app()._public_memory(local_db.get_memory_by_id(memory_id)))


async def privacy_policy(request: Request):
    body = await _body(request)
    if body is None:
        return _error("Body must be a JSON object", 400)
    from caretaker.privacy import ACTIONS, DEFAULT_POLICY
    new = {}
    for category in DEFAULT_POLICY:
        if category in body:
            if body[category] not in ACTIONS:
                return _error(f"'{category}' must be one of {list(ACTIONS)}", 400)
            new[category] = body[category]
    if "custom_patterns" in body:
        patterns = body["custom_patterns"]
        if not isinstance(patterns, list) or len(patterns) > 100:
            return _error("custom_patterns must be a list (max 100)", 400)
        clean = []
        for item in patterns:
            if not isinstance(item, dict) or not isinstance(item.get("pattern"), str) or not item["pattern"]:
                return _error("each custom pattern needs a 'pattern'", 400)
            try:
                re.compile(item["pattern"])
            except re.error as e:
                return _error(f"invalid pattern {item['pattern']!r}: {e}", 400)
            clean.append({"pattern": item["pattern"][:500], "label": str(item.get("label") or "custom pattern")[:60]})
        new["custom_patterns"] = clean
    if "allowlist" in body:
        allow = body["allowlist"]
        if not isinstance(allow, list) or not all(isinstance(a, str) for a in allow) or len(allow) > 500:
            return _error("allowlist must be a list of text (max 500)", 400)
        new["allowlist"] = [a[:500] for a in allow if a]
    config = paths.read_config_file()
    config["privacy"] = {**(config.get("privacy") or {}), **new}
    paths.write_config_file(config)
    return await privacy_overview(request)


# ── Review: replaced and stale memories ────────────────────────────────────────

async def review(request: Request):
    stale_before = (datetime.now(timezone.utc) - timedelta(days=STALE_DAYS)).isoformat()
    project = _app()._project_param(request)
    scope_sql, scope_args = ("", []) if project is None else (" AND o.project = ?", [project])
    with local_db.get_connection() as conn:
        replaced = [dict(r) for r in conn.execute(
            "SELECT o.*, n.id AS by_id, COALESCE(n.short, n.full) AS by_text, n.status AS by_status "
            "FROM memories o LEFT JOIN memories n ON n.id = o.superseded_by "
            f"WHERE o.status = 'OUTDATED'{scope_sql} ORDER BY o.updated_at DESC LIMIT 200", scope_args)]
        stale = [dict(r) for r in conn.execute(
            "SELECT * FROM memories o WHERE o.status = 'ACTIVE' "
            f"AND COALESCE(o.last_used, o.created_at) < ?{scope_sql} "
            "ORDER BY COALESCE(o.last_used, o.created_at) LIMIT 200",
            [stale_before, *scope_args])]
    out_replaced = []
    for r in replaced:
        by = {"id": r.pop("by_id"), "text": (r.pop("by_text") or "")[:200], "status": r.pop("by_status")}
        out_replaced.append({**_app()._public_memory(r), "replaced_by": by if by["id"] else None})
    return _json({
        "replaced": out_replaced,
        "stale": [_app()._public_memory(r) for r in stale],
        "stale_days": STALE_DAYS,
    })


# ── Bulk actions ───────────────────────────────────────────────────────────────

async def bulk(request: Request):
    body = await _body(request)
    if body is None:
        return _error("Body must be a JSON object", 400)
    ids = body.get("ids")
    action = body.get("action")
    if not isinstance(ids, list) or not ids or len(ids) > MAX_BULK or not all(isinstance(i, str) for i in ids):
        return _error(f"ids must be a list of 1–{MAX_BULK} memory ids", 400)
    value = body.get("value")
    if action == "set_project":
        from caretaker.projects import normalise_project
        if not isinstance(value, str):
            return _error("value must be a project name ('' for global)", 400)
        value = normalise_project(value)
    elif action == "set_type":
        if value not in _app().MEMORY_TYPES:
            return _error(f"value must be one of {sorted(_app().MEMORY_TYPES)}", 400)
    elif action not in ("archive", "restore", "not_sensitive"):
        return _error("action must be archive, restore, not_sensitive, set_project or set_type", 400)

    from caretaker.privacy.review import mark_not_sensitive
    done = 0
    for memory_id in ids:
        if not local_db.get_memory_by_id(memory_id):
            continue
        if action == "archive":
            ok = local_db.archive_memory(memory_id)
        elif action == "restore":
            ok = local_db.restore_memory(memory_id)
        elif action == "not_sensitive":
            ok = mark_not_sensitive(memory_id)
        elif action == "set_project":
            ok = local_db.update_memory_fields(memory_id, {"project": value})
        else:
            ok = local_db.update_memory_fields(memory_id, {"type": value})
        done += bool(ok)
    return _json({"action": action, "requested": len(ids), "updated": done})


# ── Setup ──────────────────────────────────────────────────────────────────────

async def setup(request: Request):
    from caretaker.doctor import client_snippets, detect_clients, run_checks
    return _json({
        "checks": run_checks(),
        "clients": detect_clients(),
        "snippets": [{**s, "json": json.dumps(s["json"], indent=2)} if "json" in s else s for s in client_snippets()],
    })


def routes():
    return [
        Route("/api/projects", projects),
        Route("/api/summary", summary),
        Route("/api/preview", preview),
        Route("/api/privacy", privacy_overview),
        Route("/api/privacy/scan", privacy_scan, methods=["POST"]),
        Route("/api/privacy/policy", privacy_policy, methods=["PUT"]),
        Route("/api/memories/bulk", bulk, methods=["POST"]),
        Route("/api/memories/{id}/privacy", memory_privacy, methods=["POST"]),
        Route("/api/review", review),
        Route("/api/setup", setup),
    ]
