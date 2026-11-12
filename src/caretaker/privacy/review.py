"""
caretaker/privacy/review.py
Scan memories that are already stored (e.g. saved before 0.5), and the
actions behind the dashboard's privacy review queue.

Redacting stored text bumps `updated_at`, so the nightly reindex re-embeds the
cleaned summary and the old text leaves the search index. Backups and exports
made earlier still contain the original text — callers should say so.
"""

import json
from typing import Optional

from caretaker.privacy import load_policy, protect, sensitivity_of
from caretaker.storage import local_db


def _check(mem: dict, policy: dict, confirmed: bool = False):
    """Run the policy over a memory's text fields. Returns (updates, findings)."""
    updates, findings = {}, []
    for field in ("full", "short"):
        value = mem.get(field)
        if not value:
            continue
        result = protect(value, policy=policy, confirmed=confirmed)
        findings.extend(result.findings)
        if result.text != value:
            updates[field] = result.text
    return updates, findings


def scan_memories(apply: bool = False, include_reviewed: bool = False, policy: Optional[dict] = None) -> dict:
    """
    Check every stored memory against the privacy policy.
    apply=False: report only. apply=True: redact/mask in place and record findings.
    Memories the user marked "not sensitive" are skipped unless include_reviewed.
    """
    policy = policy or load_policy()
    sql = "SELECT * FROM memories" + ("" if include_reviewed else " WHERE privacy_reviewed = 0")
    with local_db.get_connection() as conn:
        rows = [dict(r) for r in conn.execute(sql).fetchall()]

    items, changed = [], 0
    for mem in rows:
        updates, findings = _check(mem, policy)
        sensitivity = sensitivity_of(findings) if findings else "none"
        # A memory redacted earlier keeps its 'redacted' state even when its
        # (already clean) text has nothing left to find.
        if not findings and mem.get("sensitivity") == "redacted":
            continue
        if findings:
            items.append({
                "id": mem["id"], "type": mem.get("type"), "status": mem.get("status"),
                "project": mem.get("project") or "", "findings": findings,
                "will_change": bool(updates), "sensitivity": sensitivity,
            })
        stored = (mem.get("sensitivity") or "none", mem.get("privacy_findings"))
        new = (sensitivity, json.dumps(findings) if findings else None)
        if apply and (updates or stored != new):
            local_db.update_memory_fields(mem["id"], {
                **updates, "sensitivity": sensitivity, "privacy_findings": new[1],
            }, touch_updated_at=bool(updates))
            changed += bool(updates)

    return {"scanned": len(rows), "with_findings": len(items), "changed": changed, "items": items}


def mark_not_sensitive(memory_id: str) -> bool:
    """User reviewed the memory and says it's fine: clear flags and skip it in future scans."""
    return local_db.update_memory_fields(memory_id, {
        "sensitivity": "none", "privacy_findings": None, "privacy_reviewed": 1,
    }, touch_updated_at=False)


def redact_memory(memory_id: str) -> Optional[dict]:
    """
    Remove everything the scan flagged in one memory (e.g. a flagged email
    address), not just what the policy redacts automatically. Topic flags
    (health, money, …) can't be cut out of a sentence — archive those instead.
    """
    mem = local_db.get_memory_by_id(memory_id)
    if not mem:
        return None
    policy = load_policy()
    strict = {**policy, **{c: "redact" for c in ("secret", "contact", "custom")}}
    if policy.get("financial") != "redact":
        strict["financial"] = "mask"
    updates, findings = _check(mem, strict, confirmed=True)   # the user asked: act on guesses too
    remaining = [f for f in findings if f["action"] == "flag"]     # topics / low-confidence guesses
    sensitivity = "redacted" if updates or mem.get("sensitivity") == "redacted" else sensitivity_of(remaining)
    local_db.update_memory_fields(memory_id, {
        **updates, "sensitivity": sensitivity,
        "privacy_findings": json.dumps(findings) if findings else None,
    }, touch_updated_at=bool(updates))
    return local_db.get_memory_by_id(memory_id)
