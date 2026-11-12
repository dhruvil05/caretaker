"""
caretaker/privacy
Sensitive-data protection, applied on-device before text is stored,
summarised (Haiku), indexed, or recalled into an agent's prompt.

Policy (config.json → "privacy", every key optional):
    {
      "secret":    "redact",   # API keys, tokens, passwords, private keys
      "financial": "mask",     # cards, IBAN, SSN, Aadhaar, PAN -> "[card number ending 4242]"
      "contact":   "flag",     # email, phone: kept, marked for review in the dashboard
      "personal":  "flag",     # health / beliefs / sexuality / money topics
      "custom":    "redact",   # your own patterns
      "custom_patterns": [{"pattern": "ACME-\\\\d+", "label": "client id"}],
      "allowlist": ["exact text that is not sensitive"]
    }
Actions: redact (remove, keep a label) | mask (keep last 4) | flag (keep, mark
for review) | allow (ignore). Medium-confidence secrets (random-looking tokens)
are only ever flagged.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from caretaker.privacy.detectors import HIGH, Finding, scan

ACTIONS = ("redact", "mask", "flag", "allow")
DEFAULT_POLICY = {
    "secret": "redact",
    "financial": "mask",
    "contact": "flag",
    "personal": "flag",
    "custom": "redact",
}

__all__ = ["ACTIONS", "DEFAULT_POLICY", "ProtectResult", "data_flow", "load_policy", "protect", "scan"]


def data_flow(config: dict) -> dict:
    """What leaves this device. Mirrors the code that actually sends data:
    the compressor uses the API only with a non-local model AND a key, and cloud
    sync runs only with a Supabase URL, key AND an encryption key."""
    compression = str(config.get("compression_model") or "local")
    uses_api = compression != "local" and bool(str(config.get("anthropic_api_key") or "").strip())
    supabase = bool(config.get("supabase_url") and config.get("supabase_key"))
    return {
        "compression": compression,
        "compression_on_device": not uses_api,
        "cloud_backup": supabase and bool(config.get("encrypt_key")),
        "cloud_missing_key": supabase and not config.get("encrypt_key"),
    }


def load_policy(config: Optional[dict] = None) -> dict:
    """Effective policy: defaults overridden by config['privacy'] (invalid values ignored)."""
    if config is None:
        from caretaker.paths import load_config
        config = load_config()
    user = config.get("privacy") or {}
    policy = dict(DEFAULT_POLICY)
    for category in DEFAULT_POLICY:
        if user.get(category) in ACTIONS:
            policy[category] = user[category]
    policy["custom_patterns"] = [p for p in (user.get("custom_patterns") or []) if isinstance(p, dict)]
    policy["allowlist"] = [a for a in (user.get("allowlist") or []) if isinstance(a, str)]
    return policy


@dataclass
class ProtectResult:
    text: str
    findings: List[dict] = field(default_factory=list)   # summaries + "action"; never the matched text

    @property
    def sensitivity(self) -> str:
        actions = {f["action"] for f in self.findings}
        if actions & {"redact", "mask"}:
            return "redacted"
        if "flag" in actions:
            return "flagged"
        return "none"

    @property
    def changed(self) -> bool:
        return any(f["action"] in ("redact", "mask") for f in self.findings)

    def describe(self) -> str:
        """Short human summary, e.g. '1 Anthropic API key redacted, 1 email address flagged'."""
        counts = {}
        for f in self.findings:
            verb = {"redact": "redacted", "mask": "masked", "flag": "flagged for review"}[f["action"]]
            key = (f["label"], verb)
            counts[key] = counts.get(key, 0) + 1
        return ", ".join(f"{n} {label} {verb}" for (label, verb), n in counts.items())


def _action_for(f: Finding, policy: dict, confirmed: bool = False) -> str:
    action = policy.get(f.category, "flag")
    if f.confidence != HIGH and action in ("redact", "mask") and not confirmed:
        return "flag"      # never rewrite text on a guess — unless the user asked to
    return action


def _replacement(text: str, f: Finding, action: str) -> str:
    if action == "redact":
        return f"[REDACTED: {f.label}]"
    tail = "".join(ch for ch in text[f.start:f.end] if ch.isalnum())[-4:]
    return f"[{f.label} ending {tail}]"


def protect(text: Optional[str], config: Optional[dict] = None, policy: Optional[dict] = None,
            confirmed: bool = False) -> ProtectResult:
    """
    Scan `text` and apply the policy. Returns the safe text plus what was done.
    confirmed=True: the user explicitly asked to clean this text, so medium-
    confidence findings (phone numbers, random-looking tokens) are acted on too.
    """
    if not text:
        return ProtectResult(text or "")
    policy = policy or load_policy(config)
    found = scan(text, policy.get("custom_patterns"), policy.get("allowlist"))

    applied = []
    out = text
    # Rewrite from the end so earlier offsets stay valid. Topic flags never rewrite.
    for f in sorted(found, key=lambda f: f.start, reverse=True):
        action = _action_for(f, policy, confirmed)
        if action == "allow":
            continue
        if action in ("redact", "mask") and f.category != "personal":
            out = out[:f.start] + _replacement(text, f, action) + out[f.end:]
        applied.append({**f.summary(), "action": action})
    applied.reverse()
    return ProtectResult(out, applied)


def merge(results: List[ProtectResult]) -> List[dict]:
    """Combine findings from several fields (e.g. short + full) of one memory."""
    combined = []
    for r in results:
        combined.extend(r.findings)
    return combined


def sensitivity_of(findings: List[dict]) -> str:
    return ProtectResult("", findings).sensitivity
