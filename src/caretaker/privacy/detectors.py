"""
caretaker/privacy/detectors.py
On-device detection of sensitive data in memory text. Nothing here makes a
network call: sending text to a cloud scanner would leak the very secrets we
are trying to protect.

Categories (each has a policy action, see policy.py):
  secret     API keys, tokens, private keys, passwords, credentials in URLs
  financial  payment cards (Luhn), IBAN (mod 97), US SSN, Aadhaar (Verhoeff), Indian PAN
  contact    email addresses, phone numbers
  personal   health / religion / sexuality / money topics (keyword based, flag only)
  custom     user-defined regex patterns from config

Detection is best-effort. Validators (Luhn, mod-97, Verhoeff) keep false
positives low for numbers; the generic "random-looking token" detector is
medium confidence and is only ever flagged, never redacted.
"""

import math
import re
from dataclasses import dataclass
from typing import List

HIGH, MEDIUM = "high", "medium"


@dataclass
class Finding:
    category: str        # secret | financial | contact | personal | custom
    kind: str            # e.g. "anthropic_api_key", "card_number"
    label: str           # human wording, e.g. "Anthropic API key"
    start: int           # span of the sensitive part within the text
    end: int
    confidence: str = HIGH

    def summary(self) -> dict:
        """What may be stored/shown: never the matched text itself."""
        return {"category": self.category, "kind": self.kind, "label": self.label, "confidence": self.confidence}


# ── Validators ─────────────────────────────────────────────────────────────────

def luhn_valid(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def iban_valid(iban: str) -> bool:
    s = iban[4:] + iban[:4]
    try:
        return int("".join(str(int(c, 36)) for c in s)) % 97 == 1
    except ValueError:
        return False


_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    return -sum((n / len(s)) * math.log2(n / len(s)) for n in counts.values())


# ── Pattern tables ─────────────────────────────────────────────────────────────
# (kind, label, regex, group) — `group` is the part to protect (0 = whole match).

SECRET_PATTERNS = [
    ("private_key", "private key",
     re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)"), 0),
    ("anthropic_api_key", "Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}"), 0),
    ("openai_api_key", "OpenAI API key", re.compile(r"\bsk-(?!ant-)(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}"), 0),
    ("github_token", "GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{22,})"), 0),
    ("aws_access_key", "AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), 0),
    ("google_api_key", "Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), 0),
    ("slack_token", "Slack token", re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}"), 0),
    ("stripe_key", "Stripe key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"), 0),
    ("jwt", "access token (JWT)",
     re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"), 0),
    ("url_credentials", "password in a URL",
     re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:([^\s@/]{3,})@", re.IGNORECASE), 1),
    ("password", "password",
     re.compile(
         r"\b(?:password|passwd|passcode|pwd|pin|secret(?:\s+key)?|api[\s_\-]?key|access[\s_\-]?token|token)"
         r"\s*(?:is|was|=|:)\s*[\"']?([^\s\"',;]{4,})",
         re.IGNORECASE), 1),
]

FINANCIAL_PATTERNS = [
    ("card_number", "card number", re.compile(r"\b(?:\d[ \-]?){12,18}\d\b"), 0),
    ("iban", "IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b"), 0),
    ("us_ssn", "US Social Security number", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), 0),
    ("aadhaar", "Aadhaar number", re.compile(r"\b[2-9]\d{3}[ \-]?\d{4}[ \-]?\d{4}\b"), 0),
    ("pan", "PAN", re.compile(r"\b[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z]\b"), 0),
]

CONTACT_PATTERNS = [
    ("email", "email address", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), 0),
    ("phone", "phone number",
     re.compile(r"(?<![\w+])(?:\+\d{1,3}[\s\-]?)?(?:\(\d{2,5}\)[\s\-]?|\d{2,5}[\s\-])?\d{3,5}[\s\-]?\d{4,5}(?!\w)"), 0),
]

PERSONAL_TOPICS = {
    "health": ["diagnosed", "diagnosis", "medication", "prescription", "therapy", "therapist", "depression",
               "anxiety disorder", "diabetes", "cancer", "hiv", "pregnant", "pregnancy", "surgery", "disability"],
    "beliefs": ["my religion", "i am religious", "my faith", "i pray", "political party i support", "i vote for"],
    "sexuality": ["my sexuality", "i am gay", "i am lesbian", "i am bisexual", "i am transgender", "coming out"],
    "money": ["my salary", "i earn", "my debt", "my loan", "credit score", "bank balance", "net worth"],
}
_TOPIC_RE = {
    topic: re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)
    for topic, words in PERSONAL_TOPICS.items()
}

_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_\-+/=])[A-Za-z0-9_\-+/=]{24,}(?![A-Za-z0-9_\-+/=])")
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


# ── Scanners ───────────────────────────────────────────────────────────────────

def _looks_like_password(value: str) -> bool:
    """A plain dictionary word after "password is" is usually prose, not a credential."""
    return any(c.isdigit() for c in value) or any(not c.isalnum() for c in value) or len(value) >= 12


def _span(m, group):
    return (m.start(group), m.end(group)) if group else (m.start(), m.end())


def _scan_secrets(text):
    out = []
    for kind, label, rx, group in SECRET_PATTERNS:
        for m in rx.finditer(text):
            s, e = _span(m, group)
            if kind == "password" and not _looks_like_password(text[s:e]):
                continue   # "the token is expired" is not a secret
            out.append(Finding("secret", kind, label, s, e, HIGH))
    # Random-looking long tokens: medium confidence (flag only). Skip UUIDs and
    # plain hex (commit SHAs, hashes) — common in developer conversations.
    for m in _TOKEN_RE.finditer(text):
        tok = m.group()
        if _UUID_RE.match(tok) or _HEX_RE.match(tok) or tok.isalpha() or tok.isdigit():
            continue
        classes = sum([any(c.islower() for c in tok), any(c.isupper() for c in tok), any(c.isdigit() for c in tok)])
        if classes >= 3 and shannon_entropy(tok) >= 3.8:
            out.append(Finding("secret", "possible_secret", "possible secret token", m.start(), m.end(), MEDIUM))
    return out


def _scan_financial(text):
    out = []
    for kind, label, rx, group in FINANCIAL_PATTERNS:
        for m in rx.finditer(text):
            raw = m.group()
            digits = re.sub(r"\D", "", raw)
            if kind == "card_number" and not (13 <= len(digits) <= 19 and digits[0] in "3456" and luhn_valid(digits)):
                continue
            if kind == "iban" and not iban_valid(raw.replace(" ", "")):
                continue
            if kind == "aadhaar" and not (len(digits) == 12 and verhoeff_valid(digits)):
                continue
            out.append(Finding("financial", kind, label, m.start(), m.end(), HIGH))
    return out


def _scan_contact(text):
    out = []
    for kind, label, rx, group in CONTACT_PATTERNS:
        for m in rx.finditer(text):
            if kind == "phone":
                digits = re.sub(r"\D", "", m.group())
                if not 10 <= len(digits) <= 15:
                    continue
            out.append(Finding("contact", kind, label, m.start(), m.end(), HIGH if kind == "email" else MEDIUM))
    return out


def _scan_personal(text):
    out = []
    for topic, rx in _TOPIC_RE.items():
        m = rx.search(text)
        if m:
            out.append(Finding("personal", f"{topic}_topic", f"{topic} topic", m.start(), m.end(), MEDIUM))
    return out


def _scan_custom(text, custom_patterns):
    out = []
    for item in custom_patterns or []:
        try:
            rx = re.compile(item["pattern"])
        except (re.error, KeyError, TypeError):
            continue
        label = str(item.get("label") or "custom pattern")
        for m in rx.finditer(text):
            if m.end() > m.start():
                out.append(Finding("custom", "custom:" + label, label, m.start(), m.end(), HIGH))
    return out


_PRIORITY = {"secret": 0, "custom": 1, "financial": 2, "contact": 3, "personal": 4}


def scan(text: str, custom_patterns=None, allowlist=None) -> List[Finding]:
    """Find sensitive spans. Overlapping findings keep the most important one."""
    if not text:
        return []
    found = (_scan_secrets(text) + _scan_custom(text, custom_patterns) + _scan_financial(text)
             + _scan_contact(text) + _scan_personal(text))
    allow = {a for a in (allowlist or []) if isinstance(a, str) and a}
    if allow:
        found = [f for f in found if text[f.start:f.end] not in allow]
    # Most important first, then longest; drop anything overlapping a kept span.
    found.sort(key=lambda f: (_PRIORITY.get(f.category, 9), f.confidence != HIGH, -(f.end - f.start), f.start))
    kept = []
    for f in found:
        if f.category == "personal" or all(f.end <= k.start or f.start >= k.end for k in kept if k.category != "personal"):
            kept.append(f)
    return sorted(kept, key=lambda f: f.start)
