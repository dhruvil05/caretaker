"""
tests/phase3/test_privacy.py
Sensitive-data detection and policy. Test values are fake but pass the same
checksums real ones do (Luhn, mod-97, Verhoeff).
"""

import pytest

from caretaker.privacy import DEFAULT_POLICY, load_policy, protect
from caretaker.privacy.detectors import iban_valid, luhn_valid, scan, verhoeff_valid

# Fake credentials, built at runtime so secret scanners don't flag this file.
ANTHROPIC_KEY = "sk-ant-" + "api03-" + "a1B2c3D4e5F6g7H8i9J0kLmNoPqRsTuV"
GITHUB_TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
CARD = "4242 4242 4242 4242"          # Luhn-valid test card
IBAN = "GB82 WEST 1234 5698 7654 32"   # mod-97 valid example IBAN
AADHAAR = "2341 2341 2346"             # Verhoeff-valid test number


def kinds(text, **kw):
    return [f.kind for f in scan(text, **kw)]


# ── Validators ─────────────────────────────────────────────────────────────────

def test_checksum_validators():
    assert luhn_valid("4242424242424242") and not luhn_valid("4242424242424241")
    assert iban_valid("GB82WEST12345698765432") and not iban_valid("GB82WEST12345698765433")
    assert verhoeff_valid("234123412346") and not verhoeff_valid("234123412345")


# ── Detection ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, kind", [
    (f"my anthropic key is {ANTHROPIC_KEY}", "anthropic_api_key"),
    (f"use {GITHUB_TOKEN} for the CI", "github_token"),
    (f"aws {AWS_KEY} in prod", "aws_access_key"),
    ("connect to postgres://admin:S3cretPw@db.example.com/app", "url_credentials"),
    ("my wifi password is hunter2!", "password"),
    ("-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----", "private_key"),
    (f"pay with {CARD}", "card_number"),
    (f"salary goes to {IBAN}", "iban"),
    (f"my aadhaar is {AADHAAR}", "aadhaar"),
    ("PAN ABCPE1234F for tax", "pan"),
    ("ssn 123-45-6789", "us_ssn"),
    ("email me at dhruvil@example.com", "email"),
    ("call +91 98765 43210 tomorrow", "phone"),
    ("I was diagnosed with diabetes last year", "health_topic"),
])
def test_detects(text, kind):
    assert kind in kinds(text)


@pytest.mark.parametrize("text", [
    "the token is expired, refresh it",                        # prose, not a password
    "commit 9fceb02d0ae598e95dc970b74767f19372d61af8 fixed it",  # git SHA
    "memory id 3b7d7c04-6b92-424e-913f-0dce7560a0f9",          # UUID
    "released on 2026-11-10 at 10:42",                         # date / time
    "order 4242 4242 4242 4241 shipped",                       # fails Luhn
    "the test suite has 287 passed and 5 skipped",
    "I prefer Python over JavaScript for backends",
])
def test_no_false_alarms(text):
    assert [f for f in scan(text) if f.category in ("secret", "financial")] == []


def test_random_looking_token_is_only_medium_confidence():
    findings = scan("token aB3dE5gH7jK9mN1pQ3sT5vX7zA9cD1fG3 in config")
    assert [(f.kind, f.confidence) for f in findings] == [("possible_secret", "medium")]


def test_custom_patterns_and_allowlist():
    assert "custom:client id" in kinds("ticket ACME-4411 is urgent", custom_patterns=[{"pattern": r"ACME-\d+", "label": "client id"}])
    assert kinds("write to team@example.com", allowlist=["team@example.com"]) == []
    # broken user regex is ignored, not fatal
    assert kinds("hello", custom_patterns=[{"pattern": "([", "label": "bad"}]) == []


# ── Policy ─────────────────────────────────────────────────────────────────────

def test_default_policy_redacts_secrets_masks_ids_flags_contact():
    r = protect(f"key {ANTHROPIC_KEY}, card {CARD}, mail me at a@b.co", policy=load_policy({}))
    assert ANTHROPIC_KEY not in r.text and "[REDACTED: Anthropic API key]" in r.text
    assert CARD not in r.text and "[card number ending 4242]" in r.text
    assert "a@b.co" in r.text                                   # contact kept…
    assert {"email": "flag"}.items() <= {f["kind"]: f["action"] for f in r.findings}.items()  # …but flagged
    assert r.sensitivity == "redacted" and r.changed
    assert "1 Anthropic API key redacted" in r.describe()
    assert all(ANTHROPIC_KEY not in str(f) for f in r.findings)  # findings never hold the secret


def test_rest_of_the_message_is_kept():
    r = protect(f"I use {ANTHROPIC_KEY} for Orbit, which runs on FastAPI", policy=load_policy({}))
    assert r.text == "I use [REDACTED: Anthropic API key] for Orbit, which runs on FastAPI"


def test_flag_only_and_clean_text():
    r = protect("my number is +91 98765 43210", policy=load_policy({}))
    assert r.text == "my number is +91 98765 43210" and r.sensitivity == "flagged"
    clean = protect("I prefer Python", policy=load_policy({}))
    assert clean.findings == [] and clean.sensitivity == "none"


def test_medium_confidence_is_never_rewritten_even_if_policy_redacts():
    tok = "aB3dE5gH7jK9mN1pQ3sT5vX7zA9cD1fG3"
    r = protect(f"token {tok}", policy=load_policy({}))
    assert tok in r.text and r.findings[0]["action"] == "flag"


def test_user_policy_overrides_and_invalid_values():
    pol = load_policy({"privacy": {"contact": "redact", "financial": "nonsense"}})
    assert pol["contact"] == "redact" and pol["financial"] == DEFAULT_POLICY["financial"]
    r = protect("mail a@b.co", policy=pol)
    assert r.text == "mail [REDACTED: email address]"
    allow_all = load_policy({"privacy": {"secret": "allow"}})
    assert ANTHROPIC_KEY in protect(ANTHROPIC_KEY, policy=allow_all).text
