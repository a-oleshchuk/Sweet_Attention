from __future__ import annotations

import re


_EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d[\d().\-\s]{6,}\d|\(\d{3}\)\s*\d{3}[-.\s]?\d{4})(?!\w)"
)
_EXTERNAL_REF_PATTERN = re.compile(r"\bext_[a-z0-9_]+\b", re.IGNORECASE)
_USERNAME_PATTERN = re.compile(r"\b[a-z0-9._-]+@student\.[a-z0-9.-]+\b", re.IGNORECASE)

_LABELED_IDENTIFIER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\baccount(?:\s+id)?\b\s*(?:is|=|:)?\s*(?:ACC\d+)\b", re.IGNORECASE),
        "your account",
    ),
    (
        re.compile(r"\bguardian(?:\s+id)?\b\s*(?:is|=|:)?\s*(?:G\d+)\b", re.IGNORECASE),
        "your profile",
    ),
    (
        re.compile(r"\bstudent(?:\s+id)?\b\s*(?:is|=|:)?\s*(?:S\d+)\b", re.IGNORECASE),
        "your child's profile",
    ),
    (
        re.compile(r"\bsubscription(?:\s+id)?\b\s*(?:is|=|:)?\s*(?:SUB\d+)\b", re.IGNORECASE),
        "the subscription",
    ),
    (
        re.compile(r"\bpayment(?:\s+id)?\b\s*(?:is|=|:)?\s*(?:PAY\d+)\b", re.IGNORECASE),
        "the payment",
    ),
    (
        re.compile(r"\binvoice(?:\s+id)?\b\s*(?:is|=|:)?\s*(?:INV\d+)\b", re.IGNORECASE),
        "the invoice",
    ),
    (
        re.compile(r"\buser\s+access(?:\s+id)?\b\s*(?:is|=|:)?\s*(?:UA\d+)\b", re.IGNORECASE),
        "the access record",
    ),
    (
        re.compile(r"\benrollment(?:\s+id)?\b\s*(?:is|=|:)?\s*(?:ENR\d+)\b", re.IGNORECASE),
        "the enrollment",
    ),
    (
        re.compile(r"\blesson(?:\s+id)?\b\s*(?:is|=|:)?\s*(?:LES\d+)\b", re.IGNORECASE),
        "the lesson",
    ),
    (
        re.compile(r"\bcourse(?:\s+id)?\b\s*(?:is|=|:)?\s*(?:COURSE\d+)\b", re.IGNORECASE),
        "the course",
    ),
]

_TOKEN_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bACC\d+\b", re.IGNORECASE), "your account"),
    (re.compile(r"\bG\d+\b", re.IGNORECASE), "your profile"),
    (re.compile(r"\bS\d+\b", re.IGNORECASE), "your child's profile"),
    (re.compile(r"\bSUB\d+\b", re.IGNORECASE), "the subscription"),
    (re.compile(r"\bPAY\d+\b", re.IGNORECASE), "the payment"),
    (re.compile(r"\bINV\d+\b", re.IGNORECASE), "the invoice"),
    (re.compile(r"\bUA\d+\b", re.IGNORECASE), "the access record"),
    (re.compile(r"\bENR\d+\b", re.IGNORECASE), "the enrollment"),
    (re.compile(r"\bLES\d+\b", re.IGNORECASE), "the lesson"),
    (re.compile(r"\bCOURSE\d+\b", re.IGNORECASE), "the course"),
]


def sanitize_user_reply(text: str) -> str:
    sanitized = text

    for pattern, replacement in _LABELED_IDENTIFIER_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)

    sanitized = _USERNAME_PATTERN.sub("the student login", sanitized)
    sanitized = _EMAIL_PATTERN.sub("the email address on file", sanitized)
    sanitized = _PHONE_PATTERN.sub("the phone number on file", sanitized)
    sanitized = _EXTERNAL_REF_PATTERN.sub("the external billing reference", sanitized)

    for pattern, replacement in _TOKEN_REPLACEMENTS:
        sanitized = pattern.sub(replacement, sanitized)

    sanitized = re.sub(r"\b(account|guardian|student|subscription|payment|invoice|username|email|phone)\s+ID\b", r"\1", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\(\s*,\s*", "(", sanitized)
    sanitized = re.sub(r"\s{2,}", " ", sanitized)
    sanitized = re.sub(r"\s+([,.)])", r"\1", sanitized)
    sanitized = re.sub(r"([(])\s+", r"\1", sanitized)
    return sanitized.strip()
