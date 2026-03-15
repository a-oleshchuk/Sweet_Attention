from __future__ import annotations

import re

from .repositories import normalize_text


_FRAUD_SIGNAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(skip|bypass|avoid)\b[^.!?\n]{0,40}\b(verification|identity check|security check|approval)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(without telling|without them knowing|without the owner knowing|without approval|without permission)\b[^.!?\n]{0,60}\b(account|guardian|owner|parent|cardholder)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(reset|change|move|send|give)\b[^.!?\n]{0,80}\b(password|login|verification code|email)\b[^.!?\n]{0,80}\b(to me|my email)\b[^.!?\n]{0,80}\b(without telling|without them knowing|without permission)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(someone else's|someone elses|another person's|another persons|other guardian's|other guardians|my ex's|my exs)\b[^.!?\n]{0,60}\b(card|account|email|login|invoice|receipt)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(edit|alter|change|backdate|delete|hide|remove)\b[^.!?\n]{0,60}\b(invoice|receipt|payment|charge|billing history|transaction history)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(refund|chargeback|reverse)\b[^.!?\n]{0,80}\b(keep|still keep|without losing)\b[^.!?\n]{0,80}\b(access|subscription|lessons|course)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(access|open|get into|take over)\b[^.!?\n]{0,60}\b(another account|someone else's account|someone elses account|other guardian's account)\b",
        re.IGNORECASE,
    ),
)


def detect_possible_user_fraud(text: str) -> list[str]:
    if not text.strip():
        return []

    normalized = normalize_text(text)
    sentences = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+|\n+", text) if chunk.strip()]
    evidence: list[str] = []

    for sentence in sentences or [text]:
        sentence_normalized = normalize_text(sentence)
        if any(pattern.search(sentence_normalized) for pattern in _FRAUD_SIGNAL_PATTERNS):
            evidence.append(sentence.strip())

    if evidence:
        return list(dict.fromkeys(evidence))
    if any(pattern.search(normalized) for pattern in _FRAUD_SIGNAL_PATTERNS):
        return [text.strip()]
    return []
