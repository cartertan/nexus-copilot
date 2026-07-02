import re

_COMPLEXITY_KEYWORDS = [
    "migrate", "migration", "architecture", "design", "enterprise",
    "compliance", "regulatory", "crypto-agile", "pqc", "post-quantum",
    "multi-domain", "zero-trust", "trade-off", "compare", "hybrid",
]

_LOCAL_OVERRIDE_TERMS = [
    "renew", "renewal", "revoke", "revocation",
    "password", "secret", "credential", "private key",
]


def score_complexity(question: str) -> int:
    score = 20
    q_lower = question.lower()

    if len(question) > 150:
        score += 15

    for kw in _COMPLEXITY_KEYWORDS:
        if kw in q_lower:
            score += 10

    if "?" in question and len(question) > 200:
        score += 10

    return min(score, 100)


def should_use_cloud(question: str, force: bool = False) -> tuple[bool, int]:
    q_lower = question.lower()
    for term in _LOCAL_OVERRIDE_TERMS:
        if term in q_lower:
            return (False, 0)

    if force:
        return (True, 100)

    score = score_complexity(question)
    if score > 60:
        return (True, score)
    return (False, score)
