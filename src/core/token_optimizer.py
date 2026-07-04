import re

# Signals a question is comparing options / weighing trade-offs — moderate
# complexity bump.
_COMPARISON_KEYWORDS = [
    "compare", " vs ", "versus", "trade-off", "tradeoff", "difference between",
]

# Signals a systems-design / architecture-level task — large complexity bump.
_ARCHITECTURE_KEYWORDS = [
    "architecture", "design a", "multi-region", "multi-domain", "failover",
    "hsm", "zero-trust", "post-quantum", "pqc", "crypto-agile", "migration",
    "migrate",
]

# Signals enterprise scale / regulatory context — small additional bump on
# top of an architecture-level task.
_SCALE_KEYWORDS = [
    "national bank", "enterprise", "compliance", "regulatory", "global",
    "multi-national",
]

_LOCAL_OVERRIDE_TERMS = [
    "renew", "renewal", "revoke", "revocation",
    "password", "secret", "credential", "private key",
]


def _length_bonus(question: str) -> int:
    n = len(question)
    if n <= 30:
        return 0
    if n <= 80:
        return 3
    if n <= 150:
        return 8
    if n <= 250:
        return 12
    return 18


def _keyword_bonus(q_lower: str, keywords: list[str], weight: int, cap: int) -> int:
    total = sum(weight for kw in keywords if kw in q_lower)
    return min(total, cap)


def score_complexity(question: str) -> int:
    q_lower = question.lower()

    score = 15
    score += _length_bonus(question)
    score += _keyword_bonus(q_lower, _COMPARISON_KEYWORDS, weight=15, cap=25)
    score += _keyword_bonus(q_lower, _ARCHITECTURE_KEYWORDS, weight=20, cap=40)
    score += _keyword_bonus(q_lower, _SCALE_KEYWORDS, weight=10, cap=20)

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
