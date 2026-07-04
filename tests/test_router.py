"""
Locks in the routing-gate fix: sensitivity must key on data signals
(customer/RFP/pricing/account), never on task topic (e.g. PKI), and
complexity scoring must be calibrated so trivial questions stay local and
enterprise-scale design questions go to cloud.
"""

from unittest.mock import patch

import pytest

from src.core.providers import ProviderRegistry
from src.core.router import TaskRouter, detect_sensitive_class
from src.core.token_optimizer import score_complexity

CONFIG = {
    "providers": {
        "ollama": {"base_url": "http://localhost:11434/v1", "api_key": "ollama"},
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        },
    },
    "sensitive_classes": {
        "customer_data": ["customer", "client"],
        "rfp_content": ["rfp", "rfi", "tender"],
        "account_info": ["account", "acct #"],
        "pricing": ["pricing", "price list", "quote", "rate card"],
    },
    "task_models": {
        "router": {"primary": "ollama/gemma4:e4b"},
        "pki_qa": {
            "cloud_chain": ["openrouter/z-ai/glm-5.2"],
            "fallback_chain": ["ollama/qwen3:30b"],
            "thinking_mode": False,
        },
        "rfp_analysis": {
            "cloud_chain": ["openrouter/z-ai/glm-5.2"],
            "fallback_chain": ["ollama/qwen3:30b"],
            "thinking_mode": True,
        },
    },
}


def make_router() -> TaskRouter:
    registry = ProviderRegistry(CONFIG["providers"])
    return TaskRouter(CONFIG, registry)


# ----------------------------------------------------------------------
# detect_sensitive_class: signals only, never topic
# ----------------------------------------------------------------------


def test_sensitive_gate_ignores_pki_topic():
    assert detect_sensitive_class("Explain what a CRL is", CONFIG["sensitive_classes"]) is None
    assert (
        detect_sensitive_class(
            "Design a multi-region PKI architecture with HSM failover for a national bank",
            CONFIG["sensitive_classes"],
        )
        is None
    )


def test_sensitive_gate_fires_on_data_signals():
    result = detect_sensitive_class(
        "Summarize this customer RFP pricing section", CONFIG["sensitive_classes"]
    )
    assert result is not None


# ----------------------------------------------------------------------
# score_complexity calibration
# ----------------------------------------------------------------------


def test_score_trivial_factual_question():
    assert score_complexity("Explain what a CRL is") < 20


def test_score_simple_task():
    assert score_complexity("draft a follow-up email") < 30


def test_score_comparison_question():
    score = score_complexity("compare OCSP vs CRL for a mobile fleet")
    assert 40 <= score <= 55


def test_score_enterprise_architecture_question():
    score = score_complexity(
        "design a multi-region PKI architecture with HSM failover for a national bank"
    )
    assert 65 <= score <= 80


# ----------------------------------------------------------------------
# TaskRouter.route(): end-to-end gate + scoring behavior
# classify() calls a live LLM; it's patched here since task-topic
# classification is orthogonal to the gate/scoring logic under test.
# ----------------------------------------------------------------------


def test_crl_question_routes_local_gate_off():
    router = make_router()
    with patch.object(TaskRouter, "classify", return_value="PKI_QA"):
        decision = router.route("Explain what a CRL is")

    assert decision["where_run"] == "local"
    assert decision["forced_local"] is False
    assert decision["local_forced_by"] == "none"
    assert decision["complexity_score"] < 20


def test_bank_architecture_routes_cloud_gate_off():
    router = make_router()
    with patch.object(TaskRouter, "classify", return_value="PKI_QA"):
        decision = router.route(
            "Design a multi-region PKI architecture with HSM failover for a national bank"
        )

    assert decision["where_run"] == "cloud"
    assert decision["model"] == "openrouter/z-ai/glm-5.2"
    assert decision["forced_local"] is False
    assert decision["local_forced_by"] == "none"
    assert decision["complexity_score"] > 60


def test_customer_rfp_pricing_routes_local_gate_on():
    router = make_router()
    with patch.object(TaskRouter, "classify", return_value="RFP_ANALYSIS"):
        decision = router.route("Summarize this customer RFP pricing section")

    assert decision["where_run"] == "local"
    assert decision["forced_local"] is True
    assert decision["local_forced_by"] == "sensitive_gate"
    assert decision["sensitive_class"] is not None
