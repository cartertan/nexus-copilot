import logging
from typing import Optional

from src.core.providers import ProviderRegistry
from src.core.token_optimizer import should_use_cloud

logger = logging.getLogger(__name__)

VALID_TASK_TYPES = {
    "EMAIL_DRAFT",
    "EMAIL_SEND",
    "MEETING_SUMMARY",
    "RFP_ANALYSIS",
    "PKI_QA",
    "RESEARCH",
    "GENERAL",
    "BENCHMARK",
}

_SYSTEM_PROMPT = (
    "You are a task classifier. Classify the user's task into exactly one category:\n\n"
    "EMAIL_SEND — explicitly sending an email to someone (e.g. 'send an email to', 'email John at')\n"
    "EMAIL_DRAFT — drafting or composing an email or message without explicitly sending it\n"
    "MEETING_SUMMARY — summarising meeting notes or a transcript\n"
    "RFP_ANALYSIS — analysing an RFP/RFI/tender document or requirement\n"
    "PKI_QA — any question about PKI, certificates, CAs, OCSP, CRLs, HSMs, TLS/SSL, "
    "encryption, or post-quantum cryptography — including 'what is', 'explain', or "
    "'how does X work' questions on these topics\n"
    "RESEARCH — open-ended research on a market, vendor landscape, or technology trend "
    "not specifically about PKI/certificates\n"
    "GENERAL — anything else\n"
    "BENCHMARK — explicit benchmark requests\n\n"
    "Respond with ONLY the matching category string — no explanation, no punctuation, "
    "no extra text."
)


def detect_sensitive_class(
    text: str, sensitive_classes: dict[str, list[str]]
) -> Optional[str]:
    """
    Return the name of the first sensitive data class whose signal terms
    appear in `text`, or None if no sensitive data signal is present.

    This is purely a data-sensitivity check: it looks for signals like
    "customer", "RFP", "pricing", "account" in the actual text. It never
    keys off task topic (e.g. PKI/crypto subject matter is not sensitive
    on its own — only actual customer/account/pricing data is).
    """
    q_lower = text.lower()
    for class_name, terms in sensitive_classes.items():
        for term in terms:
            if term and term.lower() in q_lower:
                return class_name
    return None


class TaskRouter:
    def __init__(self, config: dict, registry: ProviderRegistry):
        self._config = config
        self._registry = registry
        self._sensitive_classes: dict[str, list[str]] = config.get(
            "sensitive_classes", {}
        )
        router_cfg = config.get("task_models", {}).get("router", {})
        self._router_model: str = router_cfg.get("primary", "ollama/gemma4:e4b")

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, user_input: str) -> str:
        """Classify user input into a task type string using the router model.

        This is purely topic classification and never consults sensitivity
        signals — sensitivity is detected independently in route().
        """
        client, model = self._registry.resolve(self._router_model)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify this task: {user_input}"},
        ]
        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=16
        )
        raw = response.choices[0].message.content.strip().upper()
        for task_type in VALID_TASK_TYPES:
            if task_type in raw:
                return task_type
        return "GENERAL"

    # ------------------------------------------------------------------
    # Routing decision (no LLM call to the worker model)
    # ------------------------------------------------------------------

    def route(self, user_input: str) -> dict:
        """
        Determine where this prompt should run and on which model.

        Returns a RoutingDecision dict:
          task_type, model, where_run, complexity_score,
          forced_local, sensitive_class, local_forced_by,
          fallback_reason, thinking_mode

        forced_local / sensitive_class reflect the data-sensitivity gate only.
        local_forced_by explains *why* local was forced (sensitive_gate,
        cloud_model_blocked, or none). fallback_reason is reserved for
        genuine fallback events (cloud was expected but unavailable) and is
        never used to describe the gate or a natural low-complexity choice.
        """
        task_type = self.classify(user_input)
        key = task_type.lower()
        task_cfg = self._config.get("task_models", {}).get(key, {})

        cloud_chain: list[str] = task_cfg.get("cloud_chain", [])
        fallback_chain: list[str] = task_cfg.get("fallback_chain", [])
        thinking_mode: bool = task_cfg.get("thinking_mode", False)

        use_cloud, score = should_use_cloud(user_input)
        sensitive_class = detect_sensitive_class(user_input, self._sensitive_classes)
        forced_local: bool = sensitive_class is not None

        local_forced_by = "none"
        fallback_reason: Optional[str] = None
        cloud_attempted = False

        if forced_local:
            local_forced_by = "sensitive_gate"
            ordered = fallback_chain
        elif not use_cloud:
            ordered = fallback_chain
        else:
            credits = self._registry.check_openrouter_credits()
            if credits is not None and credits <= 0:
                local_forced_by = "cloud_model_blocked"
                fallback_reason = "no_credits"
                ordered = fallback_chain
            else:
                cloud_attempted = True
                ordered = cloud_chain + fallback_chain

        selected: Optional[str] = None
        where_run = "none"

        for model_string in ordered:
            try:
                self._registry.resolve(model_string)  # validates; raises on :cloud
                selected = model_string
                provider = model_string.split("/")[0]
                where_run = "local" if provider == "ollama" else "cloud"
                # Only a genuine fallback if cloud was actually attempted
                # (score/gate said cloud) and we still ended up on a
                # fallback-chain model.
                if (
                    cloud_attempted
                    and cloud_chain
                    and local_forced_by == "none"
                    and fallback_reason is None
                    and model_string in fallback_chain
                ):
                    local_forced_by = "cloud_model_blocked"
                    fallback_reason = "cloud_unavailable"
                break
            except ValueError as exc:
                logger.warning("Skipping model %r: %s", model_string, exc)

        decision = {
            "task_type": task_type,
            "model": selected,
            "where_run": where_run,
            "complexity_score": score,
            "forced_local": forced_local,
            "sensitive_class": sensitive_class,
            "local_forced_by": local_forced_by,
            "fallback_reason": fallback_reason,
            "thinking_mode": thinking_mode,
        }
        logger.info(
            "route: task=%s model=%s where_run=%s local_forced_by=%s "
            "fallback_reason=%s score=%d",
            task_type,
            selected,
            where_run,
            local_forced_by,
            fallback_reason,
            score,
        )
        return decision

    # ------------------------------------------------------------------
    # Execution with chain walking
    # ------------------------------------------------------------------

    def execute(
        self,
        user_input: str,
        system: str = "",
        max_tokens: int = 2048,
    ) -> dict:
        """
        Route and execute the prompt, walking the chain on failure.
        Returns the routing decision merged with the LLM response fields.
        """
        decision = self.route(user_input)
        task_type = decision["task_type"]
        key = task_type.lower()
        task_cfg = self._config.get("task_models", {}).get(key, {})

        cloud_chain: list[str] = task_cfg.get("cloud_chain", [])
        fallback_chain: list[str] = task_cfg.get("fallback_chain", [])
        forced_local: bool = decision["forced_local"]
        use_cloud = decision["where_run"] == "cloud"

        cloud_attempted = False
        if forced_local or not use_cloud:
            ordered = fallback_chain
        else:
            credits = self._registry.check_openrouter_credits()
            if credits is not None and credits <= 0:
                ordered = fallback_chain
            else:
                cloud_attempted = True
                ordered = cloud_chain + fallback_chain

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_input})

        last_error: Optional[str] = None
        for model_string in ordered:
            try:
                client, model_name = self._registry.resolve(model_string)
            except ValueError as exc:
                logger.warning("Skipping model %r: %s", model_string, exc)
                last_error = str(exc)
                continue

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                provider = model_string.split("/")[0]
                actual_where = "local" if provider == "ollama" else "cloud"
                actual_local_forced_by = decision["local_forced_by"]
                actual_fallback = decision["fallback_reason"]
                if (
                    cloud_attempted
                    and cloud_chain
                    and model_string in fallback_chain
                    and actual_fallback is None
                ):
                    actual_fallback = "cloud_unavailable"
                    actual_local_forced_by = "cloud_model_blocked"

                logger.info(
                    "execute: success model=%s where_run=%s local_forced_by=%s "
                    "fallback_reason=%s",
                    model_string,
                    actual_where,
                    actual_local_forced_by,
                    actual_fallback,
                )
                return {
                    **decision,
                    "model": model_string,
                    "where_run": actual_where,
                    "local_forced_by": actual_local_forced_by,
                    "fallback_reason": actual_fallback,
                    "content": content,
                    "success": True,
                    "error": None,
                }
            except Exception as exc:
                logger.warning("Model %r failed: %s", model_string, exc)
                last_error = str(exc)

        return {
            **decision,
            "content": None,
            "success": False,
            "error": last_error or "All models in chain failed",
        }
