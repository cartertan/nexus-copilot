import logging
import os
from typing import Optional

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)

_CREDITS_URL = "https://openrouter.ai/api/v1/auth/key"


class ProviderRegistry:
    """Resolves 'provider/model' strings into OpenAI-SDK clients from config."""

    def __init__(self, providers_config: dict):
        self._cfg = providers_config
        self._clients: dict[str, OpenAI] = {}

    def _build_client(self, provider: str) -> OpenAI:
        pcfg = self._cfg.get(provider)
        if not pcfg:
            raise ValueError(f"Unknown provider: {provider!r}")
        base_url = pcfg.get("base_url", "")
        api_key = pcfg.get("api_key")
        if not api_key:
            env_var = pcfg.get("api_key_env", "")
            api_key = os.environ.get(env_var, "none")
        return OpenAI(api_key=api_key, base_url=base_url)

    def resolve(self, model_string: str) -> tuple[OpenAI, str]:
        """
        Parse 'provider/model' → (OpenAI client, model_name).

        Raises ValueError if the string contains ':cloud' or has no '/' separator.
        'ollama/qwen3:30b'         → (ollama_client, 'qwen3:30b')
        'openrouter/z-ai/glm-5.2' → (openrouter_client, 'z-ai/glm-5.2')
        """
        if ":cloud" in model_string.lower():
            raise ValueError(
                f"Rejected: model string contains ':cloud' — {model_string!r}"
            )

        provider, sep, model_name = model_string.partition("/")
        if not sep or not model_name:
            raise ValueError(
                f"Invalid model string (expected provider/model): {model_string!r}"
            )

        if provider not in self._clients:
            self._clients[provider] = self._build_client(provider)

        return self._clients[provider], model_name

    def check_openrouter_credits(self) -> Optional[float]:
        """Return remaining OpenRouter credits, or None if the check fails."""
        or_cfg = self._cfg.get("openrouter", {})
        api_key = or_cfg.get("api_key") or os.environ.get(
            or_cfg.get("api_key_env", ""), ""
        )
        if not api_key:
            logger.warning("No OpenRouter API key; skipping credit check")
            return None

        try:
            resp = requests.get(
                _CREDITS_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            limit = data.get("limit")
            usage = data.get("usage", 0.0)
            if limit is None:
                return float("inf")
            return float(limit) - float(usage)
        except Exception as exc:
            logger.warning("OpenRouter credit check failed: %s", exc)
            return None
