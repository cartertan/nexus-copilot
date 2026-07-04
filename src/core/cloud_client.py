import logging
from typing import Optional

from src.core.providers import ProviderRegistry

logger = logging.getLogger(__name__)


def call_model(
    model_string: str,
    prompt: str,
    registry: ProviderRegistry,
    system: str = "",
    max_tokens: int = 2048,
) -> dict:
    """
    Call a single model identified by 'provider/model' through the registry.
    Returns a result dict with content, where_run, success, error.
    """
    try:
        client, model_name = registry.resolve(model_string)
    except ValueError as exc:
        return {
            "content": None,
            "where_run": "rejected",
            "success": False,
            "error": str(exc),
        }

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        provider = model_string.split("/")[0]
        where_run = f"local:{model_name}" if provider == "ollama" else f"cloud:{model_name}"
        return {"content": content, "where_run": where_run, "success": True, "error": None}
    except Exception as exc:
        return {"content": None, "where_run": "failed", "success": False, "error": str(exc)}


def call_with_fallback(
    chain: list[str],
    prompt: str,
    registry: ProviderRegistry,
    system: str = "",
    max_tokens: int = 2048,
) -> dict:
    """
    Walk a list of 'provider/model' strings, returning the first successful result.
    Logs fallback_reason when the first model is skipped.
    """
    last: Optional[dict] = None
    for i, model_string in enumerate(chain):
        result = call_model(model_string, prompt, registry, system=system, max_tokens=max_tokens)
        if result["success"]:
            if i > 0:
                logger.info(
                    "fallback: using %r after %d failure(s); fallback_reason=cloud_unavailable",
                    model_string,
                    i,
                )
            return result
        logger.warning("Model %r failed: %s", model_string, result["error"])
        last = result

    return last or {
        "content": None,
        "where_run": "none",
        "success": False,
        "error": "Empty chain",
    }
