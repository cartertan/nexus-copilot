import os
import requests
from openai import OpenAI


def call_glm(prompt: str, system: str = "", max_tokens: int = 2048) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return {"content": None, "where_run": "cloud:unavailable", "success": False, "error": "No API key"}

    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="z-ai/glm-5.2",
            messages=messages,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return {"content": content, "where_run": "cloud:glm-5.2", "success": True, "error": None}
    except Exception as e:
        return {"content": None, "where_run": "cloud:failed", "success": False, "error": str(e)}


def call_with_fallback(prompt: str, system: str = "", max_tokens: int = 2048) -> dict:
    result = call_glm(prompt, system=system, max_tokens=max_tokens)
    if result["success"]:
        return result

    try:
        full_prompt = f"{system}\n\n{prompt}".strip() if system else prompt
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "qwen3:30b", "prompt": full_prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json().get("response", "")
        return {
            "content": content,
            "where_run": "local:qwen3:30b",
            "success": True,
            "error": None,
            "fallback": True,
        }
    except Exception as e:
        return {
            "content": None,
            "where_run": "local:failed",
            "success": False,
            "error": str(e),
            "fallback": True,
        }
