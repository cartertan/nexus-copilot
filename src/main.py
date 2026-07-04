"""
Entry point: classify a prompt and print the routing decision as JSON.

Usage:
    python -m src.main "your prompt here"
    python -m src.main --prompt "your prompt here"
"""

import json
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.core.providers import ProviderRegistry
from src.core.router import TaskRouter

_REPO_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _REPO_ROOT / "config" / "models.yaml"
_ENV_PATH = _REPO_ROOT / ".env"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s — %(message)s",
)


def load_config() -> dict:
    load_dotenv(_ENV_PATH)
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def build_router(config: dict) -> TaskRouter:
    registry = ProviderRegistry(config.get("providers", {}))
    return TaskRouter(config, registry)


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]

    if not args:
        print("Usage: python -m src.main <prompt>", file=sys.stderr)
        sys.exit(1)

    # Accept either positional arg or --prompt <value>
    if args[0] == "--prompt" and len(args) > 1:
        prompt = args[1]
    else:
        prompt = " ".join(args)

    config = load_config()
    router = build_router(config)
    decision = router.route(prompt)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
