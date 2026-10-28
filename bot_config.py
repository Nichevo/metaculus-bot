"""
Model selection for this bot.

forecasting-tools' built-in defaults are old models (gpt-4o, claude-3-7), so
we pin Claude here instead. Each role can be overridden with an env var (or a
GitHub Actions repository variable of the same name):

    FORECASTER_MODEL   writes the forecasts (the "default" llm)
    PARSER_MODEL       turns forecast text into structured predictions
    SUMMARIZER_MODEL   summarizes research
    RESEARCHER_MODEL   gathers research, e.g. "asknews/news-summaries"

Model names use litellm syntax, e.g. "openrouter/anthropic/claude-sonnet-5.5"
or "anthropic/claude-sonnet-5-5". Any role left unset falls back to the
forecasting-tools default.
"""
from __future__ import annotations

import os

from forecasting_tools import GeneralLlm

from bot_helpers import _is_real_env

# OpenRouter comes first because the Metaculus tournament hands out free
# OpenRouter credits.
_CLAUDE_DEFAULTS = {
    "OPENROUTER_API_KEY": {
        "default": "openrouter/anthropic/claude-sonnet-5.5",
        "cheap": "openrouter/anthropic/claude-haiku-5.5",
    },
    "ANTHROPIC_API_KEY": {
        "default": "anthropic/claude-sonnet-5-5",
        "cheap": "anthropic/claude-haiku-5-5",
    },
}


def _env(name: str) -> str | None:
    val = os.getenv(name)
    return val.strip() if val and val.strip() else None


def build_llms() -> dict[str, str | GeneralLlm]:
    defaults: dict[str, str] = {}
    for key_name, models in _CLAUDE_DEFAULTS.items():
        if _is_real_env(key_name):
            defaults = models
            break

    forecaster = _env("FORECASTER_MODEL") or defaults.get("default")
    parser = _env("PARSER_MODEL") or defaults.get("cheap")
    summarizer = _env("SUMMARIZER_MODEL") or defaults.get("cheap")
    researcher = _env("RESEARCHER_MODEL")

    llms: dict[str, str | GeneralLlm] = {}
    if forecaster:
        llms["default"] = GeneralLlm(
            model=forecaster,
            temperature=0.3,
            timeout=120,
            allowed_tries=2,
        )
    if parser:
        llms["parser"] = parser
    if summarizer:
        llms["summarizer"] = summarizer
    if researcher:
        llms["researcher"] = researcher
    return llms
