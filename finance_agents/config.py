"""Shared settings: LLM provider, API keys, model names and bank-wide rules.

Keys are read from environment variables (or a local .env file). Never hard-code them.

LLM_PROVIDER picks who runs the models:
  gemini  (default, FREE key from https://aistudio.google.com/apikey)
  openai  (paid, the provider used in the course)
  custom  (any OpenAI-compatible server: set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL)
All three are reached through the OpenAI Python SDK via an OpenAI-compatible endpoint.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Where persistent data lives (SQLite sessions, Chroma memory)
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
SESSIONS_DB = DATA_DIR / "portfolio_sessions.db"
CHROMA_DIR = DATA_DIR / "chroma_ptf_rebal"

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

_PRESETS = {
    "gemini": {
        "key_env": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        # flash-lite everywhere: the free tier allows only ~20 requests/day for "flash" models.
        # To use a stronger model for agents, set MODEL_AGENT=gemini-3.5-flash in .env.
        "models": dict(summary="gemini-3.5-flash-lite", smart="gemini-3.5-flash-lite",
                       agent="gemini-3.5-flash-lite", react="gemini-3.5-flash-lite",
                       news="gemini-3.5-flash-lite", embedding="gemini-embedding-001"),
    },
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "base_url": None,
        "models": dict(summary="gpt-4.1-mini", smart="gpt-4.1", agent="gpt-4o-mini",
                       react="gpt-4o", news="gpt-4.1", embedding="text-embedding-3-small"),
    },
    "custom": {
        "key_env": "LLM_API_KEY",
        "base_url": os.getenv("LLM_BASE_URL"),
        "models": {k: os.getenv("LLM_MODEL", "") for k in
                   ("summary", "smart", "agent", "react", "news")}
                  | {"embedding": os.getenv("LLM_EMBEDDING_MODEL", "")},
    },
}
if PROVIDER not in _PRESETS:
    raise SystemExit(f"LLM_PROVIDER must be one of {list(_PRESETS)}, got '{PROVIDER}'.")

_P = _PRESETS[PROVIDER]
LLM_KEY_ENV: str = _P["key_env"]
LLM_BASE_URL: str | None = _P["base_url"]

# Models per job (override any of them in .env)
MODEL_SUMMARY = os.getenv("MODEL_SUMMARY", _P["models"]["summary"])      # Lab 1A, 2C, vote
MODEL_SMART = os.getenv("MODEL_SMART", _P["models"]["smart"])            # Lab 1B, 1C
MODEL_AGENT = os.getenv("MODEL_AGENT", _P["models"]["agent"])            # Lab 2A, 2B, 2F
MODEL_REACT = os.getenv("MODEL_REACT", _P["models"]["react"])            # Lab 2D
MODEL_NEWS = os.getenv("MODEL_NEWS", _P["models"]["news"])               # Lab 2E
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", _P["models"]["embedding"])

# Tool guardrail: tickers the bank does not allow agents to query (compliance list)
RESTRICTED_TICKERS = {
    t.strip().upper() for t in os.getenv("RESTRICTED_TICKERS", "").split(",") if t.strip()
}


def require_key(name: str) -> str:
    """Return an API key or stop with a clear message telling the user what to set."""
    value = os.getenv(name)
    if not value:
        raise SystemExit(
            f"Missing {name}. Copy .env.example to .env and fill it in, "
            f"or run: export {name}=...  (see README.md for where to get it)"
        )
    return value


def llm_key() -> str:
    return require_key(LLM_KEY_ENV)


# Requests per minute to allow (Gemini free tier: 15 for flash-lite). 0 = no pacing.
LLM_RPM = int(os.getenv("LLM_RPM", "12" if PROVIDER == "gemini" else "0"))
_limiter = None


def _get_limiter():
    global _limiter
    if _limiter is None:
        from .pacing import RateLimiter
        _limiter = RateLimiter(LLM_RPM)
    return _limiter


def get_client():
    """Sync OpenAI-SDK client pointed at the chosen provider, paced for free-tier limits."""
    from openai import DefaultHttpxClient, OpenAI

    from .pacing import PacedTransport

    http = DefaultHttpxClient(transport=PacedTransport(_get_limiter()))
    # Pacing handles per-minute limits; one SDK retry covers other brief errors (e.g. 503).
    return OpenAI(api_key=llm_key(), base_url=LLM_BASE_URL, max_retries=1, http_client=http)


_agents_ready = False


def setup_agents() -> None:
    """Point the OpenAI Agents SDK at the chosen provider (call before running any Agent)."""
    global _agents_ready
    if _agents_ready:
        return
    from agents import set_default_openai_api, set_default_openai_client, set_tracing_disabled
    from openai import AsyncOpenAI, DefaultAsyncHttpxClient

    from .pacing import PacedAsyncTransport

    http = DefaultAsyncHttpxClient(transport=PacedAsyncTransport(_get_limiter()))
    client = AsyncOpenAI(api_key=llm_key(), base_url=LLM_BASE_URL, max_retries=1, http_client=http)
    set_default_openai_client(client, use_for_tracing=False)
    if PROVIDER != "openai":
        # Non-OpenAI providers speak Chat Completions, not the Responses API,
        # and can't receive OpenAI traces.
        set_default_openai_api("chat_completions")
        set_tracing_disabled(True)
    _agents_ready = True


def current_year() -> int:
    """Today's year, passed to agents so they pick the right 'most recent fiscal year'."""
    return date.today().year
