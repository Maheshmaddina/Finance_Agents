"""Plain-code checks used across labs (no LLM needed, so they are cheap and testable).

- Tool guardrail: is this a valid, allowed ticker before we call a data API?
- Output contract checks: did the model return the right types?
"""
from __future__ import annotations

import re

from .config import RESTRICTED_TICKERS

# 1–5 letters, optional class/exchange suffix: AAPL, BRK.B, BRK-B, RELIANCE.NS, SAP.DE
_TICKER_RE = re.compile(r"^[A-Z0-9]{1,10}([.\-][A-Z0-9]{1,4})?$")


class GuardrailError(ValueError):
    """Raised when an input or output breaks a bank rule."""


def check_ticker(symbol: str) -> str:
    """Normalise a ticker and refuse anything malformed or on the restricted list."""
    cleaned = (symbol or "").strip().upper()
    if not _TICKER_RE.match(cleaned):
        raise GuardrailError(f"'{symbol}' does not look like a stock ticker.")
    if cleaned in RESTRICTED_TICKERS:
        raise GuardrailError(f"{cleaned} is on the bank's restricted list.")
    return cleaned


def count_links(text: str, domain: str) -> int:
    """Count distinct links to a given domain inside a text (used by the news check)."""
    pattern = rf"https?://(?:www\.)?{re.escape(domain)}[^\s)\]\"'>]*"
    return len(set(re.findall(pattern, text or "")))
