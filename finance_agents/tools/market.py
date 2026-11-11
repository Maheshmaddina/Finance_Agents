"""Live market data from Yahoo Finance (via yfinance).

The plain functions here do the work; `price_tool` wraps one as an agent tool.
Arithmetic (moving averages) stays in code — never ask an LLM to do sums.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from ..guardrails import GuardrailError, check_ticker


def fetch_latest_price(symbol: str) -> dict:
    """Fetch the current stock price and its timestamp for the given ticker symbol.

    Args:
        symbol: Stock ticker, e.g. "NVDA" for NVIDIA or "AAPL" for Apple.
    """
    try:
        symbol = check_ticker(symbol)
    except GuardrailError as exc:
        return {"error": str(exc)}

    ticker = yf.Ticker(symbol)
    # Minute bars give the latest trade during market hours; fall back to daily bars.
    hist = ticker.history(period="1d", interval="1m")
    if hist.empty:
        hist = ticker.history(period="5d")
    if hist.empty:
        return {"error": f"No price data found for ticker: {symbol}"}

    last = hist.iloc[-1]
    return {
        "ticker": symbol,
        "price": round(float(last["Close"]), 4),
        "datetime": last.name.strftime("%Y-%m-%d %H:%M %Z").strip(),
        "date": str(last.name.date()),
    }


def ma20_trend(prices: pd.Series, lookback: int = 5) -> dict:
    """Label the trend by comparing today's 20-day moving average with `lookback` days earlier."""
    ma20 = prices.rolling(window=20).mean().dropna()
    if len(ma20) <= lookback:
        raise ValueError("Not enough price history for a 20-day moving average comparison.")
    today_ma = float(ma20.iloc[-1])
    earlier_ma = float(ma20.iloc[-1 - lookback])
    return {
        "ma20_today": round(today_ma, 2),
        "ma20_week_ago": round(earlier_ma, 2),
        "trend": "Uptrend" if today_ma > earlier_ma else "Downtrend",
    }


def fetch_trend(symbol: str) -> dict:
    """Plan-then-execute trend detector: 3 months of closes -> MA20 -> compare -> label."""
    symbol = check_ticker(symbol)
    prices = yf.Ticker(symbol).history(period="3mo")["Close"]
    return {"ticker": symbol, **ma20_trend(prices)}
