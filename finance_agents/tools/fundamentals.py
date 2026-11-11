"""Financial ratios from Financial Modeling Prep, calculated by Finance Toolkit.

Used by Lab 2A (all three families) and Lab 2F (profitability + valuation only).
Results are cached per ticker so the Lab 2F guardrail can re-check the agent's numbers
against the source without a second API call.
"""
from __future__ import annotations

import math
from functools import lru_cache

import pandas as pd

from ..config import require_key
from ..guardrails import check_ticker

PROFITABILITY = ["Return on Equity", "Return on Assets", "Gross Margin",
                 "Operating Margin", "Net Profit Margin"]
VALUATION = ["Price-to-Earnings", "Price-to-Book", "Price-to-Cash-Flow",
             "Price-to-Free-Cash-Flow", "EV-to-EBITDA", "EV-to-Sales"]
LIQUIDITY = ["Current Ratio", "Quick Ratio", "Cash Ratio", "Working Capital",
             "Operating Cash Flow Ratio", "Operating Cash Flow to Sales Ratio",
             "Short Term Coverage Ratio"]

FAMILIES = {
    "profitability_ratios": PROFITABILITY,
    "valuation_ratios": VALUATION,
    "liquidity_ratios": LIQUIDITY,
}


def pick_ratios(ratios: pd.DataFrame, names: list[str]) -> dict:
    """Keep only the requested rows that exist, as compact {ratio: {year: value}}.

    `index.intersection` quietly drops names that are not in the table, so a typo
    can't crash `.loc` with a KeyError.
    """
    subset = ratios.loc[ratios.index.intersection(names)].round(4)
    out: dict[str, dict[str, float | None]] = {}
    for name, row in subset.iterrows():
        out[str(name)] = {
            str(period): (None if pd.isna(v) or math.isinf(v) else float(v))
            for period, v in row.items()
        }
    return out


@lru_cache(maxsize=32)
def _all_ratios(ticker: str) -> pd.DataFrame:
    from financetoolkit import Toolkit  # imported lazily: it is slow to load

    toolkit = Toolkit([ticker], api_key=require_key("FINANCIAL_MODELING_PREP_API_KEY"),
                      start_date="2022-01-01", progress_bar=False)
    ratios = toolkit.ratios.collect_all_ratios()
    if isinstance(ratios.index, pd.MultiIndex):  # multi-ticker layout: (ticker, ratio)
        ratios = ratios.xs(ticker, level=0)
    if ratios.empty:
        raise ValueError(f"No ratio data returned for {ticker}.")
    return ratios


def fetch_fundamentals(ticker: str, families: tuple[str, ...] = tuple(FAMILIES)) -> dict:
    """Return the requested ratio families for one ticker, one value per fiscal year."""
    ticker = check_ticker(ticker)
    ratios = _all_ratios(ticker)
    return {"ticker": ticker, **{f: pick_ratios(ratios, FAMILIES[f]) for f in families}}
