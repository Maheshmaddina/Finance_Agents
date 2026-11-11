"""Offline tests: no API keys, no network. They cover the plain-code parts the agents rely on."""
import json

import pandas as pd
import pytest

from finance_agents.guardrails import GuardrailError, check_ticker, count_links
from finance_agents.labs.lab1a_earnings import validate_card
from finance_agents.labs.lab2bc_rebalancing import (EpisodicMemory, MEMORY_MARKER, _text_of,
                                                    history_to_string, memory_to_document,
                                                    with_memory)
from finance_agents.labs.lab2f_buy_hold_sell import (AnalysisOutput, DecisionOutput,
                                                     ProfitabilityView, RatioPoint,
                                                     RetrievalOutput, Risk, ValuationView,
                                                     check_analysis, check_decision,
                                                     check_retrieval)
from finance_agents.labs.reasoning import parse_final
from finance_agents.tools.fundamentals import pick_ratios
from finance_agents.tools.market import ma20_trend


# --- guardrails ---
@pytest.mark.parametrize("raw,clean", [("aapl", "AAPL"), (" NVDA ", "NVDA"), ("BRK.B", "BRK.B"),
                                       ("RELIANCE.NS", "RELIANCE.NS")])
def test_ticker_ok(raw, clean):
    assert check_ticker(raw) == clean


@pytest.mark.parametrize("bad", ["", "Apple Inc", "DROP TABLE;", "A/B"])
def test_ticker_rejected(bad):
    with pytest.raises(GuardrailError):
        check_ticker(bad)


def test_count_links_distinct():
    text = "a https://www.reuters.com/x b (https://reuters.com/y) c https://www.reuters.com/x https://bbc.com/z"
    assert count_links(text, "reuters.com") == 2


def test_news_filter_source_and_window():
    from datetime import date
    from finance_agents.tools.news import filter_articles
    raw = [
        {"url": "https://www.reuters.com/a", "date": "2026-11-09T10:00:00+00:00", "title": "new"},
        {"url": "https://www.reuters.com/a", "date": "2026-11-09T10:00:00+00:00", "title": "dup"},
        {"url": "https://www.reuters.com/b", "date": "2026-11-01T10:00:00+00:00", "title": "old"},
        {"url": "https://www.ft.com/c", "date": "2026-11-09T10:00:00+00:00", "title": "other site"},
        {"url": "https://www.reuters.com/d", "date": "", "title": "no date"},
    ]
    kept = filter_articles(raw, "reuters.com", date(2026, 11, 8))
    assert [k["title"] for k in kept] == ["new"]


# --- Lab 1A contract ---
GOOD_CARD = {"company": "Apple", "quarter": "FY2025 Q3", "revenue_usd": 98700000000,
             "revenue_growth_yoy_pct": 6, "eps_usd": 1.72, "guidance": "low-mid single digit growth",
             "highlights": ["Services record"], "risks": ["tariffs"], "outlook": "positive"}


def test_card_valid():
    card, problems = validate_card(json.dumps(GOOD_CARD))
    assert problems == [] and card.revenue_usd > 90e9


def test_card_rejects_text_revenue():  # the Gemini/Llama failure from the course
    bad = {**GOOD_CARD, "revenue_usd": "$98.7B (+6% YoY)"}
    card, problems = validate_card(json.dumps(bad))
    assert card is None and any("revenue_usd" in p for p in problems)


def test_card_rejects_bad_outlook_and_fence():
    bad = {**GOOD_CARD, "outlook": "bullish"}
    _, problems = validate_card("```json\n" + json.dumps(bad) + "\n```")
    assert any("outlook" in p for p in problems)


# --- tools ---
def test_pick_ratios_drops_unknown_names():
    df = pd.DataFrame({"2024": [0.9, 1.5], "2025": [0.89, float("nan")]},
                      index=["Current Ratio", "Quick Ratio"])
    out = pick_ratios(df, ["Current Ratio", "Quick Ratio", "Typo Ratio"])
    assert set(out) == {"Current Ratio", "Quick Ratio"}
    assert out["Quick Ratio"]["2025"] is None


def test_ma20_trend():
    up = pd.Series(range(1, 61), dtype=float)
    assert ma20_trend(up)["trend"] == "Uptrend"
    assert ma20_trend(up[::-1].reset_index(drop=True))["trend"] == "Downtrend"


# --- self-consistency parsing ---
def test_parse_final():
    assert parse_final("step 1...\nFINAL: Neutral surprise") == "Neutral surprise"
    assert parse_final("**FINAL: Positive surprise.**") == "Positive surprise"
    assert parse_final("no final line") is None


# --- Lab 2B/2C ---
def test_text_of_letter_and_envelope():
    assert _text_of("hi") == "hi"
    assert _text_of([{"type": "output_text", "text": "a"}, {"type": "refusal"}]) == "a"


def test_memory_marker_stripped_from_journal():
    q = with_memory("Low vol, bullish?", ["old page"])
    assert MEMORY_MARKER in q
    assert "old page" not in history_to_string([("user", q)])


def test_memory_document_flat_metadata():
    m = EpisodicMemory(conversation_summary="s", portfolio_allocation_trajectory="t",
                       what_worked="w", what_didnt_work="d", key_points=["a", "b"])
    doc, meta = memory_to_document(m)
    assert "Key Points: a, b" in doc and meta["key_points"] == "a | b"
    assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())


# --- Lab 2F guardrails ---
SOURCE = {"profitability_ratios": {"Net Profit Margin": {"2025": 0.269}},
          "valuation_ratios": {"Price-to-Earnings": {"2025": 36.4}}}
POINTS = [RatioPoint(family="profitability_ratios", ratio="Net Profit Margin", fiscal_year="2025", value=0.269),
          RatioPoint(family="valuation_ratios", ratio="Price-to-Earnings", fiscal_year="2025", value=36.4)]


def test_retrieval_check():
    check_retrieval(RetrievalOutput(ticker="AAPL", fundamentals=POINTS), SOURCE)
    wrong = POINTS[:1] + [POINTS[1].model_copy(update={"value": 21.0})]
    with pytest.raises(GuardrailError):
        check_retrieval(RetrievalOutput(ticker="AAPL", fundamentals=wrong), SOURCE)


def _analysis(risks):
    return AnalysisOutput(ticker="AAPL", fundamentals=POINTS,
                          profitability=ProfitabilityView(rating="strong", evidence=["x"]),
                          valuation=ValuationView(rating="expensive", evidence=["y"]), risks=risks)


def test_analysis_check_catches_untraced_risk():
    check_analysis(_analysis([Risk(risk="valuation", cited_ratios=["Price-to-Earnings"])]))
    with pytest.raises(GuardrailError):
        check_analysis(_analysis([Risk(risk="market competition", cited_ratios=[])]))


def test_decision_check():
    check_decision(DecisionOutput(ticker="AAPL", stance="HOLD", reasons=["a", "b", "c"], watchlist=["x", "y", "z"]))
    with pytest.raises(GuardrailError):
        check_decision(DecisionOutput(ticker="AAPL", stance="HOLD", reasons=["a"], watchlist=["x", "y", "z"]))


# --- free-tier pacing ---
def test_rate_limiter_window():
    from finance_agents.pacing import RateLimiter, _is_per_minute
    lim = RateLimiter(3)
    assert [lim._delay() for _ in range(3)] == [0.0, 0.0, 0.0]
    assert 59 < lim._delay() <= 60          # 4th call must wait for the window to roll
    assert _is_per_minute(b"Generate content requests per minute limit exceeded quota")
    assert not _is_per_minute(b"Generate content requests per day limit exceeded quota")
