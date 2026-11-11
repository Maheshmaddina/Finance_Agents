"""Lab 2F · Buy / Hold / Sell: three agents pass a JSON baton.

Retrieval (facts only) -> Analysis (ratings + risks) -> Decision (stance + watchlist).

Fixes the course asks for, applied here:
- typed output models (output_type) instead of JSON described in words -> no broken JSON / code fences
- "every risk must cite a ratio from the input", checked in code (the 'competition' leak)
- task guardrails between agents:
    * retrieval numbers are re-checked against the data source (no invented numbers)
    * stance is BUY/HOLD/SELL, exactly 3 reasons and 3 watchlist items
"""
from __future__ import annotations

import asyncio
import json
import math
from typing import Literal

from agents import Agent, Runner, function_tool
from pydantic import BaseModel

from ..config import MODEL_AGENT, current_year, setup_agents
from ..guardrails import GuardrailError
from ..tools.fundamentals import fetch_fundamentals

FAMILIES = ("profitability_ratios", "valuation_ratios")  # liquidity left out: long-term call


@function_tool
def get_company_fundamentals(ticker: str) -> dict:
    """Get the profitability and valuation fundamental ratios for a ticker, per fiscal year.

    Args:
        ticker: Stock ticker, e.g. "AAPL".
    """
    return fetch_fundamentals(ticker, FAMILIES)


# ---------- The baton (typed contracts between relay runners) ----------

class RatioPoint(BaseModel):
    family: Literal["profitability_ratios", "valuation_ratios"]
    ratio: str
    fiscal_year: str
    value: float | None


class RetrievalOutput(BaseModel):
    ticker: str
    fundamentals: list[RatioPoint]


class ProfitabilityView(BaseModel):
    rating: Literal["strong", "neutral", "weak"]
    evidence: list[str]


class ValuationView(BaseModel):
    rating: Literal["cheap", "fair", "expensive"]
    evidence: list[str]


class Risk(BaseModel):
    risk: str
    cited_ratios: list[str]


class AnalysisOutput(BaseModel):
    ticker: str
    fundamentals: list[RatioPoint]
    profitability: ProfitabilityView
    valuation: ValuationView
    risks: list[Risk]


class DecisionOutput(BaseModel):
    ticker: str
    stance: Literal["BUY", "HOLD", "SELL"]
    reasons: list[str]
    watchlist: list[str]


# ---------- Agents ----------

def build_agents() -> tuple[Agent, Agent, Agent]:
    retrieval = Agent(
        name="Fundamental Retrieval Agent", model=MODEL_AGENT, tools=[get_company_fundamentals],
        output_type=RetrievalOutput,
        instructions=f"""You are the Fundamental Retrieval Agent. The current year is {current_year()}.
Tasks:
1) Call the fundamentals tool for the specified company's ticker.
2) Copy EVERY ratio value the tool returns, for every fiscal year, into `fundamentals`,
   using the exact ratio names and year labels from the tool. Keep values unchanged.
Keep the output strictly factual: no interpretation.""",
    )
    analysis = Agent(
        name="Fundamental Analysis Agent", model=MODEL_AGENT, output_type=AnalysisOutput,
        instructions="""You are the Fundamental Analysis Agent.
Input: JSON from the Retrieval Agent.
Tasks:
1) Interpret the fundamentals along 2 dimensions:
   - Profitability (margins, ROA, ROE) — remember ROE is inflated when buybacks shrink equity.
   - Valuation (P/E, P/B, EV/EBITDA, EV/Sales) — compare the latest year with earlier years.
2) Identify 2-3 key risks (e.g., valuation risk, margin sustainability).
3) Copy `fundamentals` unchanged from the input.
Rules:
- Keep evidence short and specific, quoting numbers and years from the input.
- Every risk MUST cite at least one ratio name from the input in `cited_ratios`.
  Do not add risks that no supplied ratio supports (no general knowledge, e.g. competition).""",
    )
    decision = Agent(
        name="Decision Agent", model=MODEL_AGENT, output_type=DecisionOutput,
        instructions="""You are the Decision Agent.
Input: JSON from the Fundamental Analysis Agent.
Based only on that input:
1) Produce a simple stance: BUY, HOLD, or SELL.
2) Provide exactly 3 reasons grounded in the analysis.
3) Provide exactly 3 watchlist items: specific metric thresholds that would change your stance
   (e.g. "net profit margin falls below 25%").""",
    )
    return retrieval, analysis, decision


# ---------- Task guardrails (plain code) ----------

def check_retrieval(out: RetrievalOutput, source: dict, tol: float = 0.01) -> None:
    """Every number the agent passed on must match the data source."""
    if not out.fundamentals:
        raise GuardrailError("Retrieval returned no ratios.")
    for p in out.fundamentals:
        try:
            truth = source[p.family][p.ratio][p.fiscal_year]
        except KeyError:
            raise GuardrailError(f"{p.ratio} {p.fiscal_year} is not in the source data.") from None
        if (truth is None) != (p.value is None) or (
                truth is not None and not math.isclose(p.value, truth, rel_tol=tol, abs_tol=1e-4)):
            raise GuardrailError(f"{p.ratio} {p.fiscal_year}: agent said {p.value}, source says {truth}.")


def check_analysis(out: AnalysisOutput) -> None:
    names = {p.ratio.lower() for p in out.fundamentals}
    if not out.risks:
        raise GuardrailError("Analysis listed no risks.")
    for r in out.risks:
        if not r.cited_ratios or not all(c.lower() in names for c in r.cited_ratios):
            raise GuardrailError(f"Risk not traced to a supplied ratio: '{r.risk}' cites {r.cited_ratios}")


def check_decision(out: DecisionOutput) -> None:
    if len(out.reasons) != 3:
        raise GuardrailError(f"Expected 3 reasons, got {len(out.reasons)}.")
    if len(out.watchlist) != 3:
        raise GuardrailError(f"Watchlist must have 3 items, got {len(out.watchlist)}.")


async def _run_step(agent: Agent, payload: str, check, retries: int = 1):
    """Run one relay leg; if its guardrail fails, retry once with the error as feedback."""
    for attempt in range(retries + 1):
        prompt = payload if attempt == 0 else f"{payload}\n\nYour previous output was rejected: {err}. Fix it."
        out = (await Runner.run(agent, prompt)).final_output
        try:
            check(out)
            return out
        except GuardrailError as exc:
            err = str(exc)
            print(f"[guardrail] {agent.name}: {err}")
    raise GuardrailError(f"{agent.name} failed its check: {err}")


async def pipeline(company: str, ticker: str) -> tuple[AnalysisOutput, DecisionOutput]:
    """Run the three-agent relay; returns (analysis, decision). Used by the CLI and the web page."""
    retrieval, analysis, decision = build_agents()
    source = fetch_fundamentals(ticker, FAMILIES)  # cached; also used by the tool

    facts = await _run_step(retrieval, f"Provide {company} ({ticker}) fundamentals",
                            lambda o: check_retrieval(o, source))
    print(f"[1/3] retrieval OK: {len(facts.fundamentals)} verified ratio values")

    view = await _run_step(analysis, facts.model_dump_json(), check_analysis)
    print(f"[2/3] analysis OK: profitability={view.profitability.rating}, "
          f"valuation={view.valuation.rating}, {len(view.risks)} traced risks")

    verdict = await _run_step(decision, view.model_dump_json(), check_decision)
    print("[3/3] decision OK")
    return view, verdict


def run(company: str = "Apple", ticker: str = "AAPL") -> DecisionOutput:
    setup_agents()
    view, verdict = asyncio.run(pipeline(company, ticker))
    print(json.dumps({"analysis": view.model_dump(exclude={"fundamentals"}),
                      "decision": verdict.model_dump()}, indent=2))
    print("\nFor learning only — not investment advice.")
    return verdict
