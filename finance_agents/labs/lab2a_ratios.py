"""Lab 2A · Ratio retriever (tool use).

"Give me Apple's liquidity ratios for the latest fiscal year, and explain what each one means."
The tool supplies trustworthy numbers; the instructions push the LLM past shallow readings
(e.g. "current ratio < 1 = liquidity issues") by asking for history and context.
"""
from __future__ import annotations

from agents import Agent, Runner, function_tool

from ..config import MODEL_AGENT, current_year, setup_agents
from ..tools.fundamentals import fetch_fundamentals


@function_tool
def get_company_fundamentals(ticker: str) -> dict:
    """Get profitability, valuation and liquidity ratios for a ticker, one value per fiscal year.

    Args:
        ticker: Stock ticker, e.g. "AAPL".
    """
    return fetch_fundamentals(ticker)


def build_agent() -> Agent:
    return Agent(
        name="Fundamental Retrieval Agent",
        model=MODEL_AGENT,
        instructions=f"""You are a Fundamental Retrieval Agent at a bank. The current year is {current_year()}.
Tasks:
1) Call the fundamentals tool for the specified ticker. Never invent a number.
2) Output the requested ratios with their fiscal year, exactly as the tool returned them.
3) When you interpret a ratio, compare it with the company's own previous years and mention
   context a human analyst would add (e.g. a current ratio below 1 can be a deliberate choice for a
   company with very large operating cash flow; ROE is inflated when buybacks shrink equity).""",
        tools=[get_company_fundamentals],
    )


DEFAULT_QUESTION = f"""Proceed step by step:
1- List 3 liquidity ratios and explain their significance.
2- Provide Apple's values for the three listed liquidity ratios for the most recent fiscal year,
knowing that we are in {current_year()}, and show how they changed versus prior years."""


def run(question: str = DEFAULT_QUESTION) -> str:
    setup_agents()
    result = Runner.run_sync(build_agent(), question)
    print(result.final_output)
    return result.final_output
