"""Lab 1B · Knowledge cutoff, and Lab 1C · "What's NVIDIA trading at?" with a live price agent.

1B shows a plain LLM serving stale "latest" news. 1C shows the fix: give it a tool.
Asking for the date too is the lie detector — a tool answer carries today's trading date.
"""
from __future__ import annotations

from agents import Agent, Runner, function_tool

from ..config import MODEL_SMART, get_client, setup_agents
from ..tools.market import fetch_latest_price

get_latest_stock_price = function_tool(fetch_latest_price, name_override="get_latest_stock_price")


def build_price_agent() -> Agent:
    return Agent(
        name="Stock Price Agent",
        instructions=(
            "You answer questions about current stock prices. Always call the price tool; never "
            "answer prices from memory. Convert company names to tickers (Nvidia -> NVDA). "
            "Report the price with the date/time the tool returned. If the tool returns an error, "
            "say so instead of guessing."
        ),
        model=MODEL_SMART,
        tools=[get_latest_stock_price],
    )


def run_cutoff_demo() -> None:
    client = get_client()
    for question in ["Give me the latest news about NVIDIA.", "What is the last price of NVIDIA."]:
        r = client.chat.completions.create(model=MODEL_SMART,
                                           messages=[{"role": "user", "content": question}])
        print(f"--- Plain {MODEL_SMART}, no tools: {question}\n{r.choices[0].message.content}\n")


def run(question: str = "What is today's date, and what is the current stock price of Nvidia?") -> str:
    setup_agents()
    result = Runner.run_sync(starting_agent=build_price_agent(), input=question)
    print(result.final_output)
    return result.final_output
