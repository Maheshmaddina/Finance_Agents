"""Future with Shivank Bank · AI Agents in Finance — run any lab from the command line.

    python main.py --help
    python main.py price "What is NVIDIA trading at?"
    python main.py bhs --company Apple --ticker AAPL
"""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="AI Agents in Finance — the bank's six tools + demos")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("earnings", help="Lab 1A: earnings-call transcript -> typed JSON card")
    s.add_argument("--compare", action="store_true", help="also run Gemini / Llama and validate them")

    sub.add_parser("cutoff", help="Lab 1B: show a plain LLM's stale 'latest' answers")

    s = sub.add_parser("price", help="Lab 1C: live price agent")
    s.add_argument("question", nargs="?",
                   default="What is today's date, and what is the current stock price of Nvidia?")

    s = sub.add_parser("ratios", help="Lab 2A: ratio retriever agent")
    s.add_argument("question", nargs="?", default=None)

    sub.add_parser("rebalance-demo", help="Lab 2B+2C: scripted day-1 chat -> journal -> day-2 recall")
    s = sub.add_parser("rebalance", help="Lab 2B+2C: interactive rebalancing chat with memory")
    s.add_argument("--session", default=None, help="resume a saved session id")

    s = sub.add_parser("trend", help="2.5: MA20 trend detector (no LLM, no key needed)")
    s.add_argument("ticker", nargs="?", default="AAPL")
    sub.add_parser("vote", help="2.5: self-consistency vote on an earnings surprise")
    s = sub.add_parser("react", help="Lab 2D: ReAct price agent with LlamaIndex (needs llama-index)")
    s.add_argument("question", nargs="?", default="What is the last price of Apple?")

    s = sub.add_parser("news", help="Lab 2E: verified Reuters news loop")
    s.add_argument("--region", default="Eurozone")
    s.add_argument("--items", type=int, default=5)
    s.add_argument("--max-attempts", type=int, default=4)

    s = sub.add_parser("bhs", help="Lab 2F: Buy/Hold/Sell three-agent pipeline")
    s.add_argument("--company", default="Apple")
    s.add_argument("--ticker", default="AAPL")

    a = p.parse_args()

    if a.cmd == "earnings":
        from finance_agents.labs import lab1a_earnings
        lab1a_earnings.run(compare=a.compare)
    elif a.cmd == "cutoff":
        from finance_agents.labs import lab1bc_price
        lab1bc_price.run_cutoff_demo()
    elif a.cmd == "price":
        from finance_agents.labs import lab1bc_price
        lab1bc_price.run(a.question)
    elif a.cmd == "ratios":
        from finance_agents.labs import lab2a_ratios
        lab2a_ratios.run(a.question) if a.question else lab2a_ratios.run()
    elif a.cmd == "rebalance-demo":
        from finance_agents.labs import lab2bc_rebalancing
        lab2bc_rebalancing.run_demo()
    elif a.cmd == "rebalance":
        from finance_agents.labs import lab2bc_rebalancing
        lab2bc_rebalancing.run_chat(a.session)
    elif a.cmd == "trend":
        from finance_agents.labs import reasoning
        reasoning.run_trend(a.ticker)
    elif a.cmd == "vote":
        from finance_agents.labs import reasoning
        reasoning.run_self_consistency()
    elif a.cmd == "react":
        from finance_agents.labs import reasoning
        reasoning.run_react(a.question)
    elif a.cmd == "news":
        from finance_agents.labs import lab2e_news
        lab2e_news.run(a.region, a.items, a.max_attempts)
    elif a.cmd == "bhs":
        from finance_agents.labs import lab2f_buy_hold_sell
        lab2f_buy_hold_sell.run(a.company, a.ticker)


if __name__ == "__main__":
    import logging
    import re

    import openai

    logging.getLogger("financetoolkit").setLevel(logging.WARNING)  # hide data-download chatter
    try:
        main()
    except openai.RateLimitError as exc:
        msg = str(exc)
        model = re.search(r"model: ([\w.\-]+)", msg)
        wait = re.search(r"retry in ([\dhms.]+)", msg)
        raise SystemExit(
            "\nRate limit reached on the AI provider's free tier"
            + (f" for {model.group(1)}" if model else "")
            + (f" (resets in about {wait.group(1).split('.')[0]}s)" if wait else "")
            + ".\nWait and retry, or switch model in .env (e.g. MODEL_AGENT=gemini-3-flash-preview)."
        ) from None
