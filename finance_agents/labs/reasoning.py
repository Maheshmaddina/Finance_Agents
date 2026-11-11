"""Section 2.5 · Reasoning on a trading desk.

- trend: plan-then-execute MA20 trend detector (pure code, no LLM — workflow engineering)
- self_consistency: five reasoning paths vote on an earnings surprise
- react: Lab 2D, ReAct agent in LlamaIndex with streamed Thought/Action lines (optional extra)
"""
from __future__ import annotations

import asyncio
from collections import Counter

from ..config import LLM_BASE_URL, MODEL_REACT, MODEL_SUMMARY, PROVIDER, get_client, llm_key
from ..tools.market import fetch_latest_price, fetch_trend

LABELS = ("Positive surprise", "Neutral surprise", "Negative surprise")

COT_PROMPT = """You are a financial analysis agent.
Classify the earnings result as: Positive surprise, Neutral surprise or Negative surprise.
Reason step by step before producing the final classification.

Input:
{facts}

End with exactly one line: FINAL: Positive surprise | Neutral surprise | Negative surprise"""

DEFAULT_FACTS = """EPS: 1.25 vs consensus 1.20
Revenue: +4% YoY
Margins: Declining
Guidance: Raised
Management: CEO departure announced"""


def run_trend(ticker: str = "AAPL") -> dict:
    t = fetch_trend(ticker)
    print(f"{t['ticker']}: MA20 today={t['ma20_today']:.2f}, a week ago={t['ma20_week_ago']:.2f}"
          f" -> {t['trend']}")
    return t


def parse_final(text: str) -> str | None:
    """Pull the label from the 'FINAL: ...' line; ignore malformed paths instead of counting junk."""
    for line in reversed(text.strip().splitlines()):
        if "FINAL:" in line.upper():
            answer = line.split(":", 1)[1].strip().strip("*. ").lower()
            for label in LABELS:
                if label.lower() == answer or label.lower() in answer:
                    return label
            return None
    return None


def run_self_consistency(facts: str = DEFAULT_FACTS, paths: int = 5) -> Counter:
    client = get_client()
    votes: Counter = Counter()
    for i in range(paths):
        r = client.chat.completions.create(
            model=MODEL_SUMMARY, temperature=1.0,
            messages=[{"role": "user", "content": COT_PROMPT.format(facts=facts)}])
        label = parse_final(r.choices[0].message.content or "")
        print(f"path {i + 1}: {label or 'unparseable (ignored)'}")
        if label:
            votes[label] += 1
    print(votes)
    if votes:
        print("Majority:", votes.most_common(1)[0][0])
    return votes


async def _react(question: str) -> str:
    from llama_index.core.agent.workflow import AgentStream, ReActAgent
    from llama_index.core.tools import FunctionTool

    if PROVIDER == "openai":
        from llama_index.llms.openai import OpenAI as LlamaOpenAI
        llm = LlamaOpenAI(model=MODEL_REACT, api_key=llm_key())
    else:  # Gemini / custom: any OpenAI-compatible endpoint
        from llama_index.llms.openai_like import OpenAILike
        llm = OpenAILike(model=MODEL_REACT, api_key=llm_key(), api_base=LLM_BASE_URL,
                         is_chat_model=True, context_window=128000)

    agent = ReActAgent(
        llm=llm,
        tools=[FunctionTool.from_defaults(fetch_latest_price, name="get_latest_price")],
        timeout=120,
        verbose=False,
    )
    handler = agent.run(question)
    async for ev in handler.stream_events():
        if isinstance(ev, AgentStream):
            print(ev.delta, end="", flush=True)
    response = await handler
    print()
    return str(response)


def run_react(question: str = "What is the last price of Apple?") -> str:
    return asyncio.run(_react(question))
