"""Lab 2B + 2C · Rebalancing assistant with working memory and episodic memory.

Working memory (2B): one SQLiteSession per conversation, so turn 2 ("reduce equity to 50%")
still knows the 70/20/10 starting point. Saved to disk so it survives restarts.

Episodic memory (2C): when a conversation ends, an LLM writes a "journal page" (5-key JSON),
which is embedded and filed in ChromaDB. A new conversation searches the cabinet by meaning and
pastes the best pages into the prompt as "guidance, not strict rules".

Improvements the course recommends, applied here:
- unique memory ids per session (no silently skipped duplicates)
- all top matches are used, not just one
- the journal page is written automatically when the chat closes
- the journal page is produced with Structured Outputs (typed, not just "valid JSON")
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime

from agents import Agent, Runner, SQLiteSession
from pydantic import BaseModel

from ..config import (CHROMA_DIR, EMBEDDING_MODEL, LLM_BASE_URL, MODEL_AGENT, MODEL_SUMMARY,
                      PROVIDER, SESSIONS_DB, get_client, llm_key, setup_agents)

# One drawer per provider: embeddings from different models have different sizes and can't mix
COLLECTION_NAME = f"episodic_memory_ptf_rebal_{PROVIDER}"

PORTFOLIO_INSTRUCTIONS = """You are a Portfolio Rebalancing Agent for a professional investor.
Your goal is to propose clear, explainable rebalancing actions that keep the
portfolio aligned with the target risk/return profile and constraints.
Your role
- Analyze the current portfolio (weights, asset classes, sectors, factors, cash).
- Incorporate market regime information (volatility, trends, macro environment).
- Compare current allocation to:
  - Target allocation or risk budget (if provided),
  - Or to a reasonably diversified allocation based on the information you have.
- Propose specific rebalancing trades with approximate percentage changes and short rationales.
- Prioritize diversification, risk control, and alignment with the investor's objectives.
- Keep answers concise and structured, and state assumptions when information is missing.
- Treat the allocation provided in the user's prompt as the portfolio's current allocation.
- Check that proposed weights add up to 100%.
Always explain your reasoning. This is educational output, not investment advice."""


def build_portfolio_agent() -> Agent:
    return Agent(name="Portfolio Rebalancing Agent", model=MODEL_AGENT,
                 instructions=PORTFOLIO_INSTRUCTIONS)


# ---------- Lab 2B: reading the notepad back ----------

def _text_of(content) -> str:
    """User content is a plain string; assistant content is a list of parts (open the envelope)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") in ("output_text", "input_text"))
    return ""


async def read_history(session: SQLiteSession) -> list[tuple[str, str]]:
    """Session items -> clean (role, text) pairs. Tool-call items have no role and are skipped."""
    history = []
    for msg in await session.get_items():
        role = msg.get("role")
        if role in ("user", "assistant"):
            text = _text_of(msg.get("content", ""))
            if text:
                history.append((role, text))
    return history


# ---------- Lab 2C: journal page -> cabinet -> recall ----------

class EpisodicMemory(BaseModel):
    conversation_summary: str
    portfolio_allocation_trajectory: str
    what_worked: str
    what_didnt_work: str
    key_points: list[str]


EPISODIC_MEMORY_PROMPT_TEMPLATE = """
You will read a conversation between a user and a portfolio-rebalancing agent.
Extract an episodic memory that will help the agent in future sessions.

OUTPUT FORMAT:
Return only valid JSON in this exact schema:
{{
  "conversation_summary": string,            // 3-5 sentences: market conditions, portfolio situation, actions taken
  "portfolio_allocation_trajectory": string, // how allocations changed, with the market situation each time
  "what_worked": string,                     // alignment with goals, risk management, clear decisions
  "what_didnt_work": string,                 // unclear strategy, over/under-reaction, concentration risk, data issues
  "key_points": ["string"]                   // the most important insights to keep for future sessions
}}

GOOD key point: "High volatility + declining trend: client accepted cutting equity 70% -> 50%, preferred bonds over cash."
POOR key point: "The conversation went well."
GOOD trajectory: "[{{market: {{volatility: high, trend: declining}}, allocation: {{stocks: 0.70, bonds: 0.20, cash: 0.10}}}}, ...]"
POOR trajectory: "The allocation changed."
Only record facts that appear in the conversation. Never record personality guesses about the client.

Here is the conversation between the user and the principal portfolio-rebalancing agent:
{conversation}
"""


MEMORY_MARKER = "\n\nHere are relevant insights from past rebalancing episodes."


def history_to_string(history: list[tuple[str, str]]) -> str:
    # Drop pasted-in past memories so old episodes aren't re-filed as part of this one
    return "\n\n".join(f'{role}: """{text.split(MEMORY_MARKER)[0]}"""' for role, text in history)


def memory_to_document(memory: EpisodicMemory) -> tuple[str, dict]:
    """Two versions of the page: readable text (embedded) and flat metadata (folder labels)."""
    document_text = (
        f"Conversation Summary: {memory.conversation_summary}\n"
        f"Portfolio Allocation Trajectory: {memory.portfolio_allocation_trajectory}\n"
        f"What Worked: {memory.what_worked}\n"
        f"What Didn't Work: {memory.what_didnt_work}\n"
        f"Key Points: {', '.join(memory.key_points)}"
    )
    metadata = memory.model_dump()
    metadata["key_points"] = " | ".join(memory.key_points)  # Chroma metadata must be flat values
    metadata["created_at"] = datetime.now().isoformat(timespec="seconds")
    return document_text, metadata


def get_collection():
    import chromadb
    from chromadb.utils import embedding_functions

    # Gemini's embeddings are reached through its OpenAI-compatible endpoint
    ef = embedding_functions.OpenAIEmbeddingFunction(api_key=llm_key(), model_name=EMBEDDING_MODEL,
                                                     api_base=LLM_BASE_URL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME, embedding_function=ef)


def write_episodic_memory(history: list[tuple[str, str]], collection=None) -> str | None:
    """Summarise a finished chat into a journal page and file it. Returns the memory id."""
    if not any(role == "assistant" for role, _ in history):
        return None
    prompt = EPISODIC_MEMORY_PROMPT_TEMPLATE.format(conversation=history_to_string(history))
    resp = get_client().chat.completions.parse(
        model=MODEL_SUMMARY, messages=[{"role": "user", "content": prompt}],
        response_format=EpisodicMemory)
    memory = resp.choices[0].message.parsed
    if memory is None:
        print("[journal page not written: the model returned no valid memory]")
        return None
    document_text, metadata = memory_to_document(memory)
    memory_id = f"memory_{datetime.now():%Y%m%d_%H%M%S_%f}"
    (collection or get_collection()).add(ids=[memory_id], documents=[document_text],
                                         metadatas=[metadata])
    print(f"\n[journal page filed as {memory_id}]\n{json.dumps(memory.model_dump(), indent=2)}")
    return memory_id


def recall(question: str, collection=None, n_results: int = 3) -> list[str]:
    collection = collection or get_collection()
    if collection.count() == 0:
        return []
    recap = collection.query(query_texts=[question], n_results=min(n_results, collection.count()))
    return recap["documents"][0]  # [question 0] -> matches, best first


def with_memory(question: str, pages: list[str]) -> str:
    if not pages:
        return question
    return question + MEMORY_MARKER + (
        " Use them as guidance, not strict rules (today's market may differ):\n"
        + "\n---\n".join(pages)
        + "\nArgue your proposition using past rebalancing episodes if relevant."
    )


# ---------- Runners ----------

async def _scripted_demo() -> None:
    """The exact course scenario: Day 1 (two turns, same notepad) -> journal -> Day 2 recall."""
    agent = build_portfolio_agent()
    collection = get_collection()
    session = SQLiteSession(f"demo_day1_{datetime.now():%Y%m%d_%H%M%S}", str(SESSIONS_DB))

    turn1 = """Current Market Situation:
- Volatility Regime: high
- Market Trend: declining
Current Portfolio State:
- Portfolio: {"stocks": 0.70, "bonds": 0.20, "cash": 0.10}
What rebalancing action should I take?"""
    turn2 = "In this case, I'm reducing equity to 50%, increasing bonds to 35% and cash to 15%."
    for q in (turn1, turn2):
        out = await Runner.run(agent, q, session=session)
        print(f"\n>>> USER: {q}\n<<< AGENT: {out.final_output}")

    write_episodic_memory(await read_history(session), collection)

    day2 = """Current Market Situation:
- Volatility Regime: low
- Market Trend: bullish
What rebalancing action should I take?"""
    pages = recall(day2, collection)
    session2 = SQLiteSession(f"demo_day2_{datetime.now():%Y%m%d_%H%M%S}", str(SESSIONS_DB))
    out = await Runner.run(agent, with_memory(day2, pages), session=session2)
    print(f"\n=== DAY 2 (new empty session, {len(pages)} memory page(s) recalled) ===")
    print(f">>> USER: {day2}\n<<< AGENT: {out.final_output}")


async def _chat(session_id: str | None) -> None:
    """Interactive chat. Memory is recalled on the first message and written on exit."""
    agent = build_portfolio_agent()
    collection = get_collection()
    session_id = session_id or f"chat_{datetime.now():%Y%m%d_%H%M%S}"
    session = SQLiteSession(session_id, str(SESSIONS_DB))
    resumed = bool(await session.get_items())
    print(f"Session '{session_id}' ({'resumed' if resumed else 'new'}). Type 'exit' to finish.")

    first = not resumed
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        if first:
            pages = recall(q, collection)
            if pages:
                print(f"[recalled {len(pages)} past episode(s)]")
            q = with_memory(q, pages)
            first = False
        out = await Runner.run(agent, q, session=session)
        print(f"\nAgent: {out.final_output}")

    write_episodic_memory(await read_history(session), collection)


def run_demo() -> None:
    setup_agents()
    asyncio.run(_scripted_demo())


def run_chat(session_id: str | None = None) -> None:
    setup_agents()
    asyncio.run(_chat(session_id))
