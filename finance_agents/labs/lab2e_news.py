"""Lab 2E · Verified news agent (evaluator–optimizer loop).

"Latest 5 Eurozone financial news items. Reuters only, last two days, a link for every item."

searcher (web search) -> evaluator (LLM judge with a fixed form) + a plain-code link count
-> retry with feedback, up to a hard limit.

Fixes vs the course notebook:
- dates are computed each run (no stale "last two days" window if the app stays open)
- the retry limit counts total searches exactly (the course's `>` allowed one extra run)
- a code check counts Reuters links, so the LLM judge can't pass a summary with missing links
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from agents import Agent, ItemHelpers, Runner, TResponseInputItem, WebSearchTool, function_tool

from ..config import MODEL_NEWS, PROVIDER, setup_agents
from ..guardrails import count_links
from ..tools.news import search_news_broad

SOURCE = "reuters.com"


def _search_tool(two_days_ago: str):
    """OpenAI: its hosted web search (as in the course). Others: free Reuters news search."""
    if PROVIDER == "openai":
        return WebSearchTool()
    since = datetime.strptime(two_days_ago, "%Y-%m-%d").date()

    @function_tool
    def search_reuters_news(topic: str) -> list[dict]:
        """Search recent Reuters articles on a region or topic (covers economy, markets, central
        bank and companies in one call). Returns title, published date, url and snippet.
        Only Reuters articles inside the allowed date window are returned.

        Args:
            topic: Region or topic, e.g. "euro zone", "ECB", "European banks".
        """
        return search_news_broad(topic, SOURCE, since)

    return search_reuters_news


@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["successful", "unsuccessful"]


def build_agents(today: str, two_days_ago: str) -> tuple[Agent, Agent]:
    searcher = Agent(
        name="web_news_searcher",
        model=MODEL_NEWS,
        instructions=(
            "You are a research news assistant specialized in Finance or any news that can have an impact "
            "on the financial health of a given company or a region. "
            "You search the web for a given region to collect the most important and recent news. "
            "Produce a concise summary of the results with citations. The summary must be "
            "2-3 paragraphs and less than 300 words. Capture the main points. Write succinctly. "
            f"Get only the latest news between 2 days ago {two_days_ago} and today {today}. "
            f"Use only this resource: https://www.{SOURCE}/. "
            "For each item you are providing, specify the date on which the article was published. "
            "For each item you are providing, specify the complete external link to the article. "
            "This is important. Otherwise, don't give me a summary, if you don't provide a date and an external link. "
            "If you don't give a source link, it means that you are hallucinating. "
            "Call the search tool at most 3 times in total, then write the summary with what you found. "
            "Only use articles your search tool actually returned, with their exact URLs. "
        ),
        tools=[_search_tool(two_days_ago)],
    )
    evaluator = Agent(
        name="news_evaluator",
        model=MODEL_NEWS,
        instructions=(
            "You will be given a user request and a financial news summary for a region (Euro, US, Asia) "
            "and you are asked to evaluate the completeness of external source links. "
            f"Check whether the news summary is supported by external article links from https://www.{SOURCE}/. "
            f"Check whether the news was published between 2 days ago {two_days_ago} and today {today}. "
            "For each item of news, the date of the news must be explicitly provided. "
            "Check whether the news summary corresponds to the region or regions requested by the user. "
            "If the user requested 'n' news items, each item must be supported by an external source link, "
            "so you need to ensure that there are 'n' source links. "
            "External links are required for the user to verify the accuracy of the news. "
            "In feedback, list exactly which items fail and why."
        ),
        output_type=EvaluationFeedback,
    )
    return searcher, evaluator


async def _loop(request: str, n_items: int, max_attempts: int) -> tuple[str, bool]:
    now = datetime.now()
    today, two_days_ago = now.strftime("%Y-%m-%d"), (now - timedelta(days=2)).strftime("%Y-%m-%d")
    searcher, evaluator = build_agents(today, two_days_ago)

    input_items: list[TResponseInputItem] = [{"content": request, "role": "user"}]
    latest_outline = ""
    for attempt in range(1, max_attempts + 1):
        search = await Runner.run(searcher, input_items, max_turns=15)
        input_items = search.to_input_list()                       # whole chat so far
        latest_outline = ItemHelpers.text_message_outputs(search.new_items)  # just the new text

        judged = await Runner.run(evaluator,
                                  f"User request: {request}\n\nNews summary:\n{latest_outline}")
        result: EvaluationFeedback = judged.final_output

        links = count_links(latest_outline, SOURCE)
        if links < n_items:  # code check overrides the judge
            result = EvaluationFeedback(
                feedback=f"Only {links} distinct {SOURCE} links for {n_items} items. {result.feedback}",
                score="unsuccessful")

        print(f"[attempt {attempt}/{max_attempts}] {result.score} - {links} {SOURCE} link(s)")
        if result.score == "successful":
            return latest_outline, True

        input_items.append({"role": "user", "content":
                            f"Feedback: {result.feedback}\nRewrite the summary. For each item in the news "
                            f"summary, provide its publication date and the full external {SOURCE} link."})
    return latest_outline, False


def run(region: str = "Eurozone", n_items: int = 5, max_attempts: int = 4) -> str:
    setup_agents()
    request = (f"Provide the latest {n_items} financial news items for the {region} and specify "
               f"the source link and publication date for each one.")
    outline, verified = asyncio.run(_loop(request, n_items, max_attempts))
    print("\n" + ("VERIFIED" if verified else
                  f"NOT VERIFIED after {max_attempts} attempts - do not trade on this") + "\n")
    print(outline)
    return outline
