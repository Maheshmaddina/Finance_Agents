"""Free news search (DuckDuckGo News via the `ddgs` package, no API key).

Used by Lab 2E when the provider isn't OpenAI (OpenAI's hosted WebSearchTool only runs there).
Source and date window are enforced in code, so the agent can only see Reuters articles
published inside the window — it can't drift to other sites or old stories.
"""
from __future__ import annotations

from datetime import date, datetime


def filter_articles(results: list[dict], domain: str, since: date) -> list[dict]:
    """Keep articles from `domain` published on or after `since`, de-duplicated by URL."""
    seen, kept = set(), []
    for r in results:
        url = r.get("url") or ""
        if domain not in url or url in seen:
            continue
        try:
            published = datetime.fromisoformat(r.get("date", "").replace("Z", "+00:00")).date()
        except ValueError:
            continue  # no reliable date -> can't verify it -> drop it
        if published < since:
            continue
        seen.add(url)
        kept.append({"title": r.get("title", ""), "published": str(published), "url": url,
                     "snippet": (r.get("body") or "")[:400]})
    return kept


def search_news_broad(topic: str, domain: str, since: date) -> list[dict]:
    """One tool call = several related searches, so a small model needs only 1-2 calls."""
    seen, merged = set(), []
    for q in (topic, f"{topic} economy", f"{topic} markets stocks", f"{topic} central bank rates",
              f"{topic} banks companies"):
        for art in search_news(q, domain, since):
            if art["url"] not in seen:
                seen.add(art["url"])
                merged.append(art)
    return merged


def search_news(query: str, domain: str, since: date, max_results: int = 25) -> list[dict]:
    from ddgs import DDGS

    # "Reuters <topic>" respects the time filter; "site:" queries return stale dates, so avoid them.
    # Two regions return different articles; merging them roughly doubles Reuters coverage.
    q = f"Reuters {query}"
    raw: list[dict] = []
    for region in ("wt-wt", "us-en"):
        try:
            raw += DDGS().news(q, region=region, timelimit="w", max_results=max_results)
        except Exception as exc:  # the free backend fails now and then; keep what we have
            print(f"[news search] '{q}' ({region}) failed: {exc}")
    return filter_articles(raw, domain, since)
