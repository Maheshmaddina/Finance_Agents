# AI Agents in Finance — Future with Shivank Bank

Working builds of every lab in the course page
[AI Agents in Finance, Sections 1 & 2](https://shrijankumar1-maker.github.io/finance-section-1/).
Your role: you joined the AI team at a (made-up) bank, and each lab is a tool a colleague asked for.

> For learning only. Nothing here is investment advice.

## The six requested tools (+ demos)

| Request | Lab | Command | File |
|---|---|---|---|
| Earnings call → fixed JSON card for the PM dashboard | 1A | `python main.py earnings [--compare]` | [lab1a_earnings.py](finance_agents/labs/lab1a_earnings.py) |
| "What is NVIDIA trading at?" with real data | 1C | `python main.py price "..."` | [lab1bc_price.py](finance_agents/labs/lab1bc_price.py) |
| Apple's profitability / valuation / liquidity ratios | 2A | `python main.py ratios` | [lab2a_ratios.py](finance_agents/labs/lab2a_ratios.py) |
| Rebalancing assistant that remembers (working + episodic memory) | 2B+2C | `python main.py rebalance-demo` / `rebalance` | [lab2bc_rebalancing.py](finance_agents/labs/lab2bc_rebalancing.py) |
| Reuters-only news, last 2 days, every item linked, checked by a 2nd agent | 2E | `python main.py news --region Eurozone --items 5` | [lab2e_news.py](finance_agents/labs/lab2e_news.py) |
| Buy / Hold / Sell memo from 3 agents passing a JSON baton | 2F | `python main.py bhs --company Apple --ticker AAPL` | [lab2f_buy_hold_sell.py](finance_agents/labs/lab2f_buy_hold_sell.py) |
| Demos: knowledge cutoff (1B), MA20 trend, self-consistency vote, ReAct (2D) | — | `cutoff`, `trend AAPL`, `vote`, `react` | [reasoning.py](finance_agents/labs/reasoning.py) |

## Web page

```bash
.venv/bin/streamlit run app.py      # opens http://localhost:8501
```

One page per tool in the sidebar: Earnings card, Live price, Ratios, Rebalancing chat,
Verified news, Buy / Hold / Sell. Each page has an "Agent log" box showing the checks and retries.

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate   # or: python3.12 -m venv .venv
pip install -r requirements.txt
cp .env.example .env        # then add GOOGLE_API_KEY and FINANCIAL_MODELING_PREP_API_KEY
python main.py --help
pytest -q                   # offline tests, no keys needed
```

### Free keys you need

| Key | Needed for | Where to get it (free) |
|---|---|---|
| `GOOGLE_API_KEY` | every lab except `trend` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) → sign in with Google → **Create API key** |
| `FINANCIAL_MODELING_PREP_API_KEY` | `ratios`, `bhs` | [Register](https://site.financialmodelingprep.com/register) → verify email → copy the key from the [dashboard](https://site.financialmodelingprep.com/developer/docs/dashboard) |

The default provider is **Google Gemini** (`LLM_PROVIDER=gemini`, model `gemini-3.5-flash-lite`),
which has a free tier. To use OpenAI as in the course, set `LLM_PROVIDER=openai` and `OPENAI_API_KEY`.
Any other OpenAI-compatible server works with `LLM_PROVIDER=custom`.

Free-tier notes:
- `gemini-3.5-flash-lite` allows 15 requests/minute. The app paces itself to 12/minute (`LLM_RPM`)
  and automatically waits out "per-minute" and "server busy" errors — you may see `[pacing]` lines.
- "Flash" (non-lite) models allow only ~20 requests/day on the free tier, so they're not the default.
  You can still try one for a single lab: `MODEL_AGENT=gemini-3.5-flash python main.py bhs`.
- If a daily cap is reached, the app stops with a message saying when it resets.
- The FMP free plan allows 250 calls/day and covers US stocks (e.g. AAPL); each `ratios`/`bhs`
  run uses a handful of calls.

Lab 2E news search: with OpenAI it uses OpenAI's hosted web search; with Gemini it uses a free
DuckDuckGo News search that only returns Reuters articles from the last two days (filtered in code).

## What goes beyond the course notebook

The course points out several weaknesses; this build fixes them:

- **Lab 1A — usable JSON, not just valid JSON.** Uses the strict typed contract and enforces it
  with Structured Outputs (`revenue_usd` is always a number). Gemini/Llama output is validated
  against the same contract and reported as OK or BROKEN.
- **Lab 1C / 2A / 2F — tool guardrail.** Tickers are checked (format + optional restricted list
  `RESTRICTED_TICKERS`) before any data API call. Agents are told today's year.
- **Lab 2A — less shallow interpretation.** The agent compares ratios with prior years and adds
  analyst context (current ratio < 1, buyback-inflated ROE).
- **Lab 2B/2C — production-ready memory.** Sessions are saved to disk; journal pages get unique ids;
  all top-3 matches are recalled; pages are written automatically when a chat ends; past memories
  aren't re-saved as new facts.
- **Lab 2E — tighter verification.** Date window recomputed each run; retry limit counts total
  searches exactly; a code check counts Reuters links so the LLM judge can't pass missing links.
- **Lab 2F — auditable memo.** Typed outputs between agents (no broken JSON); every number the
  retrieval agent passes on is re-checked against the data source; every risk must cite a supplied
  ratio (fixes the "market competition" leak); stance and 3+3 reasons/watchlist are enforced.
  A failed check gets one retry with feedback, then stops.

## Layout

```
main.py                      command-line entry point
finance_agents/
  config.py                  keys, models, paths, restricted tickers
  guardrails.py              ticker check, link counter
  tools/market.py            live price, MA20 trend (yfinance)
  tools/fundamentals.py      ratio families (Finance Toolkit + FMP)
  data/transcripts.py        made-up Apple earnings call
  labs/                      one file per lab
tests/test_offline.py        21 offline tests
data/                        created at runtime: sessions DB, Chroma memory
```
