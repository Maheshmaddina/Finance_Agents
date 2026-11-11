"""Web page for the six bank tools.  Run:  .venv/bin/streamlit run app.py

Each sidebar page is one lab from the course. For learning only — not investment advice.
"""
from __future__ import annotations

import asyncio
import contextlib
import html
import io
import threading
from datetime import datetime

import altair as alt
import openai
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Agents in Finance · Future with Shivank Bank",
                   page_icon="🏦", layout="wide")

CHART_COLOR = "#0D9488"  # validated: lightness, chroma and contrast pass on the light surface

# ---------- styling ----------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3 { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
:root { --navy:#0B1F3A; --teal:#0F766E; --teal2:#14B8A6; --mint:#5EEAD4; --ink:#0F172A;
        --muted:#64748B; --line:#E2E8F0; }

/* soft page backdrop */
.stApp { background:
  radial-gradient(900px 500px at 100% 0%, rgba(20,184,166,.08), transparent 60%),
  radial-gradient(700px 400px at 0% 100%, rgba(59,130,246,.06), transparent 60%), #F5F7FB; }

@keyframes fadeUp { from { opacity:0; transform: translateY(10px);} to { opacity:1; transform:none;} }
@keyframes floaty { 0%,100% { transform: translate(0,0);} 50% { transform: translate(-18px,12px);} }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(52,211,153,.6);} 70% { box-shadow: 0 0 0 8px rgba(52,211,153,0);} 100% { box-shadow:0 0 0 0 rgba(52,211,153,0);} }

/* sidebar: brand block above the page list */
[data-testid="stSidebarContent"] { display: flex; flex-direction: column;
  background: linear-gradient(180deg, #0B1F3A 0%, #0A1A31 100%); }
[data-testid="stSidebarHeader"] { order: -2; height: .5rem; min-height: 0; padding: 0; }
[data-testid="stSidebarUserContent"] { order: -1; padding-top: 1.2rem; padding-bottom: 0; }
[data-testid="stSidebarNavSeparator"] { display: none; }
[data-testid="stSidebarNav"] { margin-top: .9rem; }
[data-testid="stSidebarNav"] a { border-radius: 10px; transition: background .15s ease, transform .15s ease; }
[data-testid="stSidebarNav"] a:hover { background: rgba(45,212,191,.10); transform: translateX(2px); }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: linear-gradient(90deg, rgba(20,184,166,.28), rgba(20,184,166,.06));
  box-shadow: inset 3px 0 0 #2DD4BF; }
[data-testid="stNavSectionHeader"] { color:#64748B !important; text-transform: uppercase; letter-spacing:.08em; font-size:.72rem !important; font-weight:700; }

/* layout */
[data-testid="stMainBlockContainer"] { padding-top: 2rem; max-width: 1220px; }
[data-testid="stHeader"] { background: transparent; }

/* hero banners */
.hero {
  background: radial-gradient(900px 380px at 88% -30%, rgba(45,212,191,.45), transparent 60%),
              radial-gradient(500px 300px at 10% 130%, rgba(59,130,246,.35), transparent 60%),
              linear-gradient(135deg, #0B1F3A 0%, #12325E 55%, #0F766E 100%);
  color: #F8FAFC; border-radius: 24px; padding: 2.8rem 2.8rem; margin-bottom: 1.8rem;
  box-shadow: 0 30px 60px -30px rgba(11,31,58,.65); position: relative; overflow: hidden;
  animation: fadeUp .45s ease both; }
.hero::before { content:""; position:absolute; inset:0; pointer-events:none;
  background-image: linear-gradient(rgba(255,255,255,.06) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,.06) 1px, transparent 1px);
  background-size: 36px 36px; -webkit-mask-image: linear-gradient(90deg, transparent 20%, #000 75%);
  mask-image: linear-gradient(90deg, transparent 20%, #000 75%); }
.hero::after { content:""; position:absolute; width:260px; height:260px; right:-60px; top:-80px; border-radius:50%;
  background: radial-gradient(circle, rgba(94,234,212,.35), transparent 65%); animation: floaty 9s ease-in-out infinite; }
.hero > * { position: relative; z-index: 1; }
.hero h1 { color: #FFFFFF; font-size: 2.6rem; font-weight: 900; letter-spacing: -.03em;
           margin: .5rem 0 .6rem 0; line-height: 1.1; padding: 0; max-width: 820px; }
.hero h1 .grad { background: linear-gradient(90deg, #5EEAD4, #93C5FD); -webkit-background-clip: text;
  background-clip: text; color: transparent; }
.hero p { color: #CBD5E1; font-size: 1.06rem; max-width: 680px; margin: 0; line-height: 1.6; }
.hero .eyebrow { display: inline-flex; align-items:center; gap:.45rem; font-size: .74rem; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase; color: #5EEAD4; background: rgba(45,212,191,.12);
  border: 1px solid rgba(94,234,212,.35); padding: .3rem .8rem; border-radius: 999px; }
.hero .eyebrow .dot { width:7px; height:7px; border-radius:50%; background:#34D399; animation: pulse 2s infinite; }
.hero.small { padding: 1.9rem 2.2rem; }
.hero.small h1 { font-size: 2rem; }
.hero .icon { position: absolute; right: 2.4rem; top: 50%; transform: translateY(-50%); z-index:1;
  font-size: 4.2rem; width: 104px; height: 104px; display:flex; align-items:center; justify-content:center;
  border-radius: 28px; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14);
  backdrop-filter: blur(6px); box-shadow: 0 20px 40px -20px rgba(0,0,0,.5); }
.chips { display:flex; flex-wrap:wrap; gap:.5rem; margin-top: 1.1rem; }
.chip { font-size:.8rem; font-weight:600; color:#E2E8F0; background: rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.16); padding:.3rem .75rem; border-radius:999px; }
.cta { display:flex; gap:.7rem; margin-top:1.4rem; flex-wrap: wrap; }
.cta a { text-decoration:none !important; font-weight:700; font-size:.92rem; padding:.65rem 1.15rem; border-radius:12px;
  transition: transform .15s ease, box-shadow .15s ease; }
.cta a.primary { color:#052E2B !important; background: linear-gradient(135deg,#5EEAD4,#2DD4BF);
  box-shadow: 0 10px 24px -10px rgba(45,212,191,.8); }
.cta a.ghost { color:#F8FAFC !important; background: rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.2); }
.cta a:hover { transform: translateY(-2px); }

/* section titles */
.section { display:flex; align-items:center; gap:.6rem; font-size: .78rem; font-weight: 800; letter-spacing: .1em;
  text-transform: uppercase; color: #64748B; margin: 1.8rem 0 .8rem 0; }
.section::after { content:""; flex:1; height:1px; background: linear-gradient(90deg, #E2E8F0, transparent); }

/* bordered containers = cards */
[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255,255,255,.92); border-radius: 18px;
  border-color: #E8EDF4 !important;
  box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 12px 32px -18px rgba(15,23,42,.18); animation: fadeUp .4s ease both; }

/* market strip */
.mkt { display:grid; grid-template-columns: repeat(6, minmax(0,1fr)); gap: .85rem; animation: fadeUp .5s ease both; }
.mkt .card { background:#fff; border:1px solid #E8EDF4; border-radius:16px; padding:.9rem 1rem .6rem 1rem;
  box-shadow: 0 10px 26px -20px rgba(15,23,42,.35); transition: transform .15s ease, box-shadow .15s ease; }
.mkt .card:hover { transform: translateY(-3px); box-shadow: 0 18px 34px -20px rgba(15,23,42,.45); }
.mkt .top { display:flex; align-items:center; justify-content:space-between; }
.mkt .tkr { font-weight:800; font-size:.86rem; color:#0F172A; letter-spacing:.02em; }
.mkt .name { font-size:.72rem; color:#94A3B8; }
.mkt .px { font-weight:800; font-size:1.3rem; color:#0F172A; margin-top:.25rem; letter-spacing:-.01em; }
.pill { font-size:.74rem; font-weight:700; padding:.12rem .45rem; border-radius:999px; white-space:nowrap; }
.pill.up { color:#15803D; background:#DCFCE7; } .pill.down { color:#B91C1C; background:#FEE2E2; }
.mkt svg { display:block; width:100%; height:38px; margin-top:.35rem; }

/* tool cards */
.tools { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 1rem; }
a.tool { position:relative; display:block; text-decoration:none !important; background:#fff; border:1px solid #E8EDF4;
  border-radius:20px; padding:1.3rem 1.3rem 1.1rem 1.3rem; overflow:hidden; animation: fadeUp .5s ease both;
  box-shadow: 0 12px 30px -22px rgba(15,23,42,.35); transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease; }
a.tool::before { content:""; position:absolute; left:0; right:0; top:0; height:4px; background: var(--accent); }
a.tool:hover { transform: translateY(-5px); border-color: transparent;
  box-shadow: 0 26px 46px -24px rgba(15,23,42,.45), 0 0 0 1px var(--accent); }
a.tool .ico { width:50px; height:50px; border-radius:14px; display:flex; align-items:center; justify-content:center;
  font-size:1.5rem; background: var(--soft); margin-bottom:.9rem; }
a.tool .lab { display:inline-block; font-size:.7rem; font-weight:800; letter-spacing:.06em; text-transform:uppercase;
  color: var(--accent); margin-bottom:.25rem; }
a.tool .ttl { font-weight:800; font-size:1.08rem; color:#0F172A; margin-bottom:.3rem; }
a.tool .dsc { color:#475569; font-size:.9rem; line-height:1.5; min-height:2.7rem; }
a.tool .go { margin-top:.9rem; font-weight:700; font-size:.86rem; color: var(--accent); }
a.tool:hover .go { letter-spacing:.01em; }

/* features row */
.feats { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:1rem; }
.feat { background: linear-gradient(180deg,#FFFFFF, #F8FAFC); border:1px solid #E8EDF4; border-radius:18px; padding:1.2rem; }
.feat .n { width:34px; height:34px; border-radius:10px; display:flex; align-items:center; justify-content:center;
  font-weight:800; color:#fff; background: linear-gradient(135deg,#0F766E,#14B8A6); margin-bottom:.7rem; }
.feat b { color:#0F172A; display:block; margin-bottom:.2rem; } .feat span { color:#475569; font-size:.9rem; }

/* metrics */
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E8EDF4; border-radius: 16px;
  padding: 1rem 1.15rem; box-shadow: 0 10px 26px -20px rgba(15,23,42,.35); animation: fadeUp .4s ease both; }
[data-testid="stMetricLabel"] p { color: #64748B; font-weight: 700; font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; }
[data-testid="stMetricValue"] { font-weight: 800; color: #0F172A; font-size: 1.6rem; letter-spacing: -.01em; }
[data-testid="stMetricValue"] > div { overflow: visible; text-overflow: clip; }

/* buttons */
.stButton > button { font-weight: 700; padding: .55rem 1.25rem; border-radius: 12px; transition: transform .15s ease, box-shadow .15s ease; }
.stButton > button:hover { transform: translateY(-1px); }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #0F766E, #14B8A6); border: none; color: #fff;
  box-shadow: 0 10px 22px -10px rgba(15,118,110,.85); }
.stButton > button[kind="primary"]:hover { box-shadow: 0 14px 28px -10px rgba(15,118,110,.95); color:#fff; }

/* inputs */
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div { border-radius: 12px !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within { box-shadow: 0 0 0 3px rgba(20,184,166,.18); }

/* tabs as pills */
[data-testid="stTabs"] [role="tablist"] { gap:.35rem; background:#EAEFF5; padding:.3rem; border-radius:14px;
  width: fit-content; border: none; }
[data-testid="stTabs"] [role="tab"] { border-radius:10px; padding:.45rem 1rem; margin:0; border:none; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background:#fff !important;
  box-shadow: 0 4px 12px -6px rgba(15,23,42,.35) !important; }
[data-testid="stTabs"] [role="tab"]::after, [data-testid="stTabs"] [role="tab"] > div:last-child:empty { display:none !important; }
[data-testid="stTabs"] [role="tablist"] > div:not([role]) { display:none !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"], [data-testid="stTabs"] [data-baseweb="tab-border"],
[data-testid="stTabs"] [role="tablist"] > div:empty { display:none; }

/* expanders */
[data-testid="stExpander"] details { border-radius: 14px; border-color:#E8EDF4; background:#fff; }

/* chat */
[data-testid="stChatMessage"] { background:#fff; border:1px solid #E8EDF4; border-radius:18px; padding:1rem 1.1rem;
  box-shadow: 0 10px 26px -22px rgba(15,23,42,.35); animation: fadeUp .35s ease both; }

/* verdict badge */
.verdict { display: flex; align-items: center; gap: 1.4rem; padding: 1.6rem 1.8rem; border-radius: 22px;
  border: 1px solid; margin-bottom: 1rem; animation: fadeUp .4s ease both; position:relative; overflow:hidden; }
.verdict .stance { font-size: 2.8rem; font-weight: 900; letter-spacing: .02em; line-height: 1; }
.verdict .who { font-size: 1.15rem; font-weight: 800; color: #0F172A; }
.verdict .sub { color: #475569; font-size: .9rem; margin-top:.15rem; }
.verdict.BUY  { background: linear-gradient(135deg,#F0FDF4,#DCFCE7); border-color: #86EFAC; } .verdict.BUY .stance { color: #15803D; }
.verdict.HOLD { background: linear-gradient(135deg,#FFFBEB,#FEF3C7); border-color: #FCD34D; } .verdict.HOLD .stance { color: #B45309; }
.verdict.SELL { background: linear-gradient(135deg,#FEF2F2,#FEE2E2); border-color: #FCA5A5; } .verdict.SELL .stance { color: #B91C1C; }

/* status banners */
.status { padding: .9rem 1.15rem; border-radius: 14px; font-weight: 700; margin-bottom: .9rem; border: 1px solid; animation: fadeUp .4s ease both; }
.status.ok { background: linear-gradient(135deg,#F0FDF4,#DCFCE7); color: #166534; border-color: #BBF7D0; }
.status.warn { background: linear-gradient(135deg,#FFFBEB,#FEF3C7); color: #92400E; border-color: #FDE68A; }

/* empty states */
.empty { text-align:center; padding: 2.4rem 1rem; border:2px dashed #DCE3EC; border-radius:20px; background: rgba(255,255,255,.6);
  animation: fadeUp .4s ease both; }
.empty .big { font-size:2.4rem; margin-bottom:.4rem; }
.empty b { color:#0F172A; display:block; font-size:1.02rem; margin-bottom:.2rem; }
.empty span { color:#64748B; font-size:.92rem; }

/* sidebar brand */
.brand { display: flex; align-items: center; gap: .75rem; padding: .4rem 0 .8rem 0; }
.brand .logo { width: 42px; height: 42px; border-radius: 13px; display: flex; align-items: center;
  justify-content: center; font-size: 1.3rem; background: linear-gradient(135deg,#0F766E,#2DD4BF);
  box-shadow: 0 10px 22px -10px rgba(45,212,191,.8); }
.brand .name { font-weight: 800; color: #F8FAFC; line-height: 1.15; }
.brand .sub { font-size: .75rem; color: #94A3B8; }
.live { display:inline-flex; align-items:center; gap:.45rem; font-size:.72rem; font-weight:700; color:#A7F3D0;
  background: rgba(52,211,153,.10); border:1px solid rgba(52,211,153,.25); padding:.28rem .65rem; border-radius:999px; margin-bottom:.6rem; }
.live .dot { width:7px; height:7px; border-radius:50%; background:#34D399; animation: pulse 2s infinite; }

/* footer */
.footer { display:flex; justify-content:space-between; flex-wrap:wrap; gap:.5rem; color: #94A3B8; font-size: .8rem;
  margin-top: 3rem; padding-top: 1.1rem; border-top: 1px solid #E2E8F0; }

@media (max-width: 1100px) { .mkt { grid-template-columns: repeat(3, minmax(0,1fr)); }
  .tools, .feats { grid-template-columns: repeat(2, minmax(0,1fr)); } .hero .icon { display:none; } }
@media (max-width: 700px) { .tools, .feats, .mkt { grid-template-columns: 1fr; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def hero(icon: str, title: str, lab: str, desc: str, small: bool = True,
         chips: list[str] | None = None, cta: str = "") -> None:
    chip_html = ("<div class='chips'>" + "".join(f"<span class='chip'>{c}</span>" for c in chips)
                 + "</div>") if chips else ""
    st.markdown(
        f'<div class="hero {"small" if small else ""}"><span class="eyebrow"><span class="dot"></span>{lab}</span>'
        f'<h1>{title}</h1><p>{desc}</p>{chip_html}{cta}<div class="icon">{icon}</div></div>',
        unsafe_allow_html=True)


def empty(icon: str, title: str, text: str) -> None:
    st.markdown(f'<div class="empty"><div class="big">{icon}</div><b>{title}</b><span>{text}</span></div>',
                unsafe_allow_html=True)


def section(text: str) -> None:
    st.markdown(f'<div class="section">{text}</div>', unsafe_allow_html=True)


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items)


def footer() -> None:
    st.markdown('<div class="footer"><span>🏦 Future with Shivank Bank · AI Agents in Finance</span>'
                '<span>For learning only — not investment advice.</span></div>', unsafe_allow_html=True)


# ---------- one background event loop for all agent runs ----------
# The agents' HTTP client must stay on a single event loop across Streamlit reruns.
@st.cache_resource
def _loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    from finance_agents.config import setup_agents
    asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result()
    setup_agents()
    return loop


def run_async(coro):
    """Run a coroutine on the shared loop; returns (result, captured log text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = asyncio.run_coroutine_threadsafe(coro, _loop()).result()
    return result, buf.getvalue()


def show_log(log: str) -> None:
    if log.strip():
        with st.expander("🔍 Agent log — checks, retries, pacing"):
            st.code(log, language=None)


def friendly_errors(fn):
    """Show provider limits and missing keys as messages instead of stack traces."""
    try:
        fn()
    except openai.RateLimitError:
        st.error("The AI provider's free-tier limit was reached. Wait a minute (or until tomorrow "
                 "for the daily cap) and try again.", icon="⏳")
    except SystemExit as exc:  # raised by require_key()
        st.error(str(exc), icon="🔑")
    except Exception as exc:  # noqa: BLE001 — show any other failure to the user
        st.error(f"{type(exc).__name__}: {exc}", icon="⚠️")


@st.cache_data(ttl=300, show_spinner=False)
def market_snapshot(tickers: tuple[str, ...]) -> list[dict]:
    """Last close, day change and a 1-month close series per ticker (no AI, cached 5 minutes)."""
    import yfinance as yf
    try:  # one batched request instead of one per ticker
        closes = yf.download(list(tickers), period="1mo", progress=False, auto_adjust=True,
                             threads=False)["Close"]
    except Exception:  # noqa: BLE001 — a market-data hiccup shouldn't break the page
        return []
    rows = []
    for t in tickers:
        if t not in closes:
            continue
        s = closes[t].dropna()
        if len(s) >= 2:
            last, prev = float(s.iloc[-1]), float(s.iloc[-2])
            rows.append({"ticker": t, "price": last, "change": (last / prev - 1) * 100,
                         "series": [float(v) for v in s]})
    return rows


def sparkline(values: list[float], up: bool, w: int = 160, h: int = 38) -> str:
    """Inline SVG sparkline: one 2px line with a soft area fill (direction is also shown as text)."""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    pts = [(i * w / (len(values) - 1), h - 3 - (v - lo) / span * (h - 6)) for i, v in enumerate(values)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    color = "#16A34A" if up else "#DC2626"
    gid = f"g{abs(hash((tuple(values), up))) % 10**8}"
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0%" stop-color="{color}" stop-opacity=".22"/>'
            f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'<polygon points="0,{h} {line} {w},{h}" fill="url(#{gid})"/>'
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>')


# ---------- pages ----------

NAMES = {"AAPL": "Apple", "NVDA": "NVIDIA", "MSFT": "Microsoft", "GOOGL": "Alphabet",
         "AMZN": "Amazon", "TSLA": "Tesla"}

# icon, name, lab, description, url path, accent colour, soft background
TOOLS = [
    ("📄", "Earnings card", "Lab 1A", "Turns an earnings-call transcript into a fixed, typed card.",
     "earnings", "#0F766E", "#CCFBF1"),
    ("💹", "Live price", "Lab 1C", "Answers \"What is NVIDIA trading at?\" with real data.",
     "price", "#2563EB", "#DBEAFE"),
    ("📊", "Ratio retriever", "Lab 2A", "Profitability, valuation and liquidity ratios, charted and explained.",
     "ratios", "#7C3AED", "#EDE9FE"),
    ("⚖️", "Rebalancing chat", "Lab 2B + 2C", "Remembers the chat, and learns across sessions.",
     "rebalance", "#C2410C", "#FFEDD5"),
    ("📰", "Verified news", "Lab 2E", "Reuters only, last two days, every item linked and checked.",
     "news", "#BE185D", "#FCE7F3"),
    ("🧭", "Buy / Hold / Sell", "Lab 2F", "Three agents pass JSON to produce an investment memo.",
     "bhs", "#0369A1", "#E0F2FE"),
]


def page_home():
    from finance_agents.config import MODEL_AGENT, PROVIDER

    hero("🏦", 'Your AI desk for <span class="grad">markets, ratios and research</span>',
         "Future with Shivank Bank · AI team",
         "Six agents built for analysts, fund managers and advisors — every number comes from a "
         "live data source, and every answer is checked before you see it.", small=False,
         chips=["⚡ Live market data", "🛡️ Every output verified", f"🤖 {MODEL_AGENT}", "🆓 Free-tier ready"],
         cta='<div class="cta"><a class="primary" href="bhs" target="_self">Run Buy / Hold / Sell →</a>'
             '<a class="ghost" href="ratios" target="_self">Explore ratios</a>'
             '<a class="ghost" href="news" target="_self">Today\'s verified news</a></div>')

    section("Markets today")
    snap = market_snapshot(tuple(NAMES))
    if snap:
        cards = []
        for r in snap:
            up = r["change"] >= 0
            cards.append(
                f'<div class="card"><div class="top"><div><div class="tkr">{r["ticker"]}</div>'
                f'<div class="name">{NAMES.get(r["ticker"], "")}</div></div>'
                f'<span class="pill {"up" if up else "down"}">{"▲" if up else "▼"} {abs(r["change"]):.2f}%</span></div>'
                f'<div class="px">${r["price"]:,.2f}</div>{sparkline(r["series"], r["series"][-1] >= r["series"][0])}</div>')
        st.markdown(f'<div class="mkt">{"".join(cards)}</div>', unsafe_allow_html=True)
        st.caption("Last close and 1-month trend · Yahoo Finance · refreshed every 5 minutes")
    else:
        empty("📡", "Market data is unavailable right now", "Try again in a minute.")

    section("Tools")
    cards = "".join(
        f'<a class="tool" href="{path}" target="_self" style="--accent:{acc};--soft:{soft};'
        f'animation-delay:{i * 60}ms"><div class="ico">{icon}</div><div class="lab">{lab}</div>'
        f'<div class="ttl">{name}</div><div class="dsc">{desc}</div><div class="go">Open tool →</div></a>'
        for i, (icon, name, lab, desc, path, acc, soft) in enumerate(TOOLS))
    st.markdown(f'<div class="tools">{cards}</div>', unsafe_allow_html=True)

    section("How every tool works")
    steps = [("Tools fetch the facts", "Prices, ratios and news come from data APIs — never from model memory."),
             ("Agents reason", "The model reads the facts, explains them and proposes a view."),
             ("Code checks the output", "Typed contracts, source re-checks and retry limits keep it auditable.")]
    st.markdown('<div class="feats">' + "".join(
        f'<div class="feat"><div class="n">{i}</div><b>{b}</b><span>{t}</span></div>'
        for i, (b, t) in enumerate(steps, 1)) + '</div>', unsafe_allow_html=True)

    st.caption(f"Running on {PROVIDER} · {MODEL_AGENT}")
    footer()


def page_earnings():
    from finance_agents.data.transcripts import transcript_apple_earnings_call
    from finance_agents.labs.lab1a_earnings import summarise

    hero("📄", "Earnings-call summariser", "Lab 1A",
         "Every call becomes the same card for the PM dashboard — numbers arrive as numbers, in a known unit.",
         chips=["🧾 Structured outputs", "🔢 Typed numbers", "⚡ 1 AI request"])

    with st.container(border=True):
        text = st.text_area("Transcript", transcript_apple_earnings_call, height=220,
                            help="A made-up Apple call is pre-filled. Paste any transcript.")
        go_btn = st.button("✨ Summarise call", type="primary")

    def go():
        with st.spinner("Reading the call..."):
            card = summarise(text)
        section(f"{html.escape(card.company)} · {html.escape(card.quarter)}")
        c = st.columns(4)
        c[0].metric("Revenue", f"${card.revenue_usd / 1e9:,.1f}B")
        c[1].metric("Growth YoY", f"{card.revenue_growth_yoy_pct:+.1f}%")
        c[2].metric("EPS", f"${card.eps_usd:.2f}")
        c[3].metric("Outlook", {"positive": "▲ Positive", "neutral": "● Neutral",
                                "negative": "▼ Negative"}[card.outlook])
        with st.container(border=True):
            st.markdown(f"**🎯 Guidance**  \n{card.guidance}")
        left, right = st.columns(2, gap="medium")
        with left.container(border=True):
            st.markdown("**✅ Highlights**\n" + bullets(card.highlights))
        with right.container(border=True):
            st.markdown("**⚠️ Risks**\n" + bullets(card.risks))
        with st.expander("{ } Raw JSON for the dashboard"):
            st.json(card.model_dump())

    if go_btn:
        friendly_errors(go)
    else:
        empty("✨", "Your card will appear here", "Press Summarise call to turn the transcript into a dashboard card.")
    footer()


def page_price():
    from agents import Runner

    from finance_agents.labs.lab1bc_price import build_price_agent
    from finance_agents.tools.market import fetch_latest_price, fetch_trend

    hero("💹", "Live price", "Lab 1C",
         "A plain model can't know today's price. This agent calls a price tool — and tells you the date.",
         chips=["📡 Yahoo Finance", "🕒 Timestamped", "📈 20-day trend"])

    left, right = st.columns([1, 1.4], gap="large")
    with left.container(border=True):
        st.markdown("**⚡ Quick quote** · no AI")
        ticker = st.text_input("Ticker", "NVDA").strip().upper()
        quote_btn = st.button("Get quote", type="primary")
    with right.container(border=True):
        st.markdown("**🤖 Ask the agent**")
        q = st.text_input("Question", "What is today's date, and what is the current stock price of Nvidia?")
        ask_btn = st.button("Ask agent")

    def quote():
        p = fetch_latest_price(ticker)
        if "error" in p:
            st.error(p["error"])
            return
        t = fetch_trend(ticker)
        section(f"{p['ticker']} · as of {p['datetime']}")
        m = st.columns(3)
        m[0].metric("Last price", f"${p['price']:,.2f}")
        m[1].metric("20-day average", f"${t['ma20_today']:,.2f}",
                    f"{t['ma20_today'] - t['ma20_week_ago']:+.2f} vs a week ago")
        m[2].metric("Trend", ("▲ " if t["trend"] == "Uptrend" else "▼ ") + t["trend"])

    def ask():
        with st.spinner("Agent is calling the price tool..."):
            res, log = run_async(Runner.run(build_price_agent(), q))
        section("Agent answer")
        with st.container(border=True):
            st.markdown(res.final_output)
        show_log(log)

    if quote_btn:
        friendly_errors(quote)
    if ask_btn:
        friendly_errors(ask)
    if not (quote_btn or ask_btn):
        empty("💹", "Pick a ticker or ask a question", "Quick quote uses no AI; the agent answers in plain English.")
    footer()


PERCENT_RATIOS = {"Gross Margin", "Operating Margin", "Net Profit Margin", "Return on Equity",
                  "Return on Assets", "Operating Cash Flow to Sales Ratio"}


def ratio_chart(data: dict, ratio: str) -> alt.Chart:
    """One ratio over the years: single series, 2px line, 8px+ markers, hover tooltip."""
    df = pd.DataFrame({"Fiscal year": list(data[ratio].keys()),
                       "Value": list(data[ratio].values())}).dropna()
    # Units: margins/returns are fractions -> percent; money amounts -> $ with B/M; multiples -> 2 dp
    if ratio in PERCENT_RATIOS:
        axis_fmt, label_fmt = ".0%", ".1%"
    elif df["Value"].abs().max() >= 1e6:
        axis_fmt, label_fmt = "$.2~s", "$.3~s"
    else:
        axis_fmt, label_fmt = ",.1f", ",.2f"
    base = alt.Chart(df).encode(
        x=alt.X("Fiscal year:O", axis=alt.Axis(labelAngle=0, title=None, domainColor="#CBD5E1",
                                                tickColor="#CBD5E1", labelColor="#475569")),
        y=alt.Y("Value:Q", scale=alt.Scale(zero=False),
                axis=alt.Axis(title=None, gridColor="#EEF2F7", domain=False, ticks=False,
                              labelColor="#475569", format=axis_fmt, tickCount=5)),
        tooltip=[alt.Tooltip("Fiscal year:O"), alt.Tooltip("Value:Q", title=ratio, format=label_fmt)],
    )
    line = base.mark_line(color=CHART_COLOR, strokeWidth=2)
    points = base.mark_point(color=CHART_COLOR, filled=True, size=90, stroke="#FFFFFF", strokeWidth=2)
    hit = base.mark_point(size=600, opacity=0)  # bigger hover target than the mark
    last = base.transform_window(rank="rank()", sort=[alt.SortField("Fiscal year", order="descending")]) \
        .transform_filter("datum.rank == 1") \
        .mark_text(align="left", dx=10, color="#0F172A", fontWeight=600) \
        .encode(text=alt.Text("Value:Q", format=label_fmt))
    return (line + points + hit + last).properties(title=alt.Title(ratio, anchor="start",
                                                   color="#0F172A", fontSize=14), height=280) \
        .configure_view(stroke=None).configure(background="#FFFFFF", font="Inter")


def page_ratios():
    from agents import Runner

    from finance_agents.labs.lab2a_ratios import DEFAULT_QUESTION, build_agent
    from finance_agents.tools.fundamentals import fetch_fundamentals

    hero("📊", "Ratio retriever", "Lab 2A",
         "Profitability, valuation and liquidity ratios from Financial Modeling Prep — then explained.",
         chips=["🏛️ Financial Modeling Prep", "📊 18 ratios", "🗓️ 5 fiscal years"])

    with st.container(border=True):
        c1, c2 = st.columns([1, 3])
        ticker = c1.text_input("Ticker", "AAPL", help="US stocks on the free plan").strip().upper()
        c2.markdown("<br>", unsafe_allow_html=True)
        load_btn = c2.button("Load ratios", type="primary")
    if load_btn:
        st.session_state.ratios_ticker = ticker

    tab1, tab2 = st.tabs(["📈 Ratios & trends", "🤖 Ask the agent"])
    with tab1:
        tkr = st.session_state.get("ratios_ticker")
        if not tkr:
            empty("📊", "Load a ticker to see its ratios", "Charts show how each ratio moved over five fiscal years.")
        else:
            def load():
                with st.spinner("Fetching statements and calculating ratios..."):
                    data = fetch_fundamentals(tkr)
                families = {"Profitability": "profitability_ratios", "Valuation": "valuation_ratios",
                            "Liquidity": "liquidity_ratios"}
                fam = st.segmented_control("Family", list(families), default="Profitability",
                                           label_visibility="collapsed") or "Profitability"
                fdata = data[families[fam]]
                ratio = st.selectbox("Ratio", list(fdata))
                with st.container(border=True):
                    st.altair_chart(ratio_chart(fdata, ratio), width="stretch")
                with st.expander("Table view — all ratios in this family"):
                    st.dataframe(pd.DataFrame(fdata).T, width="stretch")
            friendly_errors(load)
    with tab2:
        q = st.text_area("Question", DEFAULT_QUESTION.replace("Apple", ticker), height=110)
        if st.button("Ask agent", type="primary"):
            def ask():
                with st.spinner("Agent is fetching and explaining..."):
                    res, log = run_async(Runner.run(build_agent(), q))
                with st.container(border=True):
                    st.markdown(res.final_output)
                show_log(log)
            friendly_errors(ask)
    footer()


def page_rebalance():
    from agents import Runner, SQLiteSession

    from finance_agents.config import SESSIONS_DB
    from finance_agents.labs.lab2bc_rebalancing import (build_portfolio_agent, get_collection,
                                                        read_history, recall, with_memory,
                                                        write_episodic_memory)

    hero("⚖️", "Rebalancing assistant", "Lab 2B + 2C",
         "Remembers this conversation, and learns from past ones — used as guidance, not rules.",
         chips=["🗒️ Working memory", "🧠 Episodic memory", "🔍 Recall by meaning"])

    ss = st.session_state
    if "rebal_id" not in ss:
        ss.rebal_id = f"web_{datetime.now():%Y%m%d_%H%M%S}"
        ss.rebal_msgs = []
    session = SQLiteSession(ss.rebal_id, str(SESSIONS_DB))

    top = st.columns([3, 1])
    try:
        top[0].caption(f"🧠 Past episodes on file: **{get_collection().count()}**")
    except SystemExit:
        pass
    if top[1].button("💾 End chat & save memory", width="stretch"):
        def save():
            history, _ = run_async(read_history(session))
            with st.spinner("Writing the journal page..."):
                mid = write_episodic_memory(history)
            if mid:
                st.success(f"Saved as {mid}. Your next chat will recall it.", icon="🧠")
            else:
                st.info("Nothing to save yet.")
            ss.pop("rebal_id")
        friendly_errors(save)

    if not ss.rebal_msgs:
        with st.container(border=True):
            st.markdown("**Try this flow**\n"
                        "1. *Volatility is high and the market is declining. My portfolio is 70% stocks, "
                        "20% bonds, 10% cash. What should I do?*\n"
                        "2. Reply with your own tweak, e.g. *I'll go 50/35/15.*\n"
                        "3. Press **End chat & save memory**, then start a new chat with a calm, rising "
                        "market — watch it recall the past episode.")
    for role, text in ss.rebal_msgs:
        st.chat_message(role, avatar="🧑‍💼" if role == "user" else "🤖").markdown(text)

    if prompt := st.chat_input("Describe the market and your portfolio..."):
        st.chat_message("user", avatar="🧑‍💼").markdown(prompt)
        ss.rebal_msgs.append(("user", prompt))

        def reply():
            q = prompt
            if len(ss.rebal_msgs) == 1:  # first message: look up past episodes
                pages = recall(prompt)
                if pages:
                    st.caption(f"🧠 Recalled {len(pages)} past episode(s) as guidance")
                q = with_memory(prompt, pages)
            with st.spinner("Thinking..."):
                res, _ = run_async(Runner.run(build_portfolio_agent(), q, session=session))
            st.chat_message("assistant", avatar="🤖").markdown(res.final_output)
            ss.rebal_msgs.append(("assistant", res.final_output))
        friendly_errors(reply)


def page_news():
    from finance_agents.labs.lab2e_news import _loop as news_loop

    hero("📰", "Verified news", "Lab 2E",
         "Reuters only, last two days, a date and link for every item — checked by a second agent.",
         chips=["📰 Reuters only", "🗓️ Last 2 days", "✅ Second-agent check", "🔁 Retry limit"])

    with st.container(border=True):
        c = st.columns([1.2, 1, 1, 1])
        region = c[0].selectbox("Region", ["Eurozone", "US", "Asia", "UK", "India"])
        n = c[1].slider("Items", 1, 8, 5)
        attempts = c[2].slider("Max attempts", 1, 5, 4)
        c[3].markdown("<br>", unsafe_allow_html=True)
        go_btn = c[3].button("Get news", type="primary", width="stretch")

    def go():
        req = (f"Provide the latest {n} financial news items for the {region} and specify "
               f"the source link and publication date for each one.")
        with st.spinner("Searching Reuters and checking every link... (about a minute)"):
            (outline, ok), log = run_async(news_loop(req, n, attempts))
        if ok:
            st.markdown('<div class="status ok">✅ Verified — Reuters sources, dated, inside the last two days</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status warn">⚠️ Not verified after {attempts} attempts — do not trade on this</div>',
                        unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(outline)
        show_log(log)

    if go_btn:
        friendly_errors(go)
    else:
        empty("📰", "Choose a region and press Get news", "Every item will carry a Reuters link and its publication date.")
    footer()


def page_bhs():
    from finance_agents.labs.lab2f_buy_hold_sell import pipeline

    hero("🧭", "Buy / Hold / Sell", "Lab 2F",
         "Retrieval → Analysis → Decision. Every number is re-checked against the source, "
         "and every risk must cite a ratio.",
         chips=["1️⃣ Retrieval", "2️⃣ Analysis", "3️⃣ Decision", "🛡️ Source re-check"])

    with st.container(border=True):
        c = st.columns([1.5, 1, 1])
        company = c[0].text_input("Company", "Apple")
        ticker = c[1].text_input("Ticker", "AAPL").strip().upper()
        c[2].markdown("<br>", unsafe_allow_html=True)
        go_btn = c[2].button("Run pipeline", type="primary", width="stretch")

    def go():
        with st.status("Three agents at work...", expanded=True) as status:
            st.write("1 · Retrieval agent fetches ratios · 2 · Analysis agent rates them · "
                     "3 · Decision agent takes a stance")
            (view, verdict), log = run_async(pipeline(company, ticker))
            status.update(label="Pipeline complete — all checks passed", state="complete", expanded=False)

        icon = {"BUY": "▲", "HOLD": "■", "SELL": "▼"}[verdict.stance]
        st.markdown(
            f'<div class="verdict {verdict.stance}"><div class="stance">{icon} {verdict.stance}</div>'
            f'<div><div class="who">{html.escape(company)} ({html.escape(ticker)})</div>'
            f'<div class="sub">{len(view.fundamentals)} verified ratio values · '
            f'{len(view.risks)} traced risks</div></div></div>', unsafe_allow_html=True)

        m = st.columns(2)
        m[0].metric("Profitability", view.profitability.rating.capitalize())
        m[1].metric("Valuation", view.valuation.rating.capitalize())

        left, right = st.columns(2, gap="medium")
        with left.container(border=True):
            st.markdown("**💡 Reasons**\n" + bullets(verdict.reasons))
        with right.container(border=True):
            st.markdown("**👀 Watchlist — what would change the call**\n" + bullets(verdict.watchlist))
        with st.container(border=True):
            st.markdown("**⚠️ Risks** — each traced to a ratio\n" + bullets(
                [f"{r.risk}  \n:gray[cites: {', '.join(r.cited_ratios)}]" for r in view.risks]))
        with st.expander("📑 Evidence"):
            st.markdown("**Profitability**\n" + bullets(view.profitability.evidence))
            st.markdown("**Valuation**\n" + bullets(view.valuation.evidence))
        with st.expander(f"🔢 Verified source numbers ({len(view.fundamentals)})"):
            df = pd.DataFrame([p.model_dump() for p in view.fundamentals])
            st.dataframe(df.pivot_table(index="ratio", columns="fiscal_year", values="value"),
                         width="stretch")
        show_log(log)

    if go_btn:
        friendly_errors(go)
    else:
        empty("🧭", "Run the pipeline to get a verdict", "Takes about a minute — three agents, every number checked.")
    footer()


PAGES = {
    "home": st.Page(page_home, title="Home", icon="🏠", default=True),
    "earnings": st.Page(page_earnings, title="Earnings card", icon="📄", url_path="earnings"),
    "price": st.Page(page_price, title="Live price", icon="💹", url_path="price"),
    "ratios": st.Page(page_ratios, title="Ratios", icon="📊", url_path="ratios"),
    "rebalance": st.Page(page_rebalance, title="Rebalancing chat", icon="⚖️", url_path="rebalance"),
    "news": st.Page(page_news, title="Verified news", icon="📰", url_path="news"),
    "bhs": st.Page(page_bhs, title="Buy / Hold / Sell", icon="🧭", url_path="bhs"),
}

if __name__ == "__main__":  # Streamlit runs the script as __main__
    from finance_agents.config import MODEL_AGENT
    with st.sidebar:
        st.markdown('<div class="brand"><div class="logo">🏦</div><div><div class="name">Future with '
                    'Shivank Bank</div><div class="sub">AI Agents in Finance</div></div></div>'
                    f'<div class="live"><span class="dot"></span>Live data · {MODEL_AGENT}</div>',
                    unsafe_allow_html=True)
    nav = st.navigation({"": [PAGES["home"]],
                         "Research": [PAGES["earnings"], PAGES["price"], PAGES["ratios"]],
                         "Advisory": [PAGES["rebalance"], PAGES["news"], PAGES["bhs"]]})
    nav.run()
