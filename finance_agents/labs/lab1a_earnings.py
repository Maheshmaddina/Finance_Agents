"""Lab 1A · Earnings-call summariser -> a fixed JSON card for the PM dashboard.

The course's lesson: valid JSON is not usable JSON. Different models returned revenue as a
number, as "$98.7B (+6% YoY)", or as "$98.7 billion". So this version:
  1. uses the strict typed contract (every key, type and unit named), and
  2. enforces it with Structured Outputs (a JSON Schema the model cannot break), and
  3. validates any other model's output (Gemini, Llama) against the same contract.
"""
from __future__ import annotations

import json
import os
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ..config import MODEL_SUMMARY, PROVIDER, get_client, require_key
from ..data.transcripts import transcript_apple_earnings_call


class EarningsCard(BaseModel):
    """The dashboard card. Numbers arrive as numbers, in a known unit."""
    company: str
    quarter: str = Field(description='Fiscal quarter, like "FY2025 Q3"')
    revenue_usd: float = Field(description="Revenue in US dollars, e.g. 98700000000")
    revenue_growth_yoy_pct: float = Field(description="6 means +6% year over year")
    eps_usd: float = Field(description="Diluted EPS in US dollars")
    guidance: str
    highlights: list[str]
    risks: list[str]
    outlook: Literal["positive", "neutral", "negative"]


SYSTEM_PROMPT = """You are a finance analyst writing for portfolio managers.
Produce a concise, decision-useful summary of an earnings call.
Return strict JSON with exactly these keys and types:
  company: string
  quarter: string, like "FY2025 Q3"
  revenue_usd: number in US dollars, no symbols or words (98700000000, not "$98.7B")
  revenue_growth_yoy_pct: number (6 means +6%)
  eps_usd: number
  guidance: string
  highlights: list of strings
  risks: list of strings
  outlook: one of "positive", "neutral", "negative"
Use only facts stated in the transcript."""


def user_prompt(transcript: str) -> str:
    return f"Summarize the following earnings call transcript:\n{transcript}"


def summarise(transcript: str = transcript_apple_earnings_call) -> EarningsCard:
    """Main model with Structured Outputs: the reply must match EarningsCard."""
    response = get_client().chat.completions.parse(
        model=MODEL_SUMMARY,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_prompt(transcript)}],
        response_format=EarningsCard,
    )
    card = response.choices[0].message.parsed
    if card is None:
        raise RuntimeError("Model did not return a card: " + str(response.choices[0].message.refusal))
    return card


def summarise_openai_json_mode(transcript: str = transcript_apple_earnings_call) -> str:
    """Optional comparison: GPT-4.1-mini in plain JSON mode, as in the course (needs OPENAI_API_KEY)."""
    from openai import OpenAI

    resp = OpenAI(api_key=require_key("OPENAI_API_KEY")).chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_prompt(transcript)}],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def summarise_gemini(transcript: str = transcript_apple_earnings_call) -> str:
    """Optional comparison: Gemini 2.5 Flash in JSON mode (needs GOOGLE_API_KEY)."""
    from google import genai

    gclient = genai.Client(api_key=require_key("GOOGLE_API_KEY"))
    resp = gclient.models.generate_content(
        model="gemini-2.5-flash",
        contents=SYSTEM_PROMPT + "\n\n" + user_prompt(transcript),
        config={"response_mime_type": "application/json"},
    )
    return resp.text


def summarise_llama(transcript: str = transcript_apple_earnings_call) -> str:
    """Optional comparison: Llama 3.1 8B on your own GPU (needs HF_TOKEN + model access)."""
    import torch
    import transformers

    pipe = transformers.pipeline(
        "text-generation", model="meta-llama/Llama-3.1-8B-Instruct",
        model_kwargs={"torch_dtype": torch.float16}, device_map="auto",
        token=os.getenv("HF_TOKEN"),
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(transcript)}]
    return pipe(messages, max_new_tokens=1024)[0]["generated_text"][-1]["content"]


def validate_card(raw_json: str) -> tuple[EarningsCard | None, list[str]]:
    """Check another model's JSON against the contract. Returns (card, problems)."""
    try:
        data = json.loads(raw_json.strip().removeprefix("```json").removesuffix("```"))
    except json.JSONDecodeError as exc:
        return None, [f"not valid JSON: {exc}"]
    try:
        # strict=True: "$98.7B" is NOT silently accepted as a number
        return EarningsCard.model_validate(data, strict=True), []
    except ValidationError as exc:
        return None, [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]


def run(compare: bool = False) -> None:
    card = summarise()
    print(f"=== {MODEL_SUMMARY} via {PROVIDER} (structured outputs) ===")
    print(card.model_dump_json(indent=2))
    # The dashboard check from the course now always works, because revenue is a number:
    print("Dashboard check revenue_usd > 90e9:", card.revenue_usd > 90e9)

    if compare:
        # Plain JSON mode (no schema) — shows the course's "valid but not usable JSON" problem
        for name, fn, key in [("gpt-4.1-mini JSON mode", summarise_openai_json_mode, "OPENAI_API_KEY"),
                              ("gemini-2.5-flash JSON mode", summarise_gemini, "GOOGLE_API_KEY"),
                              ("Llama-3.1-8B", summarise_llama, "HF_TOKEN")]:
            if not os.getenv(key):
                print(f"\n(skipping {name}: {key} not set)")
                continue
            raw = fn()
            _, problems = validate_card(raw)
            print(f"\n=== {name} ===\n{raw}")
            print("Contract:", "OK" if not problems else "BROKEN -> " + "; ".join(problems))
