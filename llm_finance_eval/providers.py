"""Model providers. Each one is a callable: prompt -> response text.

- "mock":      offline, answers from the key (sanity check of the pipeline)
- "anthropic": needs ANTHROPIC_API_KEY and `pip install anthropic`
- "openai":    needs OPENAI_API_KEY and `pip install openai`
"""
from __future__ import annotations

import os
from typing import Callable

SYSTEM = "You are a senior corporate finance and M&A professional. Be precise."


def mock_provider(answer_key: dict[str, float]) -> Callable[[dict], str]:
    def call(task: dict) -> str:
        return f"(mock reasoning)\nFINAL ANSWER: {answer_key[task['id']]}"
    return call


def anthropic_provider(model: str) -> Callable[[dict], str]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def call(task: dict) -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=2000,
            system=SYSTEM,
            messages=[{"role": "user", "content": task["prompt"]}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return call


def openai_provider(model: str) -> Callable[[dict], str]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def call(task: dict) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": task["prompt"]}],
        )
        return resp.choices[0].message.content or ""
    return call


def get_provider(name: str, model: str | None, tasks: list[dict]) -> Callable[[dict], str]:
    if name == "mock":
        return mock_provider({t["id"]: t["answer"] for t in tasks})
    if name == "anthropic":
        return anthropic_provider(model or "claude-sonnet-5-5")
    if name == "openai":
        return openai_provider(model or "gpt-5")
    raise ValueError(f"unknown provider: {name}")
