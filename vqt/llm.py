"""Text-API LLM hooks. The VQT receiver only ever sends plain text (the
dictionary label), so any text-API model works — this is the compatibility
claim the continuous-latent methods cannot make.

Plug your backend into `call_llm`. Two reference adapters are provided
(OpenAI-compatible and a local echo stub). Keep the signature:
    call_llm(prompt: str, tools: list[dict] | None) -> str
returning the model's raw tool-call output for the scorer.
"""
from __future__ import annotations
from typing import Callable, Optional

LLMFn = Callable[[str, Optional[list]], str]


def stub_llm(prompt: str, tools: Optional[list] = None) -> str:
    """Offline placeholder. Returns an empty call; only for plumbing tests."""
    return ""


def openai_chat_llm(model: str = "gpt-4o-mini",
                    system: str = "You are a function-calling router. "
                    "Return exactly one function call as name(arg=value, ...).") -> LLMFn:
    """Returns a call_llm bound to an OpenAI-compatible endpoint.
    Requires `openai` and OPENAI_API_KEY. Adapt base_url for local servers
    (vLLM/Ollama/TGI expose OpenAI-compatible APIs)."""
    from openai import OpenAI          # lazy import
    client = OpenAI()

    def _call(prompt: str, tools: Optional[list] = None) -> str:
        msg = prompt
        if tools:
            names = ", ".join(t.get("name", "") for t in tools)
            msg = f"Available tools: {names}\n\nRequest: {prompt}"
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": msg}],
            temperature=0,
        )
        return r.choices[0].message.content or ""

    return _call
