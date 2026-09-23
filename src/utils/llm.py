"""
Small factory so the rest of the codebase never has to care which LLM
provider is configured. Controlled entirely by environment variables
(see .env.example).

This project is configured for Gemini by default. Anthropic/OpenAI support
is still here (useful if you switch providers later) but only
langchain-google-genai is required by requirements.txt — install
langchain-anthropic / langchain-openai yourself if you want them.
"""

from __future__ import annotations

import os
from functools import lru_cache

PROVIDER_KEYS = {
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "gemini").lower()


def model_name() -> str:
    return {
        "gemini": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
        "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    }.get(provider(), "?")


def missing_api_key() -> str | None:
    """Name of the env var that still needs to be set, or None if we're good."""
    key = PROVIDER_KEYS.get(provider())
    return key if key and not os.getenv(key) else None


@lru_cache(maxsize=None)
def get_chat_model(temperature: float = 0.0):
    """Return a LangChain chat model based on the LLM_PROVIDER env var (cached)."""
    name = provider()

    if name == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name(), temperature=temperature)

    if name == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name(), temperature=temperature)

    if name == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name(), temperature=temperature)

    raise ValueError(
        f"Unknown LLM_PROVIDER='{name}'. Use 'gemini', 'anthropic', or 'openai'."
    )


def extract_text(response) -> str:
    """
    Return the plain-text content of a chat model response, regardless of
    whether the provider returned `.content` as a plain string or as a list
    of content blocks.

    Some providers/versions (notably recent langchain-google-genai releases)
    return `.content` as a list like [{"type": "text", "text": "..."}] or
    even a list of plain strings, instead of a single string. Every call
    site in this project needs a plain string, so this normalizes it once
    instead of duplicating the same isinstance-checking logic in every agent.
    """
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)

    return str(content)
