"""
Small factory so the rest of the codebase never has to care which LLM
provider is configured. Controlled entirely by environment variables
(see .env.example).

This project is configured for Gemini by default. Anthropic/OpenAI support
is still here (useful if you switch providers later) but only
langchain-google-genai is required by requirements.txt — install the
others yourself if you want them.
"""

from __future__ import annotations

import os


def get_chat_model(temperature: float = 0.0):
    """Return a LangChain chat model based on the LLM_PROVIDER env var."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        return ChatAnthropic(model=model, temperature=temperature)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=temperature)

    raise ValueError(
        f"Unknown LLM_PROVIDER='{provider}'. Use 'gemini', 'anthropic', or 'openai'."
    )


def extract_text(response) -> str:
    """
    Return the plain-text content of a chat model response, regardless of
    whether the provider returned `.content` as a plain string or as a list
    of content blocks.

    Some providers/versions (notably recent langchain-google-genai releases)
    return `.content` as a list like [{"type": "text", "text": "..."}] or
    even a list of plain strings, instead of a single string. Every call
    site in this project (answer_agent, critic_agent) needs a plain string,
    so this normalizes it once instead of duplicating the same
    isinstance-checking logic in every agent.
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
