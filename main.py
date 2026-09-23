"""
Command-line interface — handy for testing the graph without Streamlit.

Usage:
    python main.py "What is the minimum attendance required to avoid a DX grade?"
    python main.py                # interactive chat (follow-up questions work)
    python main.py --no-trace "..."
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from src.graph import stream  # noqa: E402
from src.utils import llm as llm_utils  # noqa: E402

STATUS_TEXT = {
    "verified": "verified by the critic agent",
    "unverified": "NOT fully verified — check the sources",
    "refused": "not covered by the documents",
}


def ask(question: str, history: list[dict], show_trace: bool) -> str:
    if show_trace:
        print("--- Agent trace ---")
    state: dict = {}
    for lines, state in stream(question, chat_history=history):
        if show_trace:
            for line in lines:
                print(line, flush=True)

    answer = state.get("final_answer") or "(no answer produced)"
    print(f"\n--- Answer ({STATUS_TEXT.get(state.get('status'), '?')}) ---")
    print(answer)

    chunks = state.get("retrieved_chunks") or []
    cited = [n for n in state.get("citations") or [] if 1 <= n <= len(chunks)]
    if cited:
        print("\nSources:")
        for n in cited:
            c = chunks[n - 1]
            page = f", page {c['page']}" if "page" in c else ""
            print(f"  [{n}] {c['title']}{page} — {c.get('section') or c['source']}")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask IITB Insti-Assist a question.")
    parser.add_argument("question", nargs="*", help="omit to start an interactive chat")
    parser.add_argument("--no-trace", action="store_true", help="hide the agent trace")
    args = parser.parse_args()

    if llm_utils.missing_api_key():
        sys.exit(f"{llm_utils.missing_api_key()} is not set. Copy .env.example to .env and add your key.")

    if args.question:
        question = " ".join(args.question)
        print(f"\nQ: {question}\n")
        ask(question, [], not args.no_trace)
        return

    print("IITB Insti-Assist — ask about UG academic rules and the 2026-27 calendar.")
    print("Follow-up questions remember the conversation. Ctrl+C or 'exit' to quit.\n")
    history: list[dict] = []
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        answer = ask(question, history, not args.no_trace)
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        print()


if __name__ == "__main__":
    main()
