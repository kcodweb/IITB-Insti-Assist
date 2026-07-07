"""
Quick CLI for testing the graph without spinning up Streamlit.

Usage:
    python main.py "What is the minimum attendance required to sit an exam?"
"""

import sys

from dotenv import load_dotenv

from src.graph import run

load_dotenv()


def main():
    if len(sys.argv) < 2:
        print('Usage: python main.py "<your question>"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"\nQ: {question}\n")

    result = run(question)

    print("--- Agent trace ---")
    for line in result.get("history", []):
        print(line)

    print("\n--- Final answer ---")
    print(result.get("final_answer"))


if __name__ == "__main__":
    main()
