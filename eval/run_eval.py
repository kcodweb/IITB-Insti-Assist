"""
Evaluate the assistant against eval/questions.jsonl.

    python -m eval.run_eval          # retrieval only: no API key needed
    python -m eval.run_eval --e2e    # also run the full agent graph (needs an LLM key)

The question set has three kinds of questions:
  * answerable   (rulebook / calendar) — labelled with the page(s) that answer them
  * out_of_scope — nothing to do with IITB academics; must be refused
  * unanswerable — sounds in-scope, but the documents don't say

Retrieval metrics: Hit@1 / Hit@k (a chunk from a correct page is in the top
1 / top k) and MRR. The threshold sweep shows, for each candidate value of
RELEVANCE_THRESHOLD, how many answerable questions would still be answered
and how many out-of-scope ones would be refused — which is how the default
threshold in src/config.py was picked.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from src import config
from src.rag.retriever import search

QUESTIONS_FILE = Path(__file__).with_name("questions.jsonl")
REFUSAL_MARKERS = ("i don't know", "i do not know", "not specified", "does not specify",
                   "doesn't specify", "do not specify", "not mentioned", "does not mention",
                   "doesn't mention", "no information", "not covered", "does not contain",
                   "doesn't contain", "don't contain", "do not contain", "not provide")


def load_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_answerable(q: dict) -> bool:
    return q["kind"] not in {"out_of_scope", "unanswerable"}


def _first_relevant_rank(q: dict, chunks: list[dict]) -> int | None:
    for rank, c in enumerate(chunks, start=1):
        if c["source"] == q["source"] and c.get("page") in q["pages"]:
            return rank
    return None


def evaluate_retrieval(questions: list[dict], k: int) -> dict:
    rows = []
    for q in questions:
        result = search(q["question"], k=k)
        rank = _first_relevant_rank(q, result.chunks) if is_answerable(q) else None
        rows.append({**q, "rank": rank, "top_score": result.top_score,
                     "retrieved": [c["chunk_id"] for c in result.chunks]})

    answerable = [r for r in rows if is_answerable(r)]
    ranks = [r["rank"] for r in answerable]
    return {
        "rows": rows,
        "hit@1": sum(r == 1 for r in ranks) / len(ranks),
        f"hit@{k}": sum(r is not None for r in ranks) / len(ranks),
        "mrr": sum(1 / r for r in ranks if r) / len(ranks),
    }


def threshold_sweep(rows: list[dict], thresholds: list[float]) -> list[dict]:
    by_kind = {
        "answerable": [r["top_score"] for r in rows if is_answerable(r)],
        "out_of_scope": [r["top_score"] for r in rows if r["kind"] == "out_of_scope"],
        "unanswerable": [r["top_score"] for r in rows if r["kind"] == "unanswerable"],
    }
    table = []
    for t in thresholds:
        table.append({
            "threshold": t,
            "answerable_kept": sum(s >= t for s in by_kind["answerable"]) / len(by_kind["answerable"]),
            "out_of_scope_refused": sum(s < t for s in by_kind["out_of_scope"]) / len(by_kind["out_of_scope"]),
            "unanswerable_refused": sum(s < t for s in by_kind["unanswerable"]) / len(by_kind["unanswerable"]),
        })
    return table


def evaluate_end_to_end(questions: list[dict]) -> None:
    from src.graph import run

    results = {"answered_correctly": 0, "answerable": 0, "refused_correctly": 0, "refusable": 0}
    latencies = []
    for q in questions:
        start = time.perf_counter()
        state = run(q["question"])
        latencies.append(time.perf_counter() - start)
        answer = (state.get("final_answer") or "")
        refused = state.get("status") == "refused" or any(m in answer.lower() for m in REFUSAL_MARKERS)

        if is_answerable(q):
            results["answerable"] += 1
            ok = not refused and any(e in answer for e in q["expect_any"])
            results["answered_correctly"] += ok
        else:
            results["refusable"] += 1
            ok = refused
            results["refused_correctly"] += ok
        mark = "OK  " if ok else "MISS"
        print(f"  {mark} [{q['id']}] {state.get('status', '?'):<10} {q['question']}")

    print("\n## End-to-end\n")
    print(f"- Answerable questions answered with the expected fact: "
          f"{results['answered_correctly']}/{results['answerable']}")
    print(f"- Out-of-scope / unanswerable questions declined: "
          f"{results['refused_correctly']}/{results['refusable']}")
    print(f"- Median latency: {statistics.median(latencies):.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=config.TOP_K, help="chunks retrieved per question")
    parser.add_argument("--e2e", action="store_true", help="also run the full agent graph (needs an LLM API key)")
    parser.add_argument("--quiet", action="store_true", help="don't list individual retrieval misses")
    args = parser.parse_args()

    questions = load_questions()
    n_ans = sum(is_answerable(q) for q in questions)
    print(f"# Retrieval eval - {len(questions)} questions ({n_ans} answerable)\n")
    print(f"embedding={config.EMBEDDING_MODEL}  chunk={config.CHUNK_SIZE}/{config.CHUNK_OVERLAP}  "
          f"hybrid={config.HYBRID_SEARCH}  k={args.k}\n")

    report = evaluate_retrieval(questions, args.k)
    print("| Hit@1 | Hit@{k} | MRR |".format(k=args.k))
    print("|---|---|---|")
    print(f"| {report['hit@1']:.0%} | {report[f'hit@{args.k}']:.0%} | {report['mrr']:.3f} |\n")

    if not args.quiet:
        misses = [r for r in report["rows"] if is_answerable(r) and r["rank"] is None]
        for r in misses:
            print(f"  miss [{r['id']}] {r['question']}\n        got: {', '.join(r['retrieved'])}")
        if misses:
            print()

    rows = report["rows"]
    for kind in ("answerable", "out_of_scope", "unanswerable"):
        scores = [r["top_score"] for r in rows if (is_answerable(r) if kind == "answerable" else r["kind"] == kind)]
        print(f"top dense score, {kind:<13} min {min(scores):.3f}  median {statistics.median(scores):.3f}  max {max(scores):.3f}")

    print("\n| threshold | answerable kept | out-of-scope refused | unanswerable refused |")
    print("|---|---|---|---|")
    for row in threshold_sweep(rows, [round(0.40 + 0.025 * i, 3) for i in range(15)]):
        marker = " <-" if abs(row["threshold"] - config.RELEVANCE_THRESHOLD) < 1e-9 else ""
        print(f"| {row['threshold']:.3f}{marker} | {row['answerable_kept']:.0%} | "
              f"{row['out_of_scope_refused']:.0%} | {row['unanswerable_refused']:.0%} |")

    if args.e2e:
        print()
        evaluate_end_to_end(questions)


if __name__ == "__main__":
    main()
