"""
Search over the indexed chunks. The retrieval agent is the only caller.

Two signals are combined:
  * dense  — cosine similarity of bge embeddings (good at paraphrases:
             "can I skip classes" ~ "attendance requirement")
  * BM25   — keyword overlap (good at exact tokens small embedding models
             blur: "DX", "FF", "URA02", "CPI 7.5")
and fused with Reciprocal Rank Fusion. Whether the question is answerable
from the documents at all ("grounded") is decided on the best *dense*
score, since BM25 scores aren't comparable across questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from src import config
from src.rag.ingest import get_vectorstore

RRF_K = 60  # standard Reciprocal Rank Fusion constant
CANDIDATES = 30  # how deep each ranker looks before fusion

_STOPWORDS = set(
    "a an and are as at be by can do does for from how i if in is it its me my of on or "
    "the to what when where which who will with would should there this that".split()
)


@dataclass
class SearchResult:
    chunks: list[dict]  # best-first; each {text, source, title, section, chunk_id, page?, score}
    top_score: float  # best dense cosine similarity; the groundedness signal


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if t not in _STOPWORDS]


@lru_cache(maxsize=1)
def _keyword_index():
    from rank_bm25 import BM25Okapi

    data = get_vectorstore().get(include=["documents"])
    return BM25Okapi([_tokenize(d) for d in data["documents"]]), data["ids"]


def _to_chunk(doc, score: float | None) -> dict:
    chunk = {
        "text": doc.page_content,
        "source": doc.metadata.get("source", "unknown"),
        "title": doc.metadata.get("title", doc.metadata.get("source", "unknown")),
        "section": doc.metadata.get("section", ""),
        "chunk_id": doc.metadata.get("chunk_id", "unknown"),
        "score": None if score is None else round(float(score), 4),
    }
    if "page" in doc.metadata:
        chunk["page"] = doc.metadata["page"]
    return chunk


def search(query: str, k: int = config.TOP_K, hybrid: bool = config.HYBRID_SEARCH) -> SearchResult:
    """Rank chunks for `query`, regardless of how relevant the best one is."""
    store = get_vectorstore()
    dense = store.similarity_search_with_relevance_scores(query, k=CANDIDATES)
    if not dense:
        return SearchResult(chunks=[], top_score=0.0)
    top_score = max(score for _, score in dense)

    if not hybrid:
        return SearchResult([_to_chunk(doc, s) for doc, s in dense[:k]], top_score)

    fused: dict[str, float] = {}
    docs: dict[str, tuple] = {}
    for rank, (doc, score) in enumerate(dense):
        cid = doc.metadata["chunk_id"]
        fused[cid] = fused.get(cid, 0.0) + 1 / (RRF_K + rank + 1)
        docs[cid] = (doc, score)

    bm25, ids = _keyword_index()
    scores = bm25.get_scores(_tokenize(query))
    keyword_ranked = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:CANDIDATES]
    for rank, i in enumerate(keyword_ranked):
        if scores[i] <= 0:
            break
        fused[ids[i]] = fused.get(ids[i], 0.0) + 1 / (RRF_K + rank + 1)

    best = sorted(fused, key=fused.get, reverse=True)[:k]
    missing = [cid for cid in best if cid not in docs]  # keyword-only hits
    if missing:
        for doc in store.get_by_ids(missing):
            docs[doc.metadata["chunk_id"]] = (doc, None)

    return SearchResult([_to_chunk(*docs[cid]) for cid in best if cid in docs], top_score)


def retrieve(
    query: str, k: int = config.TOP_K, threshold: float = config.RELEVANCE_THRESHOLD
) -> list[dict]:
    """
    Top-k chunks for `query`, or [] if even the best match is below the
    relevance threshold — i.e. the documents don't cover this question and
    the assistant should say "I don't know" rather than guess.
    """
    result = search(query, k)
    return result.chunks if result.top_score >= threshold else []


def warm_up() -> None:
    """Load the embedding model and keyword index up front (so a UI's first question is fast)."""
    get_vectorstore()
    if config.HYBRID_SEARCH:
        _keyword_index()
