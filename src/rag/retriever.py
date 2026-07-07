"""
Thin wrapper around the Chroma vector store that the retrieval agent
calls. Kept separate from ingest.py so the graph doesn't need to know
about text splitting / embedding model details.
"""

from __future__ import annotations

from src.rag.ingest import get_vectorstore

# Below this similarity score we treat a chunk as "not actually relevant".
# Chroma's default similarity_search_with_relevance_scores returns a
# score in [0, 1] where higher = more similar (it normalizes cosine
# distance internally). This threshold is intentionally conservative so
# the assistant prefers saying "I don't know" over guessing.
RELEVANCE_THRESHOLD = 0.35
TOP_K = 4


def retrieve(question: str, k: int = TOP_K, threshold: float = RELEVANCE_THRESHOLD):
    """
    Return a list of dicts: {text, source, chunk_id, score}
    for chunks relevant enough to the question. Empty list means
    "nothing grounded enough was found".
    """
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_relevance_scores(question, k=k)

    chunks = []
    for doc, score in results:
        if score >= threshold:
            chunk = {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                "score": round(float(score), 4),
            }
            if "page" in doc.metadata:
                chunk["page"] = doc.metadata["page"]
            chunks.append(chunk)
    return chunks
