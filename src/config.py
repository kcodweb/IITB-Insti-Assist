"""
Central settings. Every value can be overridden with an environment
variable of the same name (e.g. in .env), so experiments like
"what if chunks were bigger?" don't need code changes. Changing any of
the indexing settings automatically triggers a rebuild of the vector
store on next run (see src/rag/ingest.py: the index manifest).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", ROOT / "data" / "raw"))
SOURCES_FILE = Path(os.getenv("SOURCES_FILE", ROOT / "data" / "sources.json"))
PERSIST_DIR = Path(os.getenv("PERSIST_DIR", ROOT / "data" / "chroma_db"))
COLLECTION_NAME = "iitb_academic_policy"

# --- indexing -----------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- retrieval ----------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))
# Minimum cosine similarity of the best dense match for a question to count
# as "grounded" in the knowledge base. Tuned with `python -m eval.run_eval`.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.55"))
# Blend keyword (BM25) search with dense search. Helps a lot on acronym-heavy
# policy text ("DX", "FR", "CPI", "URA02") that small embedding models blur.
HYBRID_SEARCH = os.getenv("HYBRID_SEARCH", "true").lower() in {"1", "true", "yes"}

# --- agents -------------------------------------------------------------------
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))
# How many previous chat messages the agents get to see for follow-ups.
HISTORY_WINDOW = int(os.getenv("HISTORY_WINDOW", "6"))
