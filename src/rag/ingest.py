"""
Ingestion pipeline: load raw academic-policy documents from data/raw/,
split them into overlapping chunks, embed them, and persist them into a
local Chroma vector store.

Run standalone:
    python -m src.rag.ingest
"""

from __future__ import annotations

import os
import glob

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import pdfplumber

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma_db")

# Chunking strategy: 500-char chunks with 100-char overlap.
# Academic policy docs are dense with clause-level detail (deadlines,
# credit limits, percentages) so we keep chunks small enough that a
# single retrieved chunk is usually a self-contained rule, with enough
# overlap that a rule split across a paragraph boundary isn't cut in half.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_raw_documents() -> list[Document]:
    """
    Load every .txt and .pdf file in data/raw/ into LangChain Documents.

    - .txt files: read whole, with the sample-data disclaimer header
      stripped if present (see _strip_disclaimer_header).
    - .pdf files: text extracted per-page with pdfplumber; each page
      becomes its own Document so citations can point to a specific
      page (metadata["page"]), which matters for real institute PDFs
      where "which page" is genuinely useful to a student checking a
      source.
    """
    docs: list[Document] = []

    for path in sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        text = _strip_disclaimer_header(text)
        if text.strip():
            docs.append(
                Document(page_content=text, metadata={"source": os.path.basename(path)})
            )

    for path in sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.pdf"))):
        docs.extend(_load_pdf(path))

    if not docs:
        raise FileNotFoundError(
            f"No .txt or .pdf documents found in {RAW_DATA_DIR}. Add source documents first."
        )
    return docs


def _load_pdf(path: str) -> list[Document]:
    """Extract text from a PDF, one Document per page (skipping blank pages)."""
    filename = os.path.basename(path)
    page_docs: list[Document] = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                # Skip pages with no extractable text (e.g. a pure image/scan).
                # If your PDFs are scanned images rather than text-based, run
                # OCR first (e.g. ocrmypdf) before dropping them into data/raw/.
                continue
            page_docs.append(
                Document(
                    page_content=text,
                    metadata={"source": filename, "page": page_number},
                )
            )
    if not page_docs:
        print(
            f"WARNING: no extractable text found in {filename}. "
            "It may be a scanned/image-only PDF — OCR it first."
        )
    return page_docs


def _strip_disclaimer_header(text: str) -> str:
    """
    The sample data files in this repo start with a bracketed disclaimer
    paragraph (e.g. "[SAMPLE / ILLUSTRATIVE DOCUMENT ...]") explaining that
    they're placeholder content, not real IIT Bombay documents. That's
    important for a human reading the raw file, but it's noise for the
    retriever — we don't want a similarity search accidentally matching the
    disclaimer instead of actual policy text. Strip it before chunking.
    """
    if text.lstrip().startswith("[SAMPLE"):
        parts = text.split("\n\n", 1)
        if len(parts) == 2:
            return parts[1].lstrip()
    return text


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    for doc in docs:
        pieces = splitter.split_text(doc.page_content)
        source = doc.metadata["source"]
        page = doc.metadata.get("page")  # present for PDF-derived docs, absent for .txt
        for i, piece in enumerate(pieces):
            if page is not None:
                chunk_id = f"{source}#p{page}-chunk-{i}"
            else:
                chunk_id = f"{source}#chunk-{i}"
            metadata = {"source": source, "chunk_id": chunk_id}
            if page is not None:
                metadata["page"] = page
            chunks.append(Document(page_content=piece, metadata=metadata))
    return chunks


def build_vectorstore(persist_dir: str = PERSIST_DIR) -> Chroma:
    docs = load_raw_documents()
    chunks = chunk_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="iitb_academic_policy",
    )
    return vectorstore


def get_vectorstore(persist_dir: str = PERSIST_DIR) -> Chroma:
    """Load an existing vector store, building it first if it doesn't exist."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="iitb_academic_policy",
        )
    return build_vectorstore(persist_dir)


if __name__ == "__main__":
    vs = build_vectorstore()
    print(f"Vector store built and persisted at: {PERSIST_DIR}")
    print(f"Total chunks indexed: {vs._collection.count()}")
