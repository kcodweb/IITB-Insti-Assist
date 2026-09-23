"""
Ingestion pipeline: load the academic-policy documents in data/raw/,
clean them, split them into section-aware chunks, embed them, and persist
them into a local Chroma vector store.

Run standalone (always does a clean rebuild):
    python -m src.rag.ingest

You normally don't need to: get_vectorstore() keeps a small manifest next
to the index (data files' hash + chunking/embedding settings) and rebuilds
automatically whenever any of those change.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

# Bump whenever the cleaning/chunking logic changes, so existing indexes
# are rebuilt instead of silently mixing old and new chunks.
INDEX_VERSION = 2
MANIFEST_NAME = "manifest.json"

# BGE models retrieve noticeably better when short queries carry this prefix
# (documents are embedded without it).
_BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

@dataclass
class Segment:
    """A run of text on one page that belongs to one section."""

    source: str
    title: str
    page: int | None
    section: str
    text: str


def list_source_files() -> list[Path]:
    raw_dir = Path(config.RAW_DATA_DIR)
    return sorted(p for p in raw_dir.glob("*") if p.suffix.lower() in {".pdf", ".txt"})


def load_source_info() -> dict:
    """Optional human-friendly titles/descriptions, keyed by file name."""
    path = Path(config.SOURCES_FILE)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def document_title(filename: str, info: dict | None = None) -> str:
    info = info if info is not None else load_source_info()
    title = info.get(filename, {}).get("title")
    return title or Path(filename).stem.replace("_", " ")


def _read_pages(path: Path) -> list[tuple[int | None, str]]:
    if path.suffix.lower() == ".txt":
        return [(None, path.read_text(encoding="utf-8"))]

    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return [(i, page.extract_text() or "") for i, page in enumerate(pdf.pages, start=1)]


def load_raw_documents() -> list[Segment]:
    """
    Read every .pdf/.txt in data/raw/ and return cleaned, section-tagged
    text segments (one or more per page).
    """
    files = list_source_files()
    if not files:
        raise FileNotFoundError(
            f"No .txt or .pdf documents found in {config.RAW_DATA_DIR}. Add source documents first."
        )

    info = load_source_info()
    segments: list[Segment] = []
    for path in files:
        pages = _read_pages(path)
        texts = clean_pages([text for _, text in pages])
        title = document_title(path.name, info)

        doc_segments = _segment_document(path.name, title, [p for p, _ in pages], texts)
        if not doc_segments:
            print(
                f"WARNING: no extractable text found in {path.name}. "
                "It may be a scanned/image-only PDF — OCR it first (e.g. ocrmypdf)."
            )
        segments.extend(doc_segments)
    return segments


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #

_PAGE_NUMBER = re.compile(r"^\d{1,3}$")
# "understand-\ning" -> "understanding" (only when the next line continues a word)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(?=[a-z])")


def clean_pages(pages: list[str]) -> list[str]:
    """
    Remove running headers/footers (any line repeated on at least half of
    the pages, e.g. the rulebook's "Move to Index" footer), bare page
    numbers, and words hyphenated across line breaks.
    """
    boilerplate: set[str] = set()
    if len(pages) >= 4:
        counts = Counter(
            line for page in pages for line in {l.strip() for l in page.splitlines() if l.strip()}
        )
        boilerplate = {line for line, n in counts.items() if n >= len(pages) / 2}

    cleaned = []
    for page in pages:
        lines = [
            line.rstrip()
            for line in page.splitlines()
            if line.strip()
            and line.strip() not in boilerplate
            and not _PAGE_NUMBER.match(line.strip())
        ]
        cleaned.append(_HYPHEN_BREAK.sub(r"\1", "\n".join(lines)))
    return cleaned


_TOC_LINE = re.compile(r"^\S.*\s\d{1,3}$")


def is_toc_page(text: str) -> bool:
    """
    Table-of-contents pages ("4.6 Course Adjustment/ Dropping of courses 17")
    mention every topic but answer nothing, so they crowd real answers out of
    the top-k. Detected as pages where most lines end in a page number.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    if lines[0].upper() in {"INDEX", "CONTENTS", "TABLE OF CONTENTS"}:
        return True
    if len(lines) < 10:
        return False
    return sum(bool(_TOC_LINE.match(l)) for l in lines) / len(lines) >= 0.6


# --------------------------------------------------------------------------- #
# Section tracking
# --------------------------------------------------------------------------- #

# "6.5 Grading", "4. 4 Registration for ...", "11. ACADEMIC REHABILITATION ..."
_NUMBERED_HEADING = re.compile(r"^(\d{1,2}(?:\.\s?\d{1,2}){0,3})\.?\s+(\S.*)$")
# "(Ref: 239th Senate Meeting ...)" / "(249th Senate Meeting)" suffixes on headings
_HEADING_REF = re.compile(r"\s*\((?:Ref\b|\d+(?:st|nd|rd|th)\s+Senate).*$")
# Calendar rows like "1 July 2026 (Wednesday)" look like "1 <Title>" headings
_STARTS_WITH_MONTH = re.compile(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I)
# "(A) SCHEDULE FOR AUTUMN SEMESTER"
_LETTERED_HEADING = re.compile(r"^\(([A-Z])\)\s+([A-Z][A-Z0-9 /&.,'()-]{3,})$")


def _looks_like_title(title: str) -> bool:
    letters = [c for c in title if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) >= 0.7:
        return True  # ALL CAPS
    words = [w for w in re.findall(r"[A-Za-z][\w/'-]*", title) if len(w) > 3]
    return bool(words) and len(title) <= 70 and sum(w[0].isupper() for w in words) / len(words) >= 0.6


@dataclass
class SectionTracker:
    """
    Follows the document's heading structure line by line, so every chunk
    can be labelled with where it came from (e.g. "6 EXAMINATION /
    ASSESSMENT > 6.5 Grading", or "(B) SCHEDULE FOR SPRING SEMESTER").

    This matters most for the calendar: the Spring and Autumn tables use
    identical wording ("End-semester examination ... 19 Apr 2027"), and
    without the section label a chunk from page 7 gives no hint which
    semester it belongs to.
    """

    path: list[tuple[int, str]] = field(default_factory=list)  # (level, label)
    major: int = 0
    lettered: bool = False  # document uses "(A) ..." headings, not "1. ..."

    def feed(self, line: str) -> bool:
        """Update the current section if `line` is a heading. Returns True if it was."""
        line = line.strip()

        m = _LETTERED_HEADING.match(line)
        if m:
            self.path = [(1, line)]
            self.lettered = True
            return True

        m = _NUMBERED_HEADING.match(line)
        if not m or self.lettered:  # in lettered docs, "1." lines are list items
            return False
        nums = [int(n) for n in re.split(r"\.\s?", m.group(1))]
        title = _HEADING_REF.sub("", m.group(2)).strip().rstrip(":").strip()
        if (
            not title
            or len(title) > 110
            or not title[0].isupper()
            or _STARTS_WITH_MONTH.match(title)
            or (title.endswith(".") and not _looks_like_title(title))  # a sentence
        ):
            return False

        if len(nums) == 1:
            # A new top-level section must follow the previous one and look
            # like a title — this rejects numbered list items in body text.
            # The very first one can have any number but must be ALL CAPS.
            if self.major == 0:
                letters = [c for c in title if c.isalpha()]
                if not letters or sum(c.isupper() for c in letters) / len(letters) < 0.7:
                    return False
            elif not (self.major < nums[0] <= self.major + 2 and _looks_like_title(title)):
                return False
            self.major = nums[0]
        elif nums[0] not in (self.major, self.major + 1):
            return False
        else:
            self.major = nums[0]

        level = len(nums)
        label = f"{'.'.join(map(str, nums))} {title}"
        self.path = [p for p in self.path if p[0] < level] + [(level, label)]
        return True

    @property
    def label(self) -> str:
        return " > ".join(label for _, label in self.path)


def _segment_document(
    source: str, title: str, page_numbers: list[int | None], texts: list[str]
) -> list[Segment]:
    tracker = SectionTracker()
    segments: list[Segment] = []

    for page, text in zip(page_numbers, texts):
        if not text.strip() or is_toc_page(text):
            continue

        page_segments: list[tuple[Segment, bool]] = []  # (segment, is only a heading)
        current: list[str] = []
        section = tracker.label
        starts_with_heading = False

        def flush():
            body = "\n".join(current).strip()
            if body:
                heading_only = starts_with_heading and len(current) == 1
                page_segments.append((Segment(source, title, page, section, body), heading_only))

        for line in text.splitlines():
            if tracker.feed(line):
                flush()
                current, section, starts_with_heading = [], tracker.label, True
            current.append(line)
        flush()

        # A heading immediately followed by a sub-heading ("2.3.4 Projects")
        # would otherwise become a heading-only chunk: fold it into the next one.
        carry = ""
        for i, (seg, heading_only) in enumerate(page_segments):
            if heading_only and i < len(page_segments) - 1:
                carry += seg.text + "\n"
                continue
            seg.text = carry + seg.text
            carry = ""
            segments.append(seg)

    return segments


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

def chunk_header(seg: Segment) -> str:
    return f"[{seg.title} | {seg.section}]" if seg.section else f"[{seg.title}]"


def chunk_documents(
    segments: list[Segment],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split segments into overlapping chunks. Each chunk is prefixed with a
    short "[document | section]" header, so both the embedding and the LLM
    see the context the raw text lost when it was cut out of the page.
    Chunks never cross page boundaries, which keeps page citations exact.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    per_page_counter: Counter = Counter()

    for seg in segments:
        header = chunk_header(seg)
        for piece in splitter.split_text(seg.text):
            i = per_page_counter[(seg.source, seg.page)]
            per_page_counter[(seg.source, seg.page)] += 1
            if seg.page is not None:
                chunk_id = f"{seg.source}#p{seg.page}-chunk-{i}"
            else:
                chunk_id = f"{seg.source}#chunk-{i}"

            metadata = {
                "source": seg.source,
                "title": seg.title,
                "section": seg.section,
                "chunk_id": chunk_id,
            }
            if seg.page is not None:
                metadata["page"] = seg.page
            chunks.append(Document(page_content=f"{header}\n{piece}", metadata=metadata))
    return chunks


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #

def get_embeddings():
    return _load_embeddings(config.EMBEDDING_MODEL)


@lru_cache(maxsize=None)
def _load_embeddings(model_name: str):
    from langchain_huggingface import HuggingFaceEmbeddings

    query_kwargs = {"normalize_embeddings": True}
    if "bge-" in model_name.lower() and "-en" in model_name.lower():
        query_kwargs["prompt"] = _BGE_QUERY_PROMPT
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs=query_kwargs,
    )


def index_fingerprint() -> dict:
    """Everything that, if changed, means the index must be rebuilt."""
    digest = hashlib.sha256()
    for path in list_source_files() + [config.SOURCES_FILE]:
        if path.exists():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return {
        "index_version": INDEX_VERSION,
        "embedding_model": config.EMBEDDING_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "data_sha256": digest.hexdigest(),
    }


def read_manifest(persist_dir: Path | None = None) -> dict | None:
    path = Path(persist_dir or config.PERSIST_DIR) / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _open_store(persist_dir: Path):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
        # Cosine distance, so relevance scores are plain cosine similarity
        # (Chroma's default is squared L2, which made the old 0.35
        # threshold mean something quite different from what it said).
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_vectorstore(persist_dir: Path | None = None):
    """(Re)build the index from scratch. Safe to run repeatedly."""
    persist_dir = Path(persist_dir or config.PERSIST_DIR)
    segments = load_raw_documents()
    chunks = chunk_documents(segments)

    persist_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(persist_dir) / MANIFEST_NAME
    manifest_path.unlink(missing_ok=True)  # an interrupted build must not look valid

    store = _open_store(persist_dir)
    # Start from an empty collection: re-running ingest used to append a
    # second copy of every chunk, and chunks of deleted files never left.
    store.reset_collection()
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])

    info = load_source_info()
    documents: dict[str, dict] = {}
    for c in chunks:
        d = documents.setdefault(
            c.metadata["source"],
            {"title": c.metadata["title"], "description": "", "pages": set(), "chunks": 0},
        )
        d["description"] = info.get(c.metadata["source"], {}).get("description", "")
        d["chunks"] += 1
        if "page" in c.metadata:
            d["pages"].add(c.metadata["page"])
    for d in documents.values():
        d["pages"] = len(d["pages"])

    manifest = {
        "fingerprint": index_fingerprint(),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_chunks": len(chunks),
        "documents": documents,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return store


def get_vectorstore():
    """Open the index, (re)building it first if it's missing or out of date."""
    return _get_vectorstore(Path(config.PERSIST_DIR))


@lru_cache(maxsize=None)
def _get_vectorstore(persist_dir: Path):
    manifest = read_manifest(persist_dir)
    if manifest and manifest.get("fingerprint") == index_fingerprint():
        return _open_store(persist_dir)
    print("Vector index is missing or out of date — building it now...")
    return build_vectorstore(persist_dir)


if __name__ == "__main__":
    store = build_vectorstore()
    manifest = read_manifest()
    print(f"Vector store built at: {config.PERSIST_DIR}")
    for source, d in manifest["documents"].items():
        print(f"  {source}: {d['pages']} pages -> {d['chunks']} chunks")
    print(f"Total chunks indexed: {manifest['total_chunks']}")
