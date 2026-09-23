from __future__ import annotations

import json

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from src import config
from src.rag import ingest, retriever
from src.rag.ingest import SectionTracker, Segment, chunk_documents, clean_pages, is_toc_page


def test_clean_pages_drops_running_footers_page_numbers_and_rejoins_hyphenation():
    words = ["alpha", "beta", "gamma", "delta", "epsilon"]
    pages = [f"Rule {w} applies to every regis-\ntered {w} student\n{i}\nMove to Index" for i, w in enumerate(words, 1)]
    cleaned = clean_pages(pages)
    assert cleaned[0] == "Rule alpha applies to every registered alpha student"
    assert all("Move to Index" not in p for p in cleaned)


def test_toc_pages_are_detected_but_tables_are_not():
    toc = "\n".join(f"{i}.{j} Some Section Title {10 + i}" for i in range(1, 4) for j in range(1, 5))
    grade_table = "Letter Grade Point\nAA 10\nAB 9\nBB 8\n" + "\n".join(
        f"This is a normal sentence of rule text number {i}." for i in range(8)
    )
    assert is_toc_page("INDEX\nSection Particulars Page")
    assert is_toc_page(toc)
    assert not is_toc_page(grade_table)


def test_section_tracker_follows_numbered_headings_and_ignores_list_items():
    t = SectionTracker()
    labels = []
    for line in [
        "5. SPECIAL FEATURES IN REGISTRATION",
        "5.1 Academic Standing (Ref: 236th Senate Meeting)",
        "1. The student has to apply to the faculty adviser with a plan.",  # list item
        "2. Faculty Advisor may recommend the application to the DUGC.",  # list item
        "5.2 Permissible Registration Load",
        "6 EXAMINATION / ASSESSMENT",
        "6.5 Grading",
    ]:
        t.feed(line)
        labels.append(t.label)
    assert labels[1] == "5 SPECIAL FEATURES IN REGISTRATION > 5.1 Academic Standing"
    assert labels[3] == labels[1]
    assert labels[-1] == "6 EXAMINATION / ASSESSMENT > 6.5 Grading"


def test_section_tracker_handles_lettered_calendar_headings_and_date_rows():
    t = SectionTracker()
    assert t.feed("(B) SCHEDULE FOR SPRING SEMESTER")
    assert not t.feed("1 July 2026 (Wednesday) -")
    assert not t.feed("1. Academic Support Programme (ASP) Committee meetings")
    assert t.label == "(B) SCHEDULE FOR SPRING SEMESTER"


def _segments():
    return [
        Segment("cal.pdf", "Calendar", 6, "(B) SCHEDULE FOR SPRING SEMESTER", "Spring lectures begin 4 Jan 2027. " * 40),
        Segment("cal.pdf", "Calendar", 7, "(B) SCHEDULE FOR SPRING SEMESTER", "End-semester examination 19 Apr 2027."),
    ]


def test_chunks_carry_section_headers_page_metadata_and_stable_ids():
    chunks = chunk_documents(_segments(), chunk_size=300, chunk_overlap=50)
    ids = [c.metadata["chunk_id"] for c in chunks]

    assert len(ids) == len(set(ids))
    assert ids == [c.metadata["chunk_id"] for c in chunk_documents(_segments(), 300, 50)]
    assert ids[-1] == "cal.pdf#p7-chunk-0"
    assert all(c.page_content.startswith("[Calendar | (B) SCHEDULE FOR SPRING SEMESTER]") for c in chunks)
    assert {c.metadata["page"] for c in chunks} == {6, 7}


@pytest.fixture
def tmp_corpus(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "rules.txt").write_text(
        "1. REGISTRATION\n1.1 Late Registration\nLate registration is allowed on payment of a fee.\n"
        "1.2 Course Adjustment\nCourses may be added or deleted in the first week of the semester.\n"
        "2. GRADING\n2.1 Grade Points\nAn AA grade carries 10 grade points and BB carries 8.\n"
        "2.2 Attendance\nStudents below eighty percent attendance may be deregistered.\n"
        "3. LEAVE\n3.1 Special Leave\nThe head of the unit may sanction up to five working days.\n"
        "3.2 Planned Break\nA planned break of up to two semesters needs faculty approval.\n",
        encoding="utf-8",
    )
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"rules.txt": {"title": "Test Rules"}}), encoding="utf-8")

    monkeypatch.setattr(config, "RAW_DATA_DIR", raw)
    monkeypatch.setattr(config, "SOURCES_FILE", sources)
    monkeypatch.setattr(config, "PERSIST_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "CHUNK_SIZE", 120)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 20)
    monkeypatch.setattr(ingest, "get_embeddings", lambda: DeterministicFakeEmbedding(size=32))
    ingest._get_vectorstore.cache_clear()
    retriever._keyword_index.cache_clear()
    yield raw
    ingest._get_vectorstore.cache_clear()
    retriever._keyword_index.cache_clear()


def test_rebuilding_the_index_never_duplicates_chunks(tmp_corpus):
    first = ingest.build_vectorstore()._collection.count()
    second = ingest.build_vectorstore()._collection.count()
    assert first == second == ingest.read_manifest()["total_chunks"]


def test_index_rebuilds_automatically_when_documents_or_settings_change(tmp_corpus, monkeypatch):
    ingest.build_vectorstore()
    fingerprint = ingest.read_manifest()["fingerprint"]
    assert fingerprint == ingest.index_fingerprint()

    (tmp_corpus / "extra.txt").write_text("A brand new circular.", encoding="utf-8")
    assert ingest.index_fingerprint() != fingerprint

    monkeypatch.setattr(config, "CHUNK_SIZE", 999)
    assert ingest.index_fingerprint()["chunk_size"] == 999


@pytest.mark.filterwarnings("ignore:Relevance scores must be between 0 and 1")
def test_keyword_search_finds_exact_terms(tmp_corpus):
    ingest.build_vectorstore()
    # Fake embeddings are random, so only the BM25 side can rank this correctly.
    result = retriever.search("AA grade points", k=1, hybrid=True)
    assert "AA grade carries 10" in result.chunks[0]["text"]
    assert result.chunks[0]["title"] == "Test Rules"
