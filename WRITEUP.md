# IITB Insti-Assist — Project Write-Up

## 1. Chosen scope and why

I built the **Academic Assistant** scope: course registration, grading
policy, the academic calendar, and exam rules. I chose this over Hostel/Club
because academic policy text is unusually well-suited to demonstrating *why*
RAG needs groundedness checking: it's full of specific, easy-to-hallucinate
numbers (attendance percentages, credit limits, deadline weeks, grade-point
values). A generic LLM asked "what's the minimum attendance to sit an exam
at IIT Bombay" will confidently produce *a* plausible-sounding number — the
interesting engineering problem is making sure the system only ever repeats
a number that's actually in a real source document, and says "I don't know"
otherwise. That also made it a natural fit to combine with a separate
multi-agent-systems assignment: instead of one retrieve-then-generate
script, I split "retrieve", "draft", and "verify against source" into three
distinct agents coordinated by a Supervisor, so the anti-hallucination
behavior is an architectural property (an independent second pass) rather
than just an instruction hoped for in one prompt.

## 2. Data sources used

I used **2 real, official IIT Bombay source documents** in `data/raw/`:

1. **`ugrulebook.pdf`** — *"Rules & Regulations for Undergraduate Programmes"*
   (applicable to B.Tech., B.S., B.Des., Dual Degree students; updated January
   2025). This is the primary rulebook covering registration, credit
   structure, grading (SPI/CPI), examinations, academic standing, branch
   change, and degree requirements — 52 pages of extractable text.
2. **`Academic_Calendar_2026-27_FINAL.pdf`** — the official Academic Calendar
   for 2026-27, giving exact semester-wise dates for registration, add/drop,
   examinations, grade submission, and vacations — 12 pages of extractable
   text.

Both are genuine institute documents, obtained directly rather than
reconstructed, which matters for a system whose entire value proposition is
"only answer from real, retrievable sources." Together they cover the two
things a student typically needs precise, easy-to-get-wrong answers for:
*what the rule is* (rulebook) and *when it applies* (calendar) — e.g. "how
is CPI calculated" pulls from the rulebook, while "when does add/drop close
this semester" pulls from the calendar.

Text is extracted per PDF page via `pdfplumber` in `src/rag/ingest.py`, with
page numbers preserved as metadata so every retrieved chunk — and therefore
every answer — can cite not just which document but which page it came
from (e.g. `ugrulebook.pdf#p15-chunk-0`, `Academic_Calendar_2026-27_FINAL.pdf#p2-chunk-4`).
Both PDFs were text-based (not scanned images), so no OCR step was needed —
`pdfplumber` extracted clean text directly from all 64 pages across the two
files.

## 3. Chunking strategy and why

- **Splitter**: `RecursiveCharacterTextSplitter` (character-based, not
  token-based) from `langchain-text-splitters`.
- **Chunk size**: 500 characters, **overlap**: 100 characters.
- **Reasoning**: Policy documents in this domain are dense with
  clause-level facts — a single sentence often *is* the entire answer to a
  question ("attendance must be ≥80%", "add/drop deadline is week 2"). A
  500-character chunk is roughly one to three sentences, which keeps each
  retrieved chunk close to a single self-contained rule rather than
  smearing several unrelated rules into one chunk (which would make the
  critic's job of verifying "is this claim actually in the context" much
  harder). The 100-character overlap exists specifically so that a rule
  sitting right at a paragraph boundary isn't split in half between two
  chunks, losing its subject or its number.
- **Metadata**: every chunk keeps `source` (file name) and a `chunk_id`
  (e.g. `exam_rules.pdf#p3-chunk-1`), plus a `page` number for PDF-derived
  chunks, which is what powers the "Sources used" / "retrieved chunks"
  display in the UI — the assistant always shows exactly which document
  and page it drew from.

## 4. Multi-agent architecture

Four agents, coordinated with the **Supervisor** orchestration pattern:

1. **Supervisor** — deterministic Python routing logic (not an LLM call) that
   reads the shared state and decides which of the other three runs next.
2. **Retrieval Agent** — the only agent with vector-store access; runs a
   similarity search and applies a relevance threshold (0.35) to decide
   whether the question is actually "grounded" in the knowledge base at all.
3. **Answer Agent** — drafts an answer using only the retrieved chunks,
   citing sources; revises if the critic rejects the draft.
4. **Critic Agent** — an independent LLM call whose only job is to check the
   draft against the retrieved context and reject anything unsupported,
   returning a structured `{approved, critique}` verdict.

State is threaded through all four via one shared `AgentState` TypedDict
(a "blackboard"), with a `draft_version` / `critiqued_version` pair used to
track whether the *current* draft has been reviewed yet — this was a real
bug I hit and fixed during testing: LangGraph merges a node's returned dict
into state key-by-key, so omitting a key from a return value does **not**
delete it, it just leaves the old value in place. My first implementation
tried to "clear" the critic's old verdict after a revision by leaving
`approved`/`critique` out of the answer agent's return value, which silently
left the stale verdict in place and caused an infinite Answer→Answer loop
instead of ever reaching Critique again. Version counters fixed it cleanly.
This was caught and fixed during development using a small mocked-LLM test
harness (routing logic tested deterministically against the happy path,
refusal-when-ungrounded, one-revision-then-approved, and
revision-budget-exhausted scenarios), which was removed from this submitted
version of the repo to keep it lean but is easy to re-add if useful for
grading — the fix itself (version counters) remains in `src/state.py`,
`src/agents/answer_agent.py`, and `src/agents/critic_agent.py`.

## 5. Known limitations / what I'd improve with more time

- **PDF text quality depends on the source**: `pdfplumber` handles
  text-based PDFs well but can't extract anything from scanned-image PDFs
  without OCR first. Both bundled PDFs are text-based and extracted
  cleanly (64/64 pages produced usable text with no OCR needed), but this
  is worth re-checking for any additional documents added later.
- **Critic reliability**: the critic is itself an LLM call and isn't a
  perfect fact-checker — it reliably catches clear numeric/factual
  mismatches but could miss a subtler misreading of the context. A stronger
  version would use a smaller, cheaper, deterministic verifier (e.g.
  string/entailment matching against the retrieved chunks) alongside the
  LLM critic.
- **Static relevance threshold**: 0.35 was chosen by manual inspection
  rather than tuned against a labeled set of
  in-scope/out-of-scope questions. With more time I'd build a small eval set
  and sweep the threshold against precision/recall on "should refuse" vs.
  "should answer" questions.
- **No conversation memory**: each question is independent; a follow-up like
  "what about for mid-sems?" doesn't know what "what" refers to. Adding a
  short rolling history to the retrieval query would fix this.
- **Single domain**: only Academic scope is implemented. The architecture
  is set up so a second scope (e.g. Hostel Life) could be added as a second
  Chroma collection with the supervisor picking which collection(s) to
  search, but that routing isn't implemented yet.
