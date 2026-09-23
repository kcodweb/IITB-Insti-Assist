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
  (e.g. `ugrulebook.pdf#p29-chunk-1`), plus a `page` number for PDF-derived
  chunks, which is what powers the "Sources" display in the UI — the
  assistant always shows exactly which document and page it drew from.

**v2 update.** Once a labelled eval set existed (section 4), I could test
these choices instead of reasoning about them. Two things changed:

- **Section headers on every chunk.** Each chunk is now prefixed with
  `[document | section]`, e.g. `[Academic Calendar 2026-27 | (B) SCHEDULE
  FOR SPRING SEMESTER]`. The calendar's Autumn and Spring tables use
  identical wording, so a bare chunk of dates from page 7 carried no clue
  which semester it belonged to. The header restores that context for both
  the retriever and the LLM. The ingester also strips running footers and
  page numbers, and skips table-of-contents pages.
- **800 / 150 instead of 500 / 100.** With the header providing context,
  slightly larger chunks retrieved better (Hit@4 87% → 92% with the same
  embedding model). Many rules, such as the academic-standing categories
  and the branch-change criteria, span several sentences that only make
  sense together.

## 4. Evaluation (v2)

`eval/questions.jsonl` has 68 hand-labelled questions: 52 answerable (each
labelled with the page(s) that answer it), 8 out-of-scope, and 8 that sound
in-scope but aren't covered by the documents. `python -m eval.run_eval`
reports Hit@k / MRR and sweeps the relevance threshold.

| Configuration | Hit@1 | Hit@4 | MRR |
|---|---|---|---|
| v1: MiniLM, raw pages, 500/100, dense | 77% | 88% | 0.817 |
| v2: bge-small, cleaned + section headers, 800/150, hybrid BM25 | 87% | 100% | 0.926 |

Two findings surprised me:
- **The old 0.35 threshold was not a cosine similarity.** The Chroma
  collection used its default squared-L2 distance, which LangChain turns
  into a "relevance" score that can go negative (off-topic questions scored
  around −0.14). The collection now uses cosine distance, so the threshold
  means what it says. It was re-tuned on the eval set to 0.55: every
  answerable question passes, and 7 of 8 off-topic questions are refused
  before any LLM call.
- **Keyword search mattered as much as the embedding model.** Adding BM25
  (fused by Reciprocal Rank Fusion) was worth +4–6 points of Hit@4 on its
  own. Questions like "What CPI is needed to apply for IDDDP?" hinge on
  acronyms that small embedding models blur.

## 5. Multi-agent architecture

Four agents, coordinated with the **Supervisor** orchestration pattern:

1. **Supervisor** — deterministic Python routing logic (not an LLM call) that
   reads the shared state and decides which of the other three runs next.
2. **Retrieval Agent** — the only agent with vector-store access; rewrites
   follow-up questions into standalone queries (v2), runs a hybrid search,
   and applies a relevance threshold (0.55 cosine in v2) to decide whether
   the question is actually "grounded" in the knowledge base at all.
3. **Answer Agent** — drafts an answer using only the retrieved chunks,
   citing each claim inline as `[1]`, `[2]` (v2); revises if the critic
   rejects the draft.
4. **Critic Agent** — the check on the answer agent. In v2 it runs a
   deterministic pass first (citations must point at real passages;
   numbers that appear nowhere in the sources are flagged), then an
   independent LLM review, and returns a structured `{approved, critique}`
   verdict.

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
revision-budget-exhausted scenarios). The v1 submission left the harness out
to keep the repo lean. It is back in v2 as `tests/test_graph.py`, alongside
tests for ingestion, index rebuilds and critic parsing. The fix itself
(version counters) remains in `src/state.py`, `src/agents/answer_agent.py`,
and `src/agents/critic_agent.py`.

## 6. Known limitations / what I'd improve with more time

Items marked ✅ were addressed in v2.

- **PDF text quality depends on the source**: `pdfplumber` handles
  text-based PDFs well but can't extract anything from scanned-image PDFs
  without OCR first. Both bundled PDFs are text-based and extracted
  cleanly (64/64 pages produced usable text with no OCR needed), but this
  is worth re-checking for any additional documents added later.
- **Critic reliability** ✅ *(partly)*: the critic is itself an LLM call and
  isn't a perfect fact-checker. v2 adds a deterministic layer in front of
  it: invalid citations are rejected outright, and numbers missing from the
  sources are flagged to the LLM critic. The number check is deliberately a
  hint rather than a hard rule, because the sources sometimes spell numbers
  out ("eighty percent"). Proper entailment checking (e.g. a small NLI model
  per cited sentence) would be the next step.
- **Static relevance threshold** ✅: now tuned on the labelled eval set
  (section 4) instead of by manual inspection. One limit remains: bge-small
  compresses similarity scores into a narrow band, so off-topic and
  on-topic questions overlap slightly, and 1 of 8 off-topic eval questions
  still passes the gate. The answer and critic agents catch it, at the cost
  of an LLM call.
- **No conversation memory** ✅: the retrieval agent now rewrites follow-ups
  such as "what about for mid-sems?" into standalone queries using the last
  few turns, and the answer agent sees the conversation too.
- **Eval set is small and self-written**: 68 questions written against these
  two documents, and the configuration was chosen on the same set. The
  numbers are indicative, not a held-out benchmark. The end-to-end mode
  (`--e2e`) checks final answers but needs an API key to run.
- **Calendar tables**: `pdfplumber` flattens the calendar's two-column
  tables, so an event name and its date can land on different lines. The
  LLM copes, but table-aware extraction (`page.extract_tables()`) would give
  cleaner chunks.
- **Single domain**: only Academic scope is implemented. The architecture
  is set up so a second scope (e.g. Hostel Life) could be added as a second
  Chroma collection with the supervisor picking which collection(s) to
  search, but that routing isn't implemented yet.
