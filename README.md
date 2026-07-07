# 🎓 IITB Insti-Assist - Academic Assistant

## Demo

![IITB Insti-Assist Demo](assets/demo.png)

A multi-agent, RAG-grounded assistant for IIT Bombay academic-policy questions
(course registration, grading, academic calendar, exam rules, branch change).

Built for a combined NLP capstone: it satisfies both a **"RAG assistant"**
brief and a **"multi-agent system"** brief at once, by implementing the RAG
pipeline *as* a small team of coordinated agents instead of one monolithic
retrieve-then-generate script.

> ℹ️ **Documents included**: `data/raw/` ships with two real IIT Bombay
> source documents, the **UG Rules & Regulations** (`ugrulebook.pdf`) and the
> **Academic Calendar 2026-27** (`Academic_Calendar_2026-27_FINAL.pdf`). See
> "Adding your documents" below if you want to add more.

---

## 1. Architecture — Supervisor pattern

```
                        ┌────────────┐
           ┌───────────▶│ supervisor │◀─────────────┐
           │            └─────┬──────┘               │
           │      RETRIEVE │ ANSWER │ CRITIQUE │ FINISH
           │                 ▼        ▼         ▼      │
           │          ┌──────────┐┌─────────┐┌────────┐│
           │          │retrieval ││ answer  ││ critic ││
           │          │  agent   ││  agent  ││ agent  ││
           │          └────┬─────┘└────┬────┘└───┬────┘│
           └───────────────┴───────────┴─────────┘
                                                    │
                                                    ▼
                                                ┌────────┐
                                                │ finish │
                                                └────────┘
```

Four distinct agents, one orchestration pattern (**Supervisor**):

| Agent | File | Responsibility |
|---|---|---|
| **Supervisor** | `src/agents/supervisor.py` | Owns all control flow. Looks at the shared state and decides which worker runs next. Never touches documents or drafts text itself. |
| **Retrieval Agent** | `src/agents/retrieval_agent.py` | The *only* agent that talks to the vector store. Embeds the question, does a similarity search, and decides whether anything relevant enough was actually found (`grounded: bool`). |
| **Answer Agent** | `src/agents/answer_agent.py` | Drafts an answer using **only** the chunks the retrieval agent found. Cites its sources. If the critic rejected a previous draft, it revises using that feedback. |
| **Critic Agent** | `src/agents/critic_agent.py` | An independent, skeptical second LLM call that checks the draft against the retrieved context and rejects anything not actually supported by it — this is the hallucination guardrail. |

All four communicate through one shared state object (`src/state.py`) —
a "blackboard" architecture — rather than passing messages directly to each
other. Every worker node routes back to the supervisor when it's done; the
supervisor is the only node allowed to decide what happens next. This is
what makes it a genuine Supervisor pattern rather than a fixed pipeline: the
same "ANSWER" node can be revisited multiple times depending on the critic's
verdict, capped at `max_revisions` (default 2) so it can't loop forever.

### Why RAG needed to be multi-agent here (not just "one agent with extra steps")
A single retrieve-then-generate call has no way to catch its own
hallucinations — if the LLM states something not in the retrieved context,
nothing stops it. Splitting "write the answer" and "check the answer" into
two independent agents with different prompts and different jobs is what
lets the system enforce "don't hallucinate, say I don't know instead" as an
actual architectural guarantee (a second, skeptical pass) rather than just a
hopeful instruction to a single call.

---

## 2. RAG pipeline details

- **Chunking**: `RecursiveCharacterTextSplitter`, 500-char chunks / 100-char
  overlap (`src/rag/ingest.py`). Academic policy text is dense with
  clause-level detail (deadlines, credit limits, percentages), so chunks are
  kept small enough that one retrieved chunk is usually a single
  self-contained rule, with enough overlap that a rule split across a
  paragraph boundary isn't cut in half.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local, free, no
  API key required).
- **Vector store**: ChromaDB, persisted to `data/chroma_db/` (built
  automatically on first run).
- **Groundedness threshold**: the retrieval agent only accepts chunks with a
  relevance score ≥ `0.35` (`src/rag/retriever.py`). If nothing clears that
  bar, `grounded=False` and the system refuses ("I don't know") instead of
  answering from the LLM's own background knowledge.

---

## 3. Setup

```bash
git clone <this-repo-url>
cd iitb-insti-assist
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GOOGLE_API_KEY (get one at https://aistudio.google.com/app/apikey)
```

The first run will download the embedding model (~90MB) and build the
Chroma index automatically — this needs internet access once.

### Adding your documents

`data/raw/` already includes two real source documents:
- `ugrulebook.pdf` — IIT Bombay Rules & Regulations for Undergraduate Programmes
- `Academic_Calendar_2026-27_FINAL.pdf` — Academic Calendar 2026-27

To add more, just drop additional PDFs (or `.txt` files) straight into
`data/raw/` — e.g.:

```
data/raw/
├── ugrulebook.pdf
├── Academic_Calendar_2026-27_FINAL.pdf
└── grading_circular.pdf          # anything else you add
```

Text is extracted per page with `pdfplumber`, so each page becomes its own
retrievable unit and citations can point to a specific page (e.g.
`academic_manual.pdf#p12-chunk-0`), not just a file name.

Then build the index:
```bash
python -m src.rag.ingest
```

Notes:
- If a PDF is a **scanned image** (no selectable text), `pdfplumber` can't
  extract anything from it — you'll see a warning printed for that file.
  Run it through OCR first (e.g. `ocrmypdf input.pdf output.pdf`) and use
  the OCR'd version instead.
- `.txt` files are also supported (loaded as-is) if you ever want to mix in
  plain-text notes alongside PDFs — just drop them in the same folder.
- After adding/removing/replacing files in `data/raw/`, always re-run
  `python -m src.rag.ingest` (or delete `data/chroma_db/` and let it rebuild
  automatically on next run) so the vector store reflects the new documents.

## 4. Running it

**Web UI**
```bash
streamlit run app.py
```

**Rebuild the vector index manually** (e.g. after editing `data/raw/`):
```bash
python -m src.rag.ingest
```

---

## 5. Example run

```
Q: What is the date for pre-registration for Minor Courses in the Autumn semester 2026?

--- Agent trace ---
[supervisor] -> routing to RETRIEVE
[retrieval_agent] found 1 relevant chunk(s) (grounded=True)
[supervisor] -> routing to ANSWER
[answer_agent] wrote initial draft
[supervisor] -> routing to CRITIQUE
[critic_agent] verdict: APPROVED
[supervisor] -> routing to FINISH
[finish] draft approved by critic — returning final answer

--- Final answer ---
Pre-registration for Minor Courses (UG) and High Demand Courses for the
Autumn semester 2026 is from 10 July 2026 (Friday) to 14 July 2026 (Tuesday).

Sources: Academic_Calendar_2026-27_FINAL.pdf

(Verified against: Academic_Calendar_2026-27_FINAL.pdf)
```

An out-of-scope question ("What's the capital of France?") returns nothing
retrieved and a polite refusal instead of a guess.

---

## 6. Project structure

```
iitb-insti-assist/
├── app.py                     # Streamlit UI
├── main.py                    # CLI entry point
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                   # source PDFs (UG Rulebook, Academic Calendar 2026-27)
│   └── chroma_db/             # built automatically, gitignored
├── src/
│   ├── state.py               # shared AgentState (the "blackboard")
│   ├── graph.py                # LangGraph wiring (Supervisor pattern)
│   ├── agents/
│   │   ├── supervisor.py
│   │   ├── retrieval_agent.py
│   │   ├── answer_agent.py
│   │   ├── critic_agent.py
│   │   └── finish_agent.py
│   ├── rag/
│   │   ├── ingest.py           # load, chunk, embed, persist
│   │   └── retriever.py        # similarity search + groundedness threshold
│   └── utils/
│       └── llm.py              # provider-agnostic chat model factory
```

## 7. Troubleshooting

**Streamlit prints a wall of `ModuleNotFoundError: No module named 'torchvision'` on startup.**
This is harmless. Streamlit's file-watcher tries to inspect every
submodule `transformers` *could* lazily import (including obscure
vision-model code that needs `torchvision`), which you don't have and don't
need — we only use `transformers` indirectly for text embeddings via
`sentence-transformers`. It's noisy console output, not a crash; the app
still runs. To silence it, either:
- run with the watcher off: `streamlit run app.py --server.fileWatcherType none`, or
- set `STREAMLIT_SERVER_FILE_WATCHER_TYPE=none` as an environment variable.

**`FileNotFoundError: No .txt or .pdf documents found in data/raw`**
You haven't added any source documents yet — see "Adding your documents"
above.

**A retrieved answer is always "I don't know" even for something you know is in a PDF.**
Check that `pdfplumber` could actually extract text from that PDF (rerun
`python -m src.rag.ingest` and watch for a "no extractable text found"
warning) — if the PDF is a scanned image, it needs OCR first.

---

## Author

**Karan Bansal**  
Roll No.: **24B3003**
