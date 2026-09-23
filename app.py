"""
Streamlit chat UI for IITB Insti-Assist.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src import config  # noqa: E402
from src.graph import stream  # noqa: E402
from src.rag.ingest import get_vectorstore, read_manifest  # noqa: E402
from src.rag.retriever import warm_up  # noqa: E402
from src.utils import llm as llm_utils  # noqa: E402

st.set_page_config(page_title="IITB Insti-Assist", page_icon="🎓")

EXAMPLES = [
    "What attendance do I need to avoid a DX grade?",
    "When are the Spring 2027 end-semester exams?",
    "How many credits do I need for a minor?",
    "What CPI do I need to apply for IDDDP?",
]

STATUS_BADGES = {
    "verified": ":green-badge[:material/verified: Checked by the critic agent]",
    "unverified": ":orange-badge[:material/warning: Critic still had concerns]",
    "refused": ":gray-badge[:material/block: Not covered by the documents]",
}

STEP_LABELS = {
    "[supervisor]": None,
    "[retrieval_agent]": "Searching the rulebook and calendar…",
    "[answer_agent]": "Drafting an answer…",
    "[critic_agent]": "Critic is fact-checking the draft…",
    "[finish]": "Wrapping up…",
}


@st.cache_resource(show_spinner="Loading the knowledge base (the first run builds the index)…")
def load_knowledge_base() -> dict:
    get_vectorstore()
    warm_up()
    return read_manifest() or {}


def _chunk_body(chunk: dict) -> str:
    """The chunk text without its "[document | section]" header line."""
    text = chunk["text"]
    return text.split("\n", 1)[1] if text.startswith("[") and "\n" in text else text


def _cited_chunks(state: dict) -> list[tuple[int, dict]]:
    chunks = state.get("retrieved_chunks") or []
    return [(n, chunks[n - 1]) for n in state.get("citations") or [] if 1 <= n <= len(chunks)]


def render_answer(message: dict, show_trace: bool) -> None:
    st.markdown(message["content"])
    meta = message.get("meta", {})

    if meta.get("status"):
        st.markdown(STATUS_BADGES[meta["status"]])

    if meta.get("cited"):
        with st.expander(f"Sources ({len(meta['cited'])})", icon=":material/menu_book:"):
            for n, chunk in meta["cited"]:
                page = f" · page {chunk['page']}" if "page" in chunk else ""
                st.markdown(f"**[{n}] {chunk['title']}{page}**")
                if chunk.get("section"):
                    st.caption(chunk["section"])
                st.text(_chunk_body(chunk))

    if show_trace and meta.get("trace"):
        with st.expander("Agent trace", icon=":material/account_tree:"):
            if meta.get("search_query") and meta["search_query"] != meta.get("question"):
                st.markdown(f"Searched for: *{meta['search_query']}*")
            st.code("\n".join(meta["trace"]), language=None, wrap_lines=True)
            for i, chunk in enumerate(meta.get("retrieved", []), start=1):
                score = f"{chunk['score']:.3f}" if chunk.get("score") is not None else "keyword match"
                st.caption(f"[{i}] {chunk['chunk_id']} — similarity {score}")


def answer(question: str, history: list[dict], max_revisions: int, show_trace: bool) -> dict:
    """Run the agent graph with a live status box; return the assistant message."""
    state: dict = {}
    with st.status("Supervisor is coordinating the agents…", expanded=show_trace) as status:
        for lines, state in stream(question, chat_history=history, max_revisions=max_revisions):
            for line in lines:
                label = next((v for k, v in STEP_LABELS.items() if line.startswith(k)), None)
                if label:
                    status.update(label=label)
                status.write(line)
        steps = sum(1 for line in state.get("history", []) if not line.startswith("[supervisor]"))
        status.update(label=f"Done — {steps} agent steps", state="complete", expanded=False)

    return {
        "role": "assistant",
        "content": state.get("final_answer") or "(no answer produced)",
        "meta": {
            "question": question,
            "status": state.get("status"),
            "cited": _cited_chunks(state),
            "retrieved": state.get("retrieved_chunks") or [],
            "search_query": state.get("search_query"),
            "trace": state.get("history", []),
        },
    }


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []

manifest = load_knowledge_base()
missing_key = llm_utils.missing_api_key()

with st.sidebar:
    st.header("🎓 Insti-Assist")
    if st.button("New chat", icon=":material/add_comment:", width="stretch"):
        st.session_state.messages = []
        st.session_state.pop("example", None)
        st.rerun()

    st.subheader("Knowledge base")
    for doc in manifest.get("documents", {}).values():
        st.markdown(f"**{doc['title']}**  \n{doc['pages']} pages · {doc['chunks']} passages")
        if doc.get("description"):
            st.caption(doc["description"])

    st.subheader("Settings")
    show_trace = st.toggle("Show agent trace", value=False)
    max_revisions = st.slider(
        "Max critic revisions", 0, 3, config.MAX_REVISIONS,
        help="How many times the answer agent may rewrite a draft the critic rejects.",
    )

    with st.expander("How it works"):
        st.markdown(
            """
- Answers come **only** from the documents listed above: the official IIT Bombay
  UG Rules & Regulations and the Academic Calendar 2026-27.
- A **Supervisor** routes between a **Retrieval Agent** (hybrid keyword + semantic
  search), an **Answer Agent** that cites every claim, and a **Critic Agent** that
  rejects anything the cited passages don't support.
- If nothing relevant is found, it says **"I don't know"** instead of guessing.
- Always double-check anything important with the Academic Office.
"""
        )
    st.caption(f"LLM: {llm_utils.provider()} / {llm_utils.model_name()}  \nEmbeddings: {config.EMBEDDING_MODEL}")

st.title("🎓 IITB Insti-Assist")
st.caption(
    "Ask about IIT Bombay UG academic rules: registration, grading, exams, "
    "branch change, and the 2026-27 academic calendar."
)

if missing_key:
    st.error(
        f"`{missing_key}` isn't set. Copy `.env.example` to `.env`, add your key, "
        "and restart the app.",
        icon=":material/key:",
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_answer(message, show_trace)
        else:
            st.markdown(message["content"])

prompt = None
if not st.session_state.messages:
    choice = st.pills("Try asking", EXAMPLES, key="example")
    if choice:
        prompt = choice

typed = st.chat_input("Ask about IITB academic rules or dates…", disabled=bool(missing_key))
prompt = typed or prompt

if prompt and not missing_key:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            reply = answer(prompt, history, max_revisions, show_trace)
        except Exception as e:  # surfaced to the user rather than a stack trace
            st.error(f"Something went wrong while running the agents: {e}")
            st.session_state.messages.pop()
        else:
            render_answer(reply, show_trace)
            st.session_state.messages.append(reply)
