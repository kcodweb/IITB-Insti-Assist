"""
Streamlit UI for IITB Insti-Assist.

Run with:
    streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.graph import run

st.set_page_config(page_title="IITB Insti-Assist — Academic", page_icon="🎓")

st.title("🎓 IITB Insti-Assist")
st.caption(
    "A multi-agent RAG assistant for IIT Bombay academic policy questions "
    "(course registration, grading, academic calendar, exam rules)."
)

with st.expander("ℹ️ About this assistant / limitations", expanded=False):
    st.markdown(
        """
- This assistant answers **only** from the documents in `data/raw/`.
- The bundled documents are **sample/illustrative placeholders** written for
  this project, not scraped official IIT Bombay documents — replace them
  with real institute sources before trusting the answers.
- If the answer isn't in the documents, it will say **"I don't know"**
  instead of guessing.
- Architecture: a **Supervisor** agent routes between a **Retrieval Agent**,
  an **Answer Agent**, and a **Critic Agent** that checks every draft for
  hallucination before it's shown to you.
"""
    )

question = st.text_input(
    "Ask a question about IITB academics:",
    placeholder="e.g. What is the minimum attendance required to sit an exam?",
)

show_trace = st.checkbox("Show agent trace (debug view)", value=False)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Supervisor is coordinating the agents..."):
        try:
            result = run(question)
        except Exception as e:  # surfaced to the user rather than a stack trace
            st.error(f"Something went wrong running the agent graph: {e}")
            result = None

    if result:
        st.markdown("### Answer")
        st.write(result.get("final_answer", "(no answer produced)"))

        sources = result.get("sources_used", [])
        if sources:
            st.markdown("**Sources used:** " + ", ".join(sources))

        chunks = result.get("retrieved_chunks", [])
        if chunks:
            with st.expander("📄 Retrieved chunks (what the agents saw)"):
                for c in chunks:
                    page_note = f" — page {c['page']}" if "page" in c else ""
                    st.markdown(f"**{c['chunk_id']}**{page_note} (score: {c['score']})")
                    st.text(c["text"])
                    st.divider()

        if show_trace:
            with st.expander("🔎 Agent trace", expanded=True):
                for line in result.get("history", []):
                    st.text(line)
