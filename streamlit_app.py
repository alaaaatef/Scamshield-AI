"""
streamlit_app.py
-----------------
The user-facing chat app. Ties together the full RAG pipeline (Stages 1-7)
by calling into prompting.py, which in turn uses 06_retrieve_context.py
and the Chroma store built in 05_create_chroma_store.py.

IMPORTANT - run the pipeline scripts in order at least once before running
this app:
    python documents.py
    python preprocessing.py
    python chunking.py
    python vector_representation.py
    python create_chroma_store.py

Then run the app:
    streamlit run streamlit_app.py
"""

import os
import importlib.util
import streamlit as st

# --- Dynamically import prompting.py --------------------------------------
# (Python module names can't start with a digit, so we load it by file path.)
_this_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "prompting", os.path.join(_this_dir, "prompting.py")
)
prompting = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prompting)
# ------------------------------------------------------------------------------

CHROMA_DIR = "chroma_store"


def get_api_key():
    """Read the Groq key from Streamlit secrets (cloud) or an environment
    variable (local), and export it so prompting.py can find it."""
    key = None
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass
    key = key or os.environ.get("GROQ_API_KEY")
    return key


def main():
    st.set_page_config(page_title="ScamShield AI", page_icon="🛡️", layout="centered")
    st.title("🛡️ ScamShield AI")
    st.caption(
        "Ask about a suspicious email, text, call, or link. Answers are grounded only in the "
        "trusted cybersecurity guides loaded into this assistant (CISA, FTC, Microsoft, US-CERT)."
    )

    if not os.path.isdir(CHROMA_DIR):
        st.warning(
            "No knowledge base found yet. Run the pipeline scripts in order first: "
            "`01_documents.py` -> `02_preprocessing.py` -> `03_chunking.py` -> "
            "`04_vector_representation.py` -> `05_create_chroma_store.py`, then reload this app."
        )
        st.stop()

    api_key = get_api_key()
    if not api_key:
        st.error(
            "No Groq API key found. Add GROQ_API_KEY to Streamlit secrets "
            "(or set it as an environment variable) to enable answers."
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Paste the suspicious message or describe what happened...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Checking against trusted sources..."):
                answer, results = prompting.generate_answer(user_input, api_key=api_key)
            st.markdown(answer)

            with st.expander("Show retrieved sources"):
                for r in results:
                    st.markdown(f"**{r['source']}** (page {r['page']}, distance {r['distance']:.3f})")
                    st.caption(r["text"][:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})

    st.divider()
    st.caption(
        "⚠️ This tool provides general educational guidance, not a guarantee. "
        "If you believe you've been scammed, contact your bank immediately and report it at "
        "reportfraud.ftc.gov."
    )


if __name__ == "__main__":
    main()
