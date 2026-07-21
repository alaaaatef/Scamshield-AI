"""
app.py
------
ScamShield AI: a Streamlit chat app that answers phishing / scam questions
using ONLY retrieved passages from your trusted PDFs (RAG), citing sources.

Free stack:
  - Embeddings: sentence-transformers (local, no API key)
  - Vector search: FAISS (local)
  - LLM: Groq API (free tier - get a key at https://console.groq.com/keys)

Run locally:
    streamlit run app.py
"""

import os
import pickle
import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

STORE_DIR = "vector_store"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"  # free-tier model on Groq
TOP_K = 5

SYSTEM_PROMPT = """You are ScamShield AI, a cybersecurity assistant that helps everyday users
figure out whether a message, email, call, or link is a phishing attempt or scam.

Rules:
1. Use ONLY the information in the "Retrieved context" below to answer. If the context does not
   contain enough information to answer confidently, say so plainly instead of guessing.
2. Never invent a source. Only cite documents that actually appear in the retrieved context.
3. Always structure your answer using these exact section headers:

Attack Type: <short label, e.g. "Phishing Email", "Tech Support Scam", "Not enough information">
Risk Level: <LOW / MEDIUM / HIGH>
Explanation: <2-4 sentences in plain language>
Warning Signs: <bullet list of red flags relevant to this case>
Recommended Actions: <bullet list of concrete next steps>
Sources: <list of document names used>

Keep the tone calm, clear, and non-technical enough for a general audience (including older adults).
"""


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_index():
    index_path = os.path.join(STORE_DIR, "index.faiss")
    chunks_path = os.path.join(STORE_DIR, "chunks.pkl")
    if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
        return None, None, None
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        data = pickle.load(f)
    return index, data["chunks"], data["metadata"]


def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    if not api_key:
        st.error(
            "No Groq API key found. Add GROQ_API_KEY to Streamlit secrets "
            "(or set it as an environment variable) to enable answers."
        )
        st.stop()
    return Groq(api_key=api_key)


def retrieve(query, embedder, index, chunks, metadata, top_k=TOP_K):
    query_vec = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    scores, ids = index.search(query_vec.astype(np.float32), top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append({
            "text": chunks[idx],
            "source": metadata[idx]["source"],
            "page": metadata[idx]["page"],
            "score": float(score),
        })
    return results


def build_context_block(results):
    lines = []
    for r in results:
        lines.append(f"[{r['source']} - page {r['page']}]\n{r['text']}")
    return "\n\n---\n\n".join(lines)


def main():
    st.set_page_config(page_title="ScamShield AI", page_icon="🛡️", layout="centered")
    st.title("🛡️ ScamShield AI")
    st.caption(
        "Ask about a suspicious email, text, call, or link. Answers are grounded only in the "
        "trusted cybersecurity guides loaded into this assistant (CISA, FTC, Microsoft, US-CERT)."
    )

    index, chunks, metadata = load_index()
    if index is None:
        st.warning(
            "No knowledge base found yet. Run `python ingest.py` locally first "
            "(after placing your PDFs in the `data/` folder), then redeploy with the "
            "generated `vector_store/` folder included."
        )
        st.stop()

    embedder = load_embedder()
    client = get_groq_client()

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
                results = retrieve(user_input, embedder, index, chunks, metadata)
                context_block = build_context_block(results)

                completion = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Retrieved context:\n{context_block}\n\n"
                                f"User question:\n{user_input}"
                            ),
                        },
                    ],
                    temperature=0.2,
                    max_tokens=700,
                )
                answer = completion.choices[0].message.content

            st.markdown(answer)

            with st.expander("Show retrieved sources"):
                for r in results:
                    st.markdown(f"**{r['source']}** (page {r['page']}, similarity {r['score']:.2f})")
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
