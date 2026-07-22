"""
07_prompting.py
-----------------
STAGE 7 of the RAG pipeline: Prompt Construction + Generation.

Takes retrieved chunks from Stage 6, builds a structured prompt, and sends it
to a free LLM (Llama 3.3 70B via the Groq API) to produce the final,
source-grounded answer.

Run standalone to test the full pipeline end-to-end from the terminal
(retrieval + generation, no Streamlit needed):
    python 07_prompting.py

Can also be imported by streamlit_app.py.
"""

import os
import importlib.util

# --- Dynamically import retrieve_context.py -------------------------------
# (Python module names can't start with a digit, so we can't do a normal
#  `import retrieve_context` - this loads it by file path instead.)
_this_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "retrieve_context", os.path.join(_this_dir, "retrieve_context.py")
)
retrieve_context = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retrieve_context)
# ------------------------------------------------------------------------------

from groq import Groq

GROQ_MODEL = "llama-3.3-70b-versatile"  # free-tier model on Groq

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


def get_groq_client(api_key=None):
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "No Groq API key found. Set the GROQ_API_KEY environment variable "
            "(or pass api_key explicitly)."
        )
    return Groq(api_key=key)


def build_context_block(results):
    lines = []
    for r in results:
        lines.append(f"[{r['source']} - page {r['page']}]\n{r['text']}")
    return "\n\n---\n\n".join(lines)


def generate_answer(query, api_key=None, top_k=None):
    """Full RAG call: retrieve relevant chunks, build the prompt, call the LLM.
    Returns (answer_text, retrieved_results)."""
    results = retrieve_context.retrieve(query, top_k=top_k or retrieve_context.TOP_K)
    context_block = build_context_block(results)

    client = get_groq_client(api_key)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Retrieved context:\n{context_block}\n\nUser question:\n{query}",
            },
        ],
        temperature=0.2,
        max_tokens=700,
    )
    answer = completion.choices[0].message.content
    return answer, results


def main():
    print("ScamShield AI - terminal test (Stages 6 + 7 combined)")
    print("Type a question, or press Enter with no text to quit.\n")
    while True:
        query = input("Your question: ").strip()
        if not query:
            break
        answer, results = generate_answer(query)
        print("\n" + "=" * 60)
        print(answer)
        print("=" * 60)
        print("\nSources used:")
        for r in results:
            print(f"  - {r['source']} (page {r['page']})")
        print()


if __name__ == "__main__":
    main()
