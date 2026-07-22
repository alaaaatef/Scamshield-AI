"""
03_chunking.py
--------------
STAGE 3 of the RAG pipeline: Chunking.

Splits each cleaned page's text into overlapping word-based chunks, so that
retrieval later can pull back small, focused pieces of text instead of whole
pages.

Run standalone:
    python 03_chunking.py

Input:
    pipeline_data/02_cleaned_documents.pkl
Output:
    pipeline_data/03_chunks.pkl
    -> a list of dicts: {"chunk_id": <int>, "source": <filename>, "page": <int>, "text": <chunk text>}
"""

import os
import pickle

INPUT_FILE = "pipeline_data/02_cleaned_documents.pkl"
OUTPUT_FILE = "pipeline_data/03_chunks.pkl"

CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40


def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_chunks(cleaned_documents):
    chunks = []
    chunk_id = 0
    for doc in cleaned_documents:
        for piece in chunk_text(doc["text"]):
            chunks.append({
                "chunk_id": chunk_id,
                "source": doc["source"],
                "page": doc["page"],
                "text": piece,
            })
            chunk_id += 1
    return chunks


def main():
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"'{INPUT_FILE}' not found. Run 02_preprocessing.py first.")

    with open(INPUT_FILE, "rb") as f:
        cleaned_documents = pickle.load(f)

    print(f"Chunking {len(cleaned_documents)} cleaned pages "
          f"(chunk size={CHUNK_SIZE_WORDS} words, overlap={CHUNK_OVERLAP_WORDS} words)...")
    chunks = build_chunks(cleaned_documents)
    print(f"Created {len(chunks)} chunks.")

    os.makedirs("pipeline_data", exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved chunks to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
