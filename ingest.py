"""
ingest.py
---------
Reads every PDF in ./data, splits the text into overlapping chunks,
embeds each chunk with a free local embedding model (no API key needed),
and saves a FAISS vector index + the chunk metadata to ./vector_store.

Run this ONCE (and again any time you add/change PDFs):
    python ingest.py
"""

import os
import pickle
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

DATA_DIR = "data"
STORE_DIR = "vector_store"
CHUNK_SIZE_WORDS = 220      # ~ roughly 350-450 tokens, good for retrieval
CHUNK_OVERLAP_WORDS = 40
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # free, runs locally, ~80MB download


def load_pdf_text(path):
    """Return a list of (page_number, text) tuples for a single PDF."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())  # collapse whitespace/newlines
        if text.strip():
            pages.append((i + 1, text))
    return pages


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


def build_index():
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"Folder '{DATA_DIR}/' not found. Create it and put your PDFs inside.")

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise SystemExit(f"No PDFs found in '{DATA_DIR}/'. Add your source PDFs first.")

    print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")
    print("Loading embedding model (first run downloads ~80MB, then it's cached)...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    all_chunks = []      # the actual text of each chunk
    all_metadata = []     # dict per chunk: {source, page}

    for fname in pdf_files:
        path = os.path.join(DATA_DIR, fname)
        print(f"Reading {fname} ...")
        pages = load_pdf_text(path)
        for page_num, page_text in pages:
            for chunk in chunk_text(page_text):
                all_chunks.append(chunk)
                all_metadata.append({"source": fname, "page": page_num})

    print(f"Created {len(all_chunks)} chunks total. Embedding them now...")
    embeddings = model.encode(
        all_chunks,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so we can use inner-product = cosine similarity
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    index.add(embeddings.astype(np.float32))

    os.makedirs(STORE_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(STORE_DIR, "index.faiss"))

    with open(os.path.join(STORE_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadata": all_metadata}, f)

    print(f"Done! Saved index + {len(all_chunks)} chunks to '{STORE_DIR}/'.")


if __name__ == "__main__":
    build_index()
