"""
04_vector_representation.py
----------------------------
STAGE 4 of the RAG pipeline: Vector Representation (Embeddings).

Loads the chunks from Stage 3 and converts each one into a numeric vector
using a free, local embedding model (sentence-transformers). No API key or
internet connection is needed after the model's first download.

Run standalone:
    python 04_vector_representation.py

Input:
    pipeline_data/03_chunks.pkl
Output:
    pipeline_data/04_embeddings.pkl
    -> dict: {"chunks": [...], "embeddings": <numpy array, shape (n_chunks, dim)>}
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FILE = "pipeline_data/03_chunks.pkl"
OUTPUT_FILE = "pipeline_data/04_embeddings.pkl"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # free, local, ~80MB download on first run


def embed_chunks(chunks, model_name=EMBED_MODEL_NAME):
    print("Loading embedding model (first run downloads ~80MB, then it's cached)...")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so cosine similarity = dot product later
    )
    return embeddings.astype(np.float32)


def main():
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"'{INPUT_FILE}' not found. Run 03_chunking.py first.")

    with open(INPUT_FILE, "rb") as f:
        chunks = pickle.load(f)

    embeddings = embed_chunks(chunks)
    print(f"Produced embeddings with shape {embeddings.shape}.")

    os.makedirs("pipeline_data", exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)
    print(f"Saved embeddings to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
