"""
06_retrieve_context.py
------------------------
STAGE 6 of the RAG pipeline: Retrieval.

Given a user question, embeds it with the SAME embedding model used in Stage 4,
then searches the Chroma store from Stage 5 for the most similar chunks.

Run standalone to test retrieval by itself (no LLM call, just shows what would
be retrieved for a sample question):
    python 06_retrieve_context.py

Can also be imported by 07_prompting.py.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "scamshield_docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

_embedder = None
_collection = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def get_collection():
    global _collection
    if _collection is None:
        if not os.path.isdir(CHROMA_DIR):
            raise SystemExit(f"'{CHROMA_DIR}/' not found. Run 05_create_chroma_store.py first.")
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(query, top_k=TOP_K):
    """Return the top_k most similar chunks to `query` as a list of dicts:
    {"text", "source", "page", "distance"}."""
    embedder = get_embedder()
    collection = get_collection()

    query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    retrieved = []
    for text, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved.append({
            "text": text,
            "source": metadata["source"],
            "page": metadata["page"],
            "distance": distance,
        })
    return retrieved


def main():
    sample_query = "I got an email saying my password expires today, is this phishing?"
    print(f"Test query: {sample_query!r}\n")
    results = retrieve(sample_query)
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['source']} - page {r['page']} (distance={r['distance']:.4f})")
        print(r["text"][:300] + "...\n")


if __name__ == "__main__":
    main()
