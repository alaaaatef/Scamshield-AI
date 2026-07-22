"""
05_create_chroma_store.py
--------------------------
STAGE 5 of the RAG pipeline: Vector Database Storage.

Loads the chunks + embeddings from Stage 4 and stores them in a persistent
ChromaDB collection on disk, so later stages (and the app) can search over
them without recomputing anything.

Run standalone:
    python 05_create_chroma_store.py

Input:
    pipeline_data/04_embeddings.pkl
Output:
    chroma_store/   (a persistent ChromaDB directory)
"""

import os
import pickle
import chromadb

INPUT_FILE = "pipeline_data/04_embeddings.pkl"
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "scamshield_docs"


def build_chroma_store(chunks, embeddings, persist_dir=CHROMA_DIR, collection_name=COLLECTION_NAME):
    client = chromadb.PersistentClient(path=persist_dir)

    # Start clean each time this script is run, so re-running it doesn't duplicate entries
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [str(c["chunk_id"]) for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )
    return collection


def main():
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"'{INPUT_FILE}' not found. Run 04_vector_representation.py first.")

    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)

    print(f"Building Chroma store with {len(data['chunks'])} chunks...")
    collection = build_chroma_store(data["chunks"], data["embeddings"])
    print(f"Chroma collection '{COLLECTION_NAME}' now has {collection.count()} items.")
    print(f"Persisted to '{CHROMA_DIR}/'.")


if __name__ == "__main__":
    main()
