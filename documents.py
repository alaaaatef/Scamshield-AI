"""
01_documents.py
---------------
STAGE 1 of the RAG pipeline: Document Loading.

Reads every PDF in ./data and extracts raw, page-level text using pypdf.
No cleaning happens here - that's handled in 02_preprocessing.py.

Run standalone:
    python 01_documents.py

Output:
    pipeline_data/01_raw_documents.pkl
    -> a list of dicts: {"source": <filename>, "page": <page number>, "text": <raw extracted text>}
"""

import os
import pickle
from pypdf import PdfReader

DATA_DIR = "data"
OUTPUT_DIR = "pipeline_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "01_raw_documents.pkl")


def load_documents(data_dir=DATA_DIR):
    """Extract raw per-page text from every PDF in data_dir."""
    if not os.path.isdir(data_dir):
        raise SystemExit(f"Folder '{data_dir}/' not found. Create it and add your PDFs.")

    pdf_files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(".pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDFs found in '{data_dir}/'.")

    documents = []
    for fname in pdf_files:
        path = os.path.join(data_dir, fname)
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            if raw_text.strip():
                documents.append({
                    "source": fname,
                    "page": i + 1,
                    "text": raw_text,
                })
    return documents


def main():
    print(f"Loading PDFs from '{DATA_DIR}/'...")
    documents = load_documents()
    print(f"Extracted {len(documents)} pages from "
          f"{len(set(d['source'] for d in documents))} PDF(s).")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(documents, f)
    print(f"Saved raw documents to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
