"""
02_preprocessing.py
--------------------
STAGE 2 of the RAG pipeline: Text Cleaning.

Loads the raw per-page text from Stage 1 and cleans it:
    - Normalizes whitespace (collapses newlines/tabs/extra spaces)
    - Removes standalone page-number lines (e.g. "12", "Page 12", "12/24")
    - Removes repeated headers/footers (lines that appear on many pages of the
      SAME document - e.g. "TLP:CLEAR" or "CISA | NSA | FBI | MS-ISAC")

Run standalone:
    python 02_preprocessing.py

Input:
    pipeline_data/01_raw_documents.pkl
Output:
    pipeline_data/02_cleaned_documents.pkl
    -> same structure as input, but with cleaned "text"
"""

import os
import re
import pickle
from collections import Counter

INPUT_FILE = "pipeline_data/01_raw_documents.pkl"
OUTPUT_FILE = "pipeline_data/02_cleaned_documents.pkl"

# A line that is just a number, or "Page 12", or "12/24" is almost always a page number
PAGE_NUMBER_PATTERN = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*/\s*\d{1,4})?\s*$", re.IGNORECASE)


def normalize_whitespace(text):
    return " ".join(text.split())


def find_repeated_lines(pages_of_one_doc, min_page_fraction=0.4):
    """
    Identify lines that repeat across many pages of the same document -
    these are almost always running headers/footers, not real content.
    """
    line_counter = Counter()
    for text in pages_of_one_doc:
        # look at raw lines (before whitespace normalization) so headers match exactly
        lines = {ln.strip() for ln in text.splitlines() if ln.strip()}
        line_counter.update(lines)

    n_pages = len(pages_of_one_doc)
    threshold = max(2, int(n_pages * min_page_fraction))
    repeated = {line for line, count in line_counter.items() if count >= threshold}
    return repeated


def clean_page_text(raw_text, repeated_lines):
    """Remove repeated header/footer lines and standalone page-number lines, then
    normalize whitespace."""
    kept_lines = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in repeated_lines:
            continue
        if PAGE_NUMBER_PATTERN.match(stripped):
            continue
        kept_lines.append(stripped)
    return normalize_whitespace(" ".join(kept_lines))


def clean_documents(documents):
    # Group raw page texts by source document so we can spot repeated headers/footers
    pages_by_source = {}
    for doc in documents:
        pages_by_source.setdefault(doc["source"], []).append(doc["text"])

    repeated_lines_by_source = {
        source: find_repeated_lines(pages) for source, pages in pages_by_source.items()
    }

    cleaned = []
    for doc in documents:
        repeated_lines = repeated_lines_by_source[doc["source"]]
        cleaned_text = clean_page_text(doc["text"], repeated_lines)
        if cleaned_text:
            cleaned.append({
                "source": doc["source"],
                "page": doc["page"],
                "text": cleaned_text,
            })
    return cleaned


def main():
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"'{INPUT_FILE}' not found. Run 01_documents.py first.")

    with open(INPUT_FILE, "rb") as f:
        documents = pickle.load(f)

    print(f"Cleaning {len(documents)} pages...")
    cleaned = clean_documents(documents)
    print(f"Kept {len(cleaned)} non-empty pages after cleaning.")

    os.makedirs("pipeline_data", exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(cleaned, f)
    print(f"Saved cleaned documents to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
