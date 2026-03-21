"""Build BM25 index from data/chunks.jsonl.

Usage:
    python3 build_index.py
"""

import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def main():
    corpus_path = Path("data/chunks.jsonl")
    out_path = Path("data/bm25_index.pkl")
    if not corpus_path.exists():
        raise FileNotFoundError("data/chunks.jsonl not found.")

    docs = []
    tokenized = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text", "")
            if not text.strip():
                continue
            docs.append(row)
            tokenized.append(tokenize(text))

    if not docs:
        raise RuntimeError("No corpus docs found in data/corpus.jsonl")

    bm25 = BM25Okapi(tokenized)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "docs": docs}, f)

    print(f"Indexed {len(docs)} chunks -> {out_path}")


if __name__ == "__main__":
    main()
