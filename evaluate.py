"""Evaluate predictions against validation_data.json.

Computes Exact Match, Token F1, and Retrieval Recall@5 (at rank k=5).

Usage:
    python3 evaluate.py <predictions_txt_path>
"""

import json
import pickle
import re
import string
import sys
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def normalize(text):
    text = (text or "").lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def token_f1(prediction, reference):
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    n_overlap = sum(overlap.values())
    if n_overlap == 0:
        return 0.0
    precision = n_overlap / len(pred_tokens)
    recall = n_overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction, reference):
    return float(normalize(prediction) == normalize(reference))


def retrieve_top_k(question, bm25, docs, k=5):
    tokens = TOKEN_RE.findall((question or "").lower())
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in ranked[:k]]


def answer_in_passages(passages, correct_answers):
    all_text = normalize(" ".join(p.get("text", "") for p in passages))
    return any(normalize(a) in all_text for a in correct_answers if normalize(a))


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 evaluate.py <predictions_txt_path>", file=sys.stderr)
        sys.exit(1)

    pred_path = Path(sys.argv[1])
    val_path = Path("validation_data.json")
    index_path = Path("data/bm25_index.pkl")

    predictions = pred_path.read_text(encoding="utf-8").splitlines()
    val_rows = [json.loads(line) for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    n = min(len(predictions), len(val_rows))
    if n == 0:
        print("No predictions or validation rows found.", file=sys.stderr)
        sys.exit(1)

    bm25, docs = None, None
    if index_path.exists():
        with index_path.open("rb") as f:
            bundle = pickle.load(f)
        bm25, docs = bundle["bm25"], bundle["docs"]

    em_total, f1_total, recall_hits = 0.0, 0.0, 0

    for i in range(n):
        correct_answers = [a.strip() for a in val_rows[i]["answer"].split("|")]
        em_total += max(exact_match(predictions[i], a) for a in correct_answers)
        f1_total += max(token_f1(predictions[i], a) for a in correct_answers)

        if bm25 is not None:
            passages = retrieve_top_k(val_rows[i]["question"], bm25, docs)
            if answer_in_passages(passages, correct_answers):
                recall_hits += 1

    print(f"Compared {n} examples")
    print(f"Exact Match: {em_total / n:.4f}")
    print(f"Token F1:    {f1_total / n:.4f}")
    if bm25 is not None:
        print(f"Recall@5:    {recall_hits / n:.4f}")


if __name__ == "__main__":
    main()
