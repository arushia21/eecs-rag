"""BM25 + LLM RAG pipeline for EECS QA.

Usage:
    python3 rag.py <questions_txt_path> <predictions_out_path>
"""

import os
import pickle
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(SCRIPT_DIR))

from llm import call_llm

INDEX_PATH = SCRIPT_DIR / "data" / "bm25_index.pkl"
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a factoid QA assistant for UC Berkeley EECS. "
    "Answer using ONLY the provided context. "
    "Give a short, precise answer — under 10 words. "
    "Do not explain or add extra words. "
    "If the answer is not in the context, say: unknown"
)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load_index():
    with INDEX_PATH.open("rb") as f:
        bundle = pickle.load(f)
    return bundle["bm25"], bundle["docs"]


def retrieve(question, bm25, docs, k=TOP_K):
    tokens = tokenize(question)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in ranked[:k]]


def make_prompt(question, passages):
    context_parts = []
    for i, p in enumerate(passages, 1):
        url = p.get("url", "")
        text = p.get("text", "")
        context_parts.append(f"[{i}] {url}\n{text}")
    context = "\n\n".join(context_parts)
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def clean_answer(raw):
    if not raw:
        return "unknown"
    answer = raw.replace("\n", " ").strip()
    for prefix in ["Answer:", "The answer is", "Based on the context,"]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip().lstrip(":").strip()
    answer = answer.strip('"').strip("'").strip()
    return answer if answer else "unknown"


def answer_question(question, bm25, docs):
    passages = retrieve(question, bm25, docs)
    prompt = make_prompt(question, passages)
    try:
        raw = call_llm(
            query=prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=64,
            temperature=0.0,
            timeout=25,
        )
        return clean_answer(raw)
    except Exception:
        return "unknown"


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 rag.py <questions_path> <predictions_path>", file=sys.stderr)
        sys.exit(1)

    questions_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    print("Loading index...", file=sys.stderr)
    bm25, docs = load_index()
    print(f"Loaded {len(docs)} chunks", file=sys.stderr)

    questions = [line.strip() for line in questions_path.read_text(encoding="utf-8").splitlines()]
    predictions = []
    start = time.time()

    for i, q in enumerate(questions):
        if not q:
            predictions.append("unknown")
            continue
        predictions.append(answer_question(q, bm25, docs))
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(questions)}] {elapsed:.1f}s", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(predictions) + "\n", encoding="utf-8")

    elapsed = time.time() - start
    print(f"Done: {len(predictions)} predictions in {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
