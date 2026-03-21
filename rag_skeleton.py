"""
rag.py — Main RAG pipeline for CS288 Assignment 3.

Called by run.sh:
    python3 rag.py <questions_txt_path> <predictions_out_path>

Reads questions (one per line), retrieves relevant chunks, generates
answers via OpenRouter LLM, and writes predictions (one per line).

TODO: Implement retrieval and generation components.
"""

import sys
import os
import json
import time

# Ensure the script's directory is on the path so llm.py is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from llm import call_llm

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.jsonl")


def load_chunks():
    """Load the retrieval corpus chunks."""
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def retrieve(question, chunks, top_k=5):
    """Retrieve the top-k most relevant chunks for a question.

    TODO: Replace this placeholder with BM25, dense retrieval, or hybrid.
    """
    # Placeholder: return first top_k chunks
    return chunks[:top_k]


def generate_answer(question, context_chunks):
    """Generate an answer using the LLM with retrieved context.

    TODO: Tune the prompt, model choice, max_tokens, temperature.
    """
    context = "\n\n".join(c["text"] for c in context_chunks)

    prompt = (
        f"Answer the following question using ONLY the provided context. "
        f"Give a short, concise answer (under 10 words). "
        f"If the answer is not in the context, respond with 'unknown'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    try:
        answer = call_llm(query=prompt, max_tokens=64, temperature=0.0)
        # Clean up: remove quotes, extra whitespace, newlines
        answer = answer.strip().strip('"').strip("'").split("\n")[0].strip()
        return answer
    except Exception as e:
        print(f"LLM error for '{question[:50]}...': {e}", file=sys.stderr)
        return "unknown"


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 rag.py <questions_path> <predictions_path>",
              file=sys.stderr)
        sys.exit(1)

    questions_path = sys.argv[1]
    predictions_path = sys.argv[2]

    # Load questions
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(questions)} questions", file=sys.stderr)

    # Load retrieval corpus
    print("Loading chunks...", file=sys.stderr)
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks", file=sys.stderr)

    # Process each question
    predictions = []
    start = time.time()

    for i, question in enumerate(questions):
        retrieved = retrieve(question, chunks)
        answer = generate_answer(question, retrieved)
        predictions.append(answer)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(questions)}] {elapsed:.1f}s elapsed",
                  file=sys.stderr)

    # Write predictions
    with open(predictions_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            # Ensure no newlines within a prediction
            f.write(pred.replace("\n", " ").strip() + "\n")

    elapsed = time.time() - start
    print(f"Done: {len(predictions)} predictions in {elapsed:.1f}s "
          f"({elapsed/len(predictions):.2f}s/question)", file=sys.stderr)


if __name__ == "__main__":
    main()
