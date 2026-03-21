"""
llm_clean.py — LLM-powered corpus cleaning using Gemini.

Reads BS4-cleaned documents from data/corpus.jsonl, sends each through
Gemini to remove navigation artifacts, repeated menus, formatting noise,
and restructure for RAG consumption. Outputs data/corpus_llm.jsonl.

This is an OFFLINE step (no runtime cost). The spec explicitly awards
extra credit for using a language model to rewrite documents.

Usage:
    pip install google-genai
    export GEMINI_API_KEY="your-key-here"
    python offline/llm_clean.py [--workers 10] [--model gemini-2.5-flash-lite]

Output:
    data/corpus_llm.jsonl  — LLM-cleaned documents
"""

import os
import sys
import json
import time
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from google import genai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CORPUS_IN = os.path.join(DATA_DIR, "corpus.jsonl")
CORPUS_OUT = os.path.join(DATA_DIR, "corpus_llm.jsonl")
PROGRESS_FILE = CORPUS_OUT + ".progress"

API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    sys.exit("ERROR: Set GEMINI_API_KEY environment variable before running.")

client = genai.Client(api_key=API_KEY)

CLEANING_PROMPT = """You are a text cleaning assistant for a university knowledge base.

Clean up the following extracted web text from the UC Berkeley EECS website. Your job:

1. REMOVE navigation artifacts, repeated menu items, breadcrumb trails, sidebar text, and footer boilerplate that slipped through HTML parsing.
2. REMOVE duplicate sentences or paragraphs (common from overlapping page sections).
3. FIX broken sentences that were split across HTML elements.
4. ORGANIZE the content with clear section headings (use ## for headings).
5. CONVERT garbled table remnants (pipe-separated fragments) into clean readable lists or structured text.
6. PRESERVE every single factual detail: names, numbers, dates, emails, phone numbers, URLs, course numbers, room numbers, award names, etc. Do NOT omit, paraphrase, or round any factual content.
7. Do NOT add any information that is not in the original text.
8. Do NOT add commentary, explanations, or meta-text like "This page contains...".
9. Output clean plain text with ## headings. No markdown formatting beyond headings.

PAGE TITLE: {title}
SOURCE URL: {url}

EXTRACTED TEXT:
{text}

CLEANED TEXT:"""

MAX_INPUT_CHARS = 8000
MIN_TEXT_LENGTH = 50

# ---------------------------------------------------------------------------
# Progress tracking (thread-safe)
# ---------------------------------------------------------------------------
progress_lock = Lock()


def load_progress():
    """Load already-cleaned URLs from progress file."""
    done = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    done[obj["url"]] = obj
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def save_progress_entry(entry):
    """Append a single cleaned entry to the progress file (thread-safe)."""
    with progress_lock:
        with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Gemini API call with retry
# ---------------------------------------------------------------------------
def _parse_retry_delay(error_str):
    m = re.search(r"retry in ([\d.]+)s", str(error_str), re.IGNORECASE)
    if m:
        return min(float(m.group(1)) + 2, 120)
    m = re.search(r"retryDelay.*?(\d+)s", str(error_str))
    if m:
        return min(int(m.group(1)) + 2, 120)
    return None


def call_gemini(prompt, model_name, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name, contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            is_quota = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

            if is_quota:
                suggested = _parse_retry_delay(err_str)
                wait = suggested if suggested else (15 * (attempt + 1))
                time.sleep(wait)
            else:
                wait = 3 * (attempt + 1)
                if attempt >= max_retries - 1:
                    return None
                time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# Clean a single document
# ---------------------------------------------------------------------------
def clean_document(doc, model_name):
    """Send a single document through Gemini for cleaning."""
    url = doc["url"]
    title = doc.get("title", "")
    text = doc.get("text", "")

    if len(text) < MIN_TEXT_LENGTH:
        return {
            "url": url,
            "title": title,
            "text": text,
            "llm_cleaned": False,
        }

    input_text = text[:MAX_INPUT_CHARS]

    prompt = CLEANING_PROMPT.format(
        title=title,
        url=url,
        text=input_text,
    )

    cleaned = call_gemini(prompt, model_name)

    if cleaned and len(cleaned) > 30:
        return {
            "url": url,
            "title": title,
            "text": cleaned,
            "text_original": text,
            "llm_cleaned": True,
        }
    else:
        return {
            "url": url,
            "title": title,
            "text": text,
            "llm_cleaned": False,
        }


# ---------------------------------------------------------------------------
# Worker function for thread pool
# ---------------------------------------------------------------------------
counter_lock = Lock()
stats = {"done": 0, "cleaned": 0, "skipped": 0, "failed": 0, "total": 0}


def process_one(doc, model_name, per_worker_delay):
    """Process a single document (called from thread pool)."""
    result = clean_document(doc, model_name)
    save_progress_entry(result)

    with counter_lock:
        stats["done"] += 1
        if result.get("llm_cleaned"):
            stats["cleaned"] += 1
        else:
            stats["failed"] += 1

        if stats["done"] % 100 == 0 or stats["done"] <= 5:
            print(
                f"  [{stats['done']}/{stats['total']}] "
                f"cleaned={stats['cleaned']} skipped={stats['skipped']} "
                f"failed={stats['failed']} | last: {doc['url'][:70]}"
            )

    time.sleep(per_worker_delay)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LLM-clean corpus via Gemini")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of concurrent workers (default 10)")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-lite",
                        help="Gemini model to use")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Per-worker delay between calls in seconds")
    args = parser.parse_args()

    if not os.path.exists(CORPUS_IN):
        print(f"ERROR: {CORPUS_IN} not found. Run clean_corpus.py first.")
        return

    # Load input corpus
    docs = []
    with open(CORPUS_IN, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(docs)} documents from {CORPUS_IN}")

    # Load progress
    done_map = load_progress()
    if done_map:
        print(f"Resuming: {len(done_map)} documents already cleaned")

    # Filter out already-done docs
    todo = [d for d in docs if d["url"] not in done_map]
    already_done = [done_map[d["url"]] for d in docs if d["url"] in done_map]

    stats["total"] = len(todo)
    stats["skipped"] = len(already_done)

    if not todo:
        print("All documents already cleaned!")
    else:
        print(f"Processing {len(todo)} documents with {args.workers} workers "
              f"using {args.model}...")
        print(f"Effective rate: ~{args.workers / args.delay:.0f} RPM "
              f"(limit: 4000 RPM)")

        start = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_one, doc, args.model, args.delay
                ): doc
                for doc in todo
            }

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    doc = futures[future]
                    print(f"  ERROR processing {doc['url'][:60]}: {e}")

        elapsed = time.time() - start
        print(f"\nProcessing done: {elapsed/60:.1f} minutes, "
              f"{len(todo)/elapsed:.1f} docs/sec")

    # Assemble final output: merge progress with order from original corpus
    print("Assembling final corpus_llm.jsonl...")
    all_done = load_progress()

    written = 0
    with open(CORPUS_OUT, "w", encoding="utf-8") as f:
        for doc in docs:
            url = doc["url"]
            if url in all_done:
                entry = all_done[url]
                out = {
                    "url": entry["url"],
                    "title": entry.get("title", ""),
                    "text": entry["text"],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                written += 1
            else:
                out = {
                    "url": doc["url"],
                    "title": doc.get("title", ""),
                    "text": doc.get("text", ""),
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                written += 1

    size_mb = os.path.getsize(CORPUS_OUT) / (1024 * 1024)
    print(f"\nDone! Wrote {written} documents to {CORPUS_OUT} ({size_mb:.1f} MB)")

    llm_count = sum(1 for v in all_done.values() if v.get("llm_cleaned"))
    print(f"  LLM-cleaned: {llm_count}")
    print(f"  Passed through (too short or API failure): {written - llm_count}")


if __name__ == "__main__":
    main()
