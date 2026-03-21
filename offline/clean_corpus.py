"""
clean_corpus.py — Convert raw crawled HTML into a clean, chunked retrieval corpus.

Two modes:
  1. Default: Reads raw HTML from crawl, BS4-cleans, outputs corpus.jsonl + chunks.jsonl
  2. --chunk-only --input <file>: Reads a pre-cleaned corpus JSONL and only chunks it

Usage:
    # Full pipeline: raw HTML -> corpus.jsonl + chunks.jsonl
    python offline/clean_corpus.py

    # Chunk-only from LLM-cleaned corpus:
    python offline/clean_corpus.py --chunk-only --input data/corpus_llm.jsonl
"""

import os
import re
import json
import argparse
import unicodedata
from bs4 import BeautifulSoup, Comment

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw_html")
MANIFEST = os.path.join(DATA_DIR, "crawl_manifest.jsonl")
CORPUS_OUT = os.path.join(DATA_DIR, "corpus.jsonl")
CHUNKS_OUT = os.path.join(DATA_DIR, "chunks.jsonl")

BOILERPLATE_TAGS = {"nav", "footer", "header", "script", "style", "noscript",
                    "aside", "iframe", "svg", "form"}


# ---------------------------------------------------------------------------
# HTML -> Clean text
# ---------------------------------------------------------------------------
def extract_title(soup):
    """Get page title from <title> or first <h1>."""
    title_tag = soup.find("title")
    if title_tag:
        text = title_tag.get_text(strip=True)
        # Remove common suffixes like "| EECS at UC Berkeley"
        text = re.sub(r"\s*\|.*$", "", text).strip()
        if text:
            return text
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return ""


def clean_html(html, url=""):
    """Convert raw HTML to structured plain text with heading markers."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove boilerplate tags
    for tag in soup.find_all(BOILERPLATE_TAGS):
        tag.decompose()
    # Remove HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    title = extract_title(soup)

    sections = []
    current_heading = ""
    current_lines = []

    def flush_section():
        nonlocal current_heading, current_lines
        text = "\n".join(current_lines).strip()
        if text and len(text) > 20:
            sections.append({
                "heading": current_heading,
                "text": text,
            })
        current_lines = []

    # Walk through content-bearing tags in document order
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6",
                               "p", "li", "td", "th", "tr",
                               "dd", "dt", "blockquote", "pre"]):
        tag_name = tag.name

        if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            flush_section()
            current_heading = tag.get_text(" ", strip=True)
            continue

        if tag_name == "tr":
            cells = [td.get_text(" ", strip=True) for td in tag.find_all(["td", "th"])]
            row = " | ".join(c for c in cells if c)
            if len(row) > 10:
                current_lines.append(row)
            continue

        if tag_name in ("td", "th"):
            # Already handled by tr
            continue

        text = tag.get_text(" ", strip=True)
        text = normalize_text(text)
        if len(text) > 15:
            current_lines.append(text)

    flush_section()

    full_text = ""
    for sec in sections:
        if sec["heading"]:
            full_text += f"\n## {sec['heading']}\n"
        full_text += sec["text"] + "\n"

    return {
        "title": title,
        "text": full_text.strip(),
        "sections": sections,
    }


def normalize_text(text):
    """Collapse whitespace and normalize unicode."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_document(doc, url, chunk_size=400, chunk_overlap=50):
    """Split a cleaned document into overlapping chunks.

    Strategy:
    1. If the document has heading-based sections, each section is a chunk
       (split further if too long).
    2. Fallback: sliding window on the full text.

    Each chunk includes the page title as prefix for retrieval context.
    """
    chunks = []
    title = doc["title"]
    sections = doc["sections"]

    if sections and len(sections) > 1:
        for sec in sections:
            heading = sec["heading"]
            text = sec["text"]
            prefix = f"{title}"
            if heading:
                prefix += f" > {heading}"

            words = text.split()
            if len(words) <= chunk_size:
                chunk_text = f"{prefix}\n{text}"
                chunks.append({
                    "url": url,
                    "title": title,
                    "section": heading,
                    "text": chunk_text,
                })
            else:
                # Split long sections with overlap
                sub_chunks = sliding_window_chunks(words, chunk_size, chunk_overlap)
                for i, sub in enumerate(sub_chunks):
                    chunk_text = f"{prefix} (part {i+1})\n{sub}"
                    chunks.append({
                        "url": url,
                        "title": title,
                        "section": f"{heading} (part {i+1})" if heading else f"part {i+1}",
                        "text": chunk_text,
                    })
    else:
        # No sections -- use sliding window on full text
        full_text = doc["text"]
        words = full_text.split()
        if len(words) <= chunk_size:
            chunks.append({
                "url": url,
                "title": title,
                "section": "",
                "text": f"{title}\n{full_text}",
            })
        else:
            sub_chunks = sliding_window_chunks(words, chunk_size, chunk_overlap)
            for i, sub in enumerate(sub_chunks):
                chunks.append({
                    "url": url,
                    "title": title,
                    "section": f"part {i+1}",
                    "text": f"{title} (part {i+1})\n{sub}",
                })

    return chunks


def sliding_window_chunks(words, size, overlap):
    """Split a word list into overlapping windows."""
    chunks = []
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start:start + size]
        if len(chunk_words) < 30:
            break
        chunks.append(" ".join(chunk_words))
    return chunks


# ---------------------------------------------------------------------------
# Chunk-only mode: read a pre-cleaned corpus JSONL and chunk it
# ---------------------------------------------------------------------------
def chunk_from_corpus(input_path, chunks_out, chunk_size=400, chunk_overlap=50):
    """Read a corpus JSONL (url, title, text) and produce chunks.jsonl."""
    docs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"Chunking {len(docs)} documents from {input_path}...")

    total_chunks = 0
    with open(chunks_out, "w", encoding="utf-8") as f:
        for i, doc in enumerate(docs):
            url = doc["url"]
            title = doc.get("title", "")
            text = doc.get("text", "")

            if len(text) < 50:
                continue

            # Build a sections list from heading markers in the text
            sections = []
            current_heading = ""
            current_lines = []

            for raw_line in text.split("\n"):
                line = raw_line.strip()
                if line.startswith("## "):
                    if current_lines:
                        sections.append({
                            "heading": current_heading,
                            "text": "\n".join(current_lines).strip(),
                        })
                        current_lines = []
                    current_heading = line[3:].strip()
                elif line:
                    current_lines.append(line)

            if current_lines:
                sections.append({
                    "heading": current_heading,
                    "text": "\n".join(current_lines).strip(),
                })

            fake_doc = {"title": title, "text": text, "sections": sections}
            doc_chunks = chunk_document(fake_doc, url, chunk_size, chunk_overlap)
            for chunk in doc_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            total_chunks += len(doc_chunks)

            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(docs)}] {total_chunks} chunks so far...")

    size_mb = os.path.getsize(chunks_out) / (1024 * 1024)
    print(f"\nDone! {total_chunks} chunks -> {chunks_out} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Full pipeline: raw HTML -> corpus.jsonl + chunks.jsonl
# ---------------------------------------------------------------------------
def main_full(chunk_size=400, chunk_overlap=50):
    if not os.path.exists(MANIFEST):
        print(f"ERROR: Manifest not found at {MANIFEST}")
        print("Run crawl.py first.")
        return

    entries = []
    with open(MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"Processing {len(entries)} crawled pages...")

    total_chunks = 0
    skipped = 0

    with open(CORPUS_OUT, "w", encoding="utf-8") as corpus_f, \
         open(CHUNKS_OUT, "w", encoding="utf-8") as chunks_f:

        for i, entry in enumerate(entries):
            url = entry["url"]
            fname = entry["filename"]
            fpath = os.path.join(RAW_DIR, fname)

            if not os.path.exists(fpath):
                skipped += 1
                continue

            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                html = f.read()

            doc = clean_html(html, url)

            if len(doc["text"]) < 50:
                skipped += 1
                continue

            corpus_f.write(json.dumps({
                "url": url,
                "title": doc["title"],
                "text": doc["text"],
            }, ensure_ascii=False) + "\n")

            doc_chunks = chunk_document(doc, url, chunk_size, chunk_overlap)
            for chunk in doc_chunks:
                chunks_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            total_chunks += len(doc_chunks)

            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(entries)}] {total_chunks} chunks so far...")

    print(f"\nDone!")
    print(f"  Pages processed: {len(entries) - skipped}")
    print(f"  Pages skipped (empty/missing): {skipped}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Corpus: {CORPUS_OUT}")
    print(f"  Chunks: {CHUNKS_OUT}")

    for path in (CORPUS_OUT, CHUNKS_OUT):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {os.path.basename(path)}: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and chunk crawled HTML")
    parser.add_argument("--chunk-size", type=int, default=400,
                        help="Max words per chunk (default 400)")
    parser.add_argument("--chunk-overlap", type=int, default=50,
                        help="Word overlap between chunks (default 50)")
    parser.add_argument("--chunk-only", action="store_true",
                        help="Skip HTML cleaning; only chunk a pre-cleaned corpus")
    parser.add_argument("--input", type=str, default=None,
                        help="Input corpus JSONL for --chunk-only mode")
    parser.add_argument("--output-chunks", type=str, default=None,
                        help="Output chunks file (default: data/chunks.jsonl)")
    args = parser.parse_args()

    if args.chunk_only:
        input_path = args.input or os.path.join(DATA_DIR, "corpus_llm.jsonl")
        output_path = args.output_chunks or CHUNKS_OUT
        chunk_from_corpus(input_path, output_path,
                          args.chunk_size, args.chunk_overlap)
    else:
        main_full(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
