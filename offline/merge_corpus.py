"""
merge_corpus.py — Merge our LLM-cleaned corpus with the staff reference corpus,
fetch any missing URLs from hidden_dev, and deduplicate.

Usage:
    python offline/merge_corpus.py

Output:
    data/corpus_merged.jsonl  — combined, deduplicated corpus
"""

import os
import sys
import json
import urllib.request
import urllib.error
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Comment

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUR_CORPUS = os.path.join(DATA_DIR, "corpus.jsonl")
REF_CORPUS = os.path.join(os.path.dirname(__file__), "..", "eecs_text_bs_rewritten.jsonl")
HIDDEN_DEV = os.path.join(os.path.dirname(__file__), "..", "hidden_dev.jsonl")
OUTPUT = os.path.join(DATA_DIR, "corpus_merged.jsonl")

BOILERPLATE_TAGS = {"nav", "footer", "header", "script", "style", "noscript",
                    "aside", "iframe", "svg", "form"}


def normalize_url(url):
    """Normalize for dedup: strip trailing slash, fragment, lowercase host."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
    query = parsed.query
    return f"{scheme}://{netloc}{path}" + (f"?{query}" if query else "")


def fetch_url(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def quick_clean(html):
    """Fast BS4 clean for a single page."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(BOILERPLATE_TAGS):
        tag.decompose()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    lines = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6",
                               "p", "li", "dd", "dt", "blockquote", "pre"]):
        text = tag.get_text(" ", strip=True)
        if len(text) > 15:
            if tag.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                lines.append(f"\n## {text}")
            else:
                lines.append(text)

    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            row = " | ".join(c for c in cells if c)
            if len(row) > 10:
                lines.append(row)

    return "\n".join(lines).strip()


def main():
    corpus = {}  # url -> {url, text}

    # 1. Load our corpus (higher priority — LLM-cleaned)
    if os.path.exists(OUR_CORPUS):
        with open(OUR_CORPUS, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                url = obj["url"]
                corpus[normalize_url(url)] = {"url": url, "text": obj.get("text", "")}
        print(f"Loaded {len(corpus)} docs from our corpus")

    # 2. Load reference corpus (fill gaps only)
    ref_added = 0
    if os.path.exists(REF_CORPUS):
        with open(REF_CORPUS, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                url = obj["url"]
                norm = normalize_url(url)
                if norm not in corpus:
                    corpus[norm] = {"url": url, "text": obj.get("text", "")}
                    ref_added += 1
        print(f"Added {ref_added} new docs from reference corpus")

    # 3. Fetch missing URLs from hidden_dev
    if os.path.exists(HIDDEN_DEV):
        dev_urls = set()
        with open(HIDDEN_DEV, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                dev_urls.add(obj["url"])

        missing = []
        for url in dev_urls:
            norm = normalize_url(url)
            if norm not in corpus:
                # Also check without query params
                base_norm = normalize_url(url.split("?")[0])
                if base_norm not in corpus:
                    missing.append(url)

        if missing:
            print(f"\nFetching {len(missing)} missing dev URLs...")
            for url in missing:
                html = fetch_url(url)
                if html:
                    text = quick_clean(html)
                    if len(text) > 50:
                        corpus[normalize_url(url)] = {"url": url, "text": text}
                        print(f"  + {url[:80]}")
                    else:
                        print(f"  SKIP (too short): {url[:80]}")
                else:
                    print(f"  FAIL: {url[:80]}")
        else:
            print("All dev URLs already covered!")

    # 4. Write merged corpus
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for entry in corpus.values():
            if len(entry.get("text", "")) > 30:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"\nMerged corpus: {len(corpus)} docs -> {OUTPUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
