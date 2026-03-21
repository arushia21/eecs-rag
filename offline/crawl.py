"""
crawl.py — BFS crawler for eecs.berkeley.edu

Uses the staff-provided fetch function. Saves raw HTML to data/raw_html/
and maintains a manifest for resume support.

Usage:
    python offline/crawl.py [--max-pages 5000] [--delay 0.4]

Output:
    data/raw_html/*.html        — one file per crawled page
    data/crawl_manifest.jsonl   — {url, filename, timestamp} per page
"""

import urllib.request
import urllib.error
from urllib.parse import urljoin, urlparse, urlunparse
from html.parser import HTMLParser
import os
import re
import json
import time
import hashlib
import argparse
from collections import deque

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENTRY_POINTS = [
    "https://eecs.berkeley.edu/",
    "https://www2.eecs.berkeley.edu/",
    "https://eecs.berkeley.edu/academics/",
    "https://eecs.berkeley.edu/people/",
    "https://eecs.berkeley.edu/research/",
    "https://eecs.berkeley.edu/about/",
    "https://www2.eecs.berkeley.edu/Faculty/Lists/faculty.html",
    "https://www2.eecs.berkeley.edu/Courses/CS/",
    "https://www2.eecs.berkeley.edu/Courses/EE/",
    "https://www2.eecs.berkeley.edu/Research/Areas/",
    "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/",
    "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/",
    "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2023/",
    "https://www2.eecs.berkeley.edu/Students/Awards/",
    "https://eecs.berkeley.edu/book/phd/",
    "https://eecs.berkeley.edu/academics/undergraduate/",
    "https://eecs.berkeley.edu/academics/graduate/",
]

EECS_PATTERN = re.compile(
    r"^https?://(?:www\d*\.)?eecs\.berkeley\.edu(?:/[^\s]*)?$"
)

SKIP_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".mp4", ".mp3",
    ".zip", ".tar", ".gz", ".bz2", ".doc", ".docx", ".ppt", ".pptx",
    ".xls", ".xlsx", ".css", ".js", ".ico", ".woff", ".woff2", ".ttf",
    ".eot",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw_html")
MANIFEST = os.path.join(DATA_DIR, "crawl_manifest.jsonl")


# ---------------------------------------------------------------------------
# Staff-provided fetch function (from Ed discussion)
# ---------------------------------------------------------------------------
def fetch_url(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode("utf-8", errors="replace")
        return html
    except Exception:
        return None


# ---------------------------------------------------------------------------
# URL normalization and validation
# ---------------------------------------------------------------------------
def normalize_url(url):
    """Normalize URL for dedup: strip fragment, trailing slash, lowercase host."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
    return urlunparse((scheme, netloc, path, "", parsed.query, ""))


def is_valid_url(url):
    """Check URL matches eecs.berkeley.edu and isn't a binary file."""
    if not EECS_PATTERN.match(url):
        return False
    path = urlparse(url).path.lower()
    ext = os.path.splitext(path)[1]
    if ext in SKIP_EXTENSIONS:
        return False
    if "login" in path or "cas.berkeley" in url:
        return False
    return True


# ---------------------------------------------------------------------------
# Simple link extractor (no external deps needed)
# ---------------------------------------------------------------------------
class LinkExtractor(HTMLParser):
    def __init__(self, base_url):
        super().__init__()
        self.base_url = base_url
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    full = urljoin(self.base_url, value).split("#")[0]
                    self.links.append(full)

    def error(self, message):
        pass


def extract_links(html, base_url):
    """Extract all <a href> links from HTML."""
    parser = LinkExtractor(base_url)
    try:
        parser.feed(html)
    except Exception:
        pass
    return parser.links


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def load_manifest():
    """Load already-crawled URLs from manifest."""
    visited = set()
    if os.path.exists(MANIFEST):
        with open(MANIFEST, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    visited.add(obj["url"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return visited


def url_to_filename(url):
    """Deterministic filename from URL hash."""
    h = hashlib.md5(url.encode()).hexdigest()[:16]
    return f"{h}.html"


# ---------------------------------------------------------------------------
# Main crawler
# ---------------------------------------------------------------------------
def crawl(max_pages=5000, delay=0.4):
    os.makedirs(RAW_DIR, exist_ok=True)

    visited = load_manifest()
    if visited:
        print(f"Resuming: {len(visited)} pages already crawled")

    queue = deque()
    seen = set(visited)

    for entry in ENTRY_POINTS:
        norm = normalize_url(entry)
        if norm not in seen:
            queue.append(norm)
            seen.add(norm)

    crawled = len(visited)
    failed = 0
    start_time = time.time()

    manifest_f = open(MANIFEST, "a", encoding="utf-8")

    try:
        while queue and crawled < max_pages:
            url = queue.popleft()

            if url in visited:
                continue

            html = fetch_url(url)
            if html is None:
                failed += 1
                visited.add(url)
                continue

            fname = url_to_filename(url)
            fpath = os.path.join(RAW_DIR, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(html)

            entry = {"url": url, "filename": fname, "timestamp": time.time()}
            manifest_f.write(json.dumps(entry) + "\n")
            manifest_f.flush()
            visited.add(url)
            crawled += 1

            links = extract_links(html, url)
            new_links = 0
            for link in links:
                norm = normalize_url(link)
                if norm not in seen and is_valid_url(norm):
                    seen.add(norm)
                    queue.append(norm)
                    new_links += 1

            elapsed = time.time() - start_time
            rate = crawled / elapsed if elapsed > 0 else 0
            if crawled % 50 == 0 or crawled < 10:
                print(
                    f"[{crawled}/{max_pages}] {url[:80]}... "
                    f"(+{new_links} links, queue={len(queue)}, "
                    f"failed={failed}, {rate:.1f} pg/s)"
                )

            time.sleep(delay)

    except KeyboardInterrupt:
        print("\nInterrupted! Progress saved to manifest.")
    finally:
        manifest_f.close()

    elapsed = time.time() - start_time
    print(f"\nDone: {crawled} pages crawled, {failed} failed, "
          f"{elapsed/60:.1f} minutes elapsed")
    print(f"Queue remaining: {len(queue)} URLs")
    print(f"Manifest: {MANIFEST}")
    print(f"Raw HTML: {RAW_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl eecs.berkeley.edu")
    parser.add_argument("--max-pages", type=int, default=5000)
    parser.add_argument("--delay", type=float, default=0.4,
                        help="Seconds between requests (default 0.4)")
    args = parser.parse_args()
    crawl(max_pages=args.max_pages, delay=args.delay)
