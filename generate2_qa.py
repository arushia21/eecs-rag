"""
generate2_qa.py — Thorough EECS QA dataset generator for CS288 Assignment 3.

Crawls diverse eecs.berkeley.edu pages across 9 categories, uses Gemini API
with hardened prompts to generate non-trivial QA pairs, and post-processes
to verify extractability and handle multi-answer "|" format.

Usage:
    pip install beautifulsoup4 google-genai
    export GEMINI_API_KEY="your-key-here"
    python generate2_qa.py

Output: reference_answers.json (JSONL, ~200+ QA pairs)
"""

import urllib.request
import urllib.error
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import json
import time
import re
import os
import sys
import random
import unicodedata
from google import genai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY", "")
OUTPUT_FILE = "reference_answers_3.json"
CRAWL_DELAY = 1.5          # seconds between HTTP requests
MAX_TEXT_PER_PAGE = 6000    # chars sent to Gemini per page
MIN_PAGE_TEXT = 200         # skip pages shorter than this
QUESTIONS_TARGET = 220      # overshoot slightly, prune later

# Gemini setup — model fallback chain for when free-tier quotas are exhausted
if not API_KEY:
    sys.exit("ERROR: Set GEMINI_API_KEY environment variable before running.")
gemini_client = genai.Client(api_key=API_KEY)
GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
]
PROGRESS_FILE = OUTPUT_FILE + ".progress"  # incremental save

# ---------------------------------------------------------------------------
# Seed URLs by category — curated for diversity
# ---------------------------------------------------------------------------
SEED_URLS = {
    "faculty": [
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/klein.html",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/abbeel.html",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/stoica.html",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/russell.html",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/garcia.html",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/keutzer.html",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/efros.html",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/malik.html",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/pister.html",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/tomlin.html",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/svlevine.html",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/hu.html",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/bokor.html",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/king.html",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/patterson.html",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/karp.html",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/stonebraker.html",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/sastry.html",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/fearing.html",
    ],
    "courses_schedules": [
        "https://www2.eecs.berkeley.edu/Courses/CS/",
        "https://www2.eecs.berkeley.edu/Courses/EE/",
        "https://www2.eecs.berkeley.edu/Courses/CS61A",
        "https://www2.eecs.berkeley.edu/Courses/CS61B",
        "https://www2.eecs.berkeley.edu/Courses/CS170",
        "https://www2.eecs.berkeley.edu/Courses/CS188",
        "https://www2.eecs.berkeley.edu/Courses/CS189",
        "https://www2.eecs.berkeley.edu/Courses/EE120",
        "https://www2.eecs.berkeley.edu/Scheduling/CS/schedule.html",
        "https://www2.eecs.berkeley.edu/Scheduling/EE/schedule.html",
        "https://www2.eecs.berkeley.edu/Scheduling/CS/schedule-draft.html",
        "https://eecs.berkeley.edu/academics/courses/numbering/",
    ],
    "history_dept": [
        "https://eecs.berkeley.edu/about/history/",
        "https://eecs.berkeley.edu/about/history/gier/",
        "https://eecs.berkeley.edu/about/history/first-women/",
        "https://eecs.berkeley.edu/about/",
        "https://eecs.berkeley.edu/about/diversity/",
        "https://eecs.berkeley.edu/about/visiting/",
    ],
    "rankings_stats": [
        "https://eecs.berkeley.edu/about/by-the-numbers/",
        "https://eecs.berkeley.edu/people/leadership/",
    ],
    "undergrad_programs": [
        "https://eecs.berkeley.edu/academics/undergraduate/",
        "https://eecs.berkeley.edu/academics/undergraduate/eecs-cs-comparison-chart/",
        "https://eecs.berkeley.edu/academics/undergraduate/cs-ba/",
        "https://eecs.berkeley.edu/academics/undergraduate/eecs-bs/",
        "https://eecs.berkeley.edu/academics/undergraduate/ece-bs/",
        "https://eecs.berkeley.edu/resources/undergrads/honors/",
        "https://eecs.berkeley.edu/cs-scholars/",
        "https://eecs.berkeley.edu/academics/undergraduate/summer-research/",
        "https://eecs.berkeley.edu/academics/prospective-women/",
        "https://eecs.berkeley.edu/resources/undergrads/eecs-2/degree-reqs-lowerdiv-2/",
    ],
    "grad_programs": [
        "https://eecs.berkeley.edu/book/phd/",
        "https://eecs.berkeley.edu/book/phd/coursework/",
        "https://eecs.berkeley.edu/academics/graduate/",
        "https://eecs.berkeley.edu/academics/graduate/research-programs/admissions/",
        "https://eecs.berkeley.edu/academics/graduate/industry-programs/meng/",
        "https://eecs.berkeley.edu/academics/graduate/industry-programs/5yrms/",
        "https://eecs.berkeley.edu/academics/graduate/faq-3/",
        "https://eecs.berkeley.edu/academics/graduate/fellowships/",
        "https://eecs.berkeley.edu/academics/graduate/recommended-coursework/",
    ],
    "awards": [
        "https://www2.eecs.berkeley.edu/Students/Awards/11",
        "https://www2.eecs.berkeley.edu/Students/Awards/17",
        "https://www2.eecs.berkeley.edu/Students/Awards/15",
        "https://www2.eecs.berkeley.edu/Students/Awards/146/",
        "https://www2.eecs.berkeley.edu/Students/Awards/100",
        "https://www2.eecs.berkeley.edu/Students/Awards/13/",
        "https://www2.eecs.berkeley.edu/Faculty/Awards/",
        "https://eecs.berkeley.edu/people/students-2/awards/",
    ],
    "research_labs": [
        "https://www2.eecs.berkeley.edu/Research/Areas/",
        "https://www2.eecs.berkeley.edu/Research/Areas/AI",
        "https://www2.eecs.berkeley.edu/Research/Areas/SEC",
        "https://www2.eecs.berkeley.edu/Research/Areas/CIR",
        "https://www2.eecs.berkeley.edu/Research/Areas/BIO",
        "https://www2.eecs.berkeley.edu/Research/Areas/ARC",
        "https://www2.eecs.berkeley.edu/Research/Areas/Centers/",
    ],
    "news_events": [
        "https://eecs.berkeley.edu/news/sylvia-ratnasamy-named-2025-acm-fellow/",
        "https://eecs.berkeley.edu/news/graduate-student-syed-tahmid-mahbub-awarded-paul-daisy-soros-fellowship/",
        "https://eecs.berkeley.edu/news/in-memoriam-beresford-parlett-1932-2026/",
        "https://eecs.berkeley.edu/news/tyler-hou-named-2026-hertz-fellowship-finalist/",
        "https://eecs.berkeley.edu/news/eric-paulos-elected-to-the-acm-chi-academy/",
        "https://eecs.berkeley.edu/2025/12/the-2025-eecs-distinguished-alumni/",
        "https://eecs.berkeley.edu/2024/05/the-2024-eecs-distinguished-alumni/",
        "https://eecs.berkeley.edu/2026/01/lensless-imaging-redefined-by-information-theory/",
        "https://eecs.berkeley.edu/2025/09/uc-berkeley-eecs-announces-the-vigyan-singhal-fund-for-chip-design-education-donor/",
        "https://eecs.berkeley.edu/2025/08/eecs-researchers-develop-a-scalable-quantum-platform-for-high-speed-communications/",
        "https://eecs.berkeley.edu/2024/12/uc-berkeley-announces-ai-center-of-excellence/",
        "https://eecs.berkeley.edu/2025/02/edmund-bussey-the-first-black-bachelor-of-science-graduate-of-electrical-engineering-at-uc-berkeley/",
        "https://eecs.berkeley.edu/research/colloquium",
    ],
    "tech_reports": [
        "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/",
        "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/",
        "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2023/",
    ],
    "advising_enrollment": [
        "https://eecs.berkeley.edu/resources/undergrads/cs/advising/",
        "https://eecs.berkeley.edu/resources/undergrads/cs/advising/enrollment/",
        "https://eecs.berkeley.edu/resources/undergrads/cs/enrollment-policy/",
        "https://eecs.berkeley.edu/resources/undergrads/faqs/getting-into-ee-classes/",
    ],
    "alumni_people": [
        "https://eecs.berkeley.edu/people/alumni/",
        "https://eecs.berkeley.edu/people/alumni/ee-distinguished-alumni/",
        "https://eecs.berkeley.edu/people/alumni/cs-distinguished-alumni/",
        "https://eecs.berkeley.edu/people/faculty/in-memoriam/",
        "https://www2.eecs.berkeley.edu/Directories/directory-nostudents.html",
    ],
}

CATEGORY_CAPS = {
    "faculty": 20,
    "courses_schedules": 15,
    "history_dept": 8,
    "rankings_stats": 4,
    "undergrad_programs": 12,
    "grad_programs": 12,
    "awards": 10,
    "research_labs": 10,
    "news_events": 15,
    "tech_reports": 6,
    "advising_enrollment": 6,
    "alumni_people": 8,
}

QUESTIONS_PER_PAGE = {
    "faculty": 2,
    "courses_schedules": 2,
    "history_dept": 3,
    "rankings_stats": 4,
    "undergrad_programs": 3,
    "grad_programs": 3,
    "awards": 2,
    "research_labs": 2,
    "news_events": 2,
    "tech_reports": 2,
    "advising_enrollment": 2,
    "alumni_people": 3,
}


# ---------------------------------------------------------------------------
# Phase 1: Crawling
# ---------------------------------------------------------------------------

def fetch_url(url):
    """Staff-provided fetch function with browser-like User-Agent."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode("utf-8", errors="replace")
        return html
    except Exception:
        return None


def is_valid_eecs_url(url):
    """URL must match the assignment regex and not point to binary files."""
    if not re.match(
        r"https?://(?:www\d*\.)?eecs\.berkeley\.edu(?:/[^\s]*)?", url
    ):
        return False
    path = urlparse(url).path.lower()
    if path.endswith((".pdf", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".zip")):
        return False
    return True


def discover_links(html, base_url, visited):
    """Extract valid eecs.berkeley.edu links from a page."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        full = urljoin(base_url, a["href"]).split("#")[0].rstrip("/")
        if is_valid_eecs_url(full) and full not in visited:
            links.append(full)
    return list(dict.fromkeys(links))  # dedupe preserving order


def crawl_category(category, seeds, cap):
    """BFS-crawl a category starting from seeds, up to `cap` pages."""
    visited = set()
    pages = []  # list of (url, html)
    queue = list(seeds)

    while queue and len(visited) < cap:
        url = queue.pop(0).rstrip("/")
        if url in visited:
            continue
        visited.add(url)

        html = fetch_url(url)
        if html is None:
            continue

        pages.append((url, html))
        new_links = discover_links(html, url, visited)
        queue.extend(new_links)
        time.sleep(CRAWL_DELAY)

    return pages


# ---------------------------------------------------------------------------
# Phase 2: Text extraction
# ---------------------------------------------------------------------------

REMOVE_TAGS = {"nav", "footer", "header", "script", "style", "noscript", "aside"}


def extract_text(html):
    """Extract meaningful text from HTML, preserving table rows as lines."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(REMOVE_TAGS):
        tag.decompose()

    chunks = []

    # Tables: format each row as "col1 | col2 | col3"
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            row_text = " | ".join(c for c in cells if c)
            if len(row_text) > 10:
                chunks.append(row_text)
        table.decompose()  # don't double-count in the next pass

    # Everything else
    for tag in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6",
                               "dd", "dt", "span", "div"]):
        text = tag.get_text(" ", strip=True)
        if len(text) > 15:
            chunks.append(text)

    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# Phase 3: QA generation via Gemini
# ---------------------------------------------------------------------------

CATEGORY_HINTS = {
    "faculty": (
        "This is a faculty profile. Pick the MOST INTERESTING facts — NOT just "
        "office/email/phone. Prefer: specific award names + years they won them, "
        "research lab names they direct, specific best-paper awards with conferences, "
        "companies they founded, specific publications or collaborators mentioned, "
        "which courses they currently teach. Make each question unique and non-trivial."
    ),
    "courses_schedules": (
        "This is a course or schedule page. Ask about: who teaches a specific "
        "course in a specific semester, what building/room a specific course meets in, "
        "the full title of a course given its number, prerequisites, "
        "how many units. Include the semester (e.g. Spring 2026) in the question."
    ),
    "history_dept": (
        "This page has EECS department history. Ask about: specific years events "
        "happened (e.g. when departments merged), names of historical figures and "
        "their roles, pioneering inventions developed at Berkeley (SPICE, BSD Unix, "
        "RISC, FinFET, INGRES), which professor led which project, "
        "first women/minority milestones with specific years, "
        "building naming origins. These are unique, non-trivial questions."
    ),
    "rankings_stats": (
        "This page has rankings and enrollment statistics. Ask about: "
        "specific US News rankings (e.g. what rank for Computer Engineering?), "
        "exact enrollment numbers, demographic percentages, "
        "number of ACM Turing Award winners, number of NAE members, "
        "number of degree programs offered, specific global ranking positions."
    ),
    "undergrad_programs": (
        "This page describes undergraduate programs. Ask about: "
        "differences between CS BA, EECS BS, and ECE BS (which college, which degree), "
        "specific breadth requirements, number of upper-division units needed, "
        "names of specific programs (CS Scholars, Honors), "
        "what CS Kickstart is, summer research opportunities, "
        "specific admission requirements or policies."
    ),
    "grad_programs": (
        "This is a graduate program page. Ask about: specific credit/unit requirements, "
        "qualifying exam details, milestone deadlines, "
        "differences between MEng and MS/PhD, who is eligible for 5th Year MS, "
        "specific fellowship names, application deadlines."
    ),
    "awards": (
        "This is an awards page. Ask about: who won a SPECIFIC award in a SPECIFIC "
        "year, the purpose/criteria of the award, when the award was first given, "
        "who the award is named after and why."
    ),
    "research_labs": (
        "This is a research area or center page. Ask about: "
        "full name of an area abbreviation (e.g. what does CIR stand for?), "
        "which specific faculty belong to an area, names of specific centers/labs, "
        "what a specific center focuses on, industry partners or collaborations."
    ),
    "news_events": (
        "This is a news article. Ask about the SPECIFIC event: "
        "who received the award/fellowship, what specific technology was developed, "
        "which year an event happened, quantitative details mentioned, "
        "names of co-authors or collaborators. Each question must require "
        "reading this specific article."
    ),
    "tech_reports": (
        "This is a technical reports listing. Ask about: "
        "the advisor of a specific thesis, the title of a thesis by a named student, "
        "which student researched a specific topic, the year of a specific thesis."
    ),
    "advising_enrollment": (
        "This is an advising or enrollment page. Ask about: "
        "specific advisor names and roles, drop-in advising hours and days, "
        "which building/room for advising, specific enrollment policies or deadlines, "
        "what email to contact for enrollment questions."
    ),
    "alumni_people": (
        "This page lists alumni or memorial faculty. Ask about: "
        "which year a specific person was named distinguished alum, "
        "what a specific person's contribution was, "
        "birth/death years for memorial faculty, historical firsts, "
        "ACM Turing Award winners who graduated from Berkeley."
    ),
}

REFERENCE_EXAMPLES = """EXAMPLES of ideal output (from the official reference data):
{"question": "What is the course number for the undergrad-level class that covers computer security topics such as cryptography, operating system security, network security, and software security?", "answer": "CS 161", "url": "URL_HERE"}
{"question": "Who is the CS student advisor?", "answer": "Gina Garcia|Floriberto Garcia|Grayson Johnston|Carol Marshall", "url": "URL_HERE"}
{"question": "Which EECS graduate student was awarded a 2025 Paul & Daisy Soros Fellowship for New Americans?", "answer": "Syed Tahmid Mahbub", "url": "URL_HERE"}
{"question": "Which university did Pieter Abbeel do his masters degree from?", "answer": "KU Leuven", "url": "URL_HERE"}
{"question": "When is the deadline for the nomination of the outstanding TA awards?", "answer": "2/18/26", "url": "URL_HERE"}
{"question": "How many minor credits do PhD students have to take?", "answer": "6+", "url": "URL_HERE"}"""


def build_prompt(text, url, category, n_questions):
    """Build the Gemini prompt for a given page."""
    hint = CATEGORY_HINTS.get(category, "")

    return f"""You are an expert at creating evaluation datasets for Retrieval-Augmented Generation (RAG) systems.

Given the text below (extracted from {url}), generate exactly {n_questions} high-quality question-answer pairs.

CATEGORY GUIDANCE:
{hint}

STRICT RULES — follow every one:
1. Each answer MUST be a SHORT text span (under 10 words) that appears verbatim or near-verbatim in the provided text.
2. Do NOT generate questions answerable by general knowledge or by an LLM without this specific document. The question should REQUIRE reading this exact page.
3. Questions should be specific: include names, years, course numbers, or other distinguishing details.
4. MULTIPLE VALID ANSWERS: If a question has multiple equally valid short-span answers from the text, include ALL of them separated by "|" with no spaces around the pipe. Example: "Gina Garcia|Floriberto Garcia|Carol Marshall"
5. Do NOT repeat similar questions. Each question must ask about a DIFFERENT fact.
6. Output ONLY valid JSON Lines — one JSON object per line, nothing else. No markdown, no commentary, no numbering.
7. Each JSON object must have exactly three keys: "question", "answer", "url"
8. The "url" field must always be exactly: "{url}"
9. Avoid yes/no questions unless the fact is truly surprising and non-obvious.
10. Avoid overly broad questions like "What is this page about?" — be specific.
11. AVOID REPETITIVE PATTERNS: Do NOT just ask "What is X's office number?", "What is X's email?", "What is X's phone number?" for every page. These are boring and formulaic. Instead, find the MOST INTERESTING and UNIQUE facts on the page — specific awards won, historical milestones, founding of companies, specific numerical statistics, policy details, or surprising facts that would be hard to guess.
12. VARIETY: Each question should test a DIFFERENT type of knowledge — mix who/what/when/where/how many questions.

{REFERENCE_EXAMPLES}

TEXT FROM THE PAGE:
{text[:MAX_TEXT_PER_PAGE]}

Generate exactly {n_questions} JSON lines now:"""


def _parse_retry_delay(error_str):
    """Extract the server-suggested retry delay from error message."""
    m = re.search(r"retry in ([\d.]+)s", str(error_str), re.IGNORECASE)
    if m:
        return min(float(m.group(1)) + 2, 120)  # add 2s buffer, cap at 2min
    m = re.search(r"retryDelay.*?(\d+)s", str(error_str))
    if m:
        return min(int(m.group(1)) + 2, 120)
    return None


def call_gemini(prompt, max_retries=6):
    """Call Gemini with rate-limit-aware retries and model fallback.

    Free tier allows 15 RPM, so on 429 we wait the server-suggested delay
    (typically 30-60s) before retrying the SAME model. Only falls back to
    the next model after repeated non-rate-limit failures.
    """
    for model_name in GEMINI_MODELS:
        consecutive_non_quota_errors = 0
        for attempt in range(max_retries):
            try:
                response = gemini_client.models.generate_content(
                    model=model_name, contents=prompt
                )
                return response.text.strip()
            except Exception as e:
                err_str = str(e)
                is_quota = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

                if is_quota:
                    consecutive_non_quota_errors = 0
                    suggested = _parse_retry_delay(err_str)
                    wait = suggested if suggested else 60
                    print(f"  [{model_name}] rate limited (attempt {attempt+1}). "
                          f"Waiting {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    consecutive_non_quota_errors += 1
                    if consecutive_non_quota_errors >= 3:
                        print(f"    {model_name} failing, trying next model...")
                        break
                    wait = 5 * (attempt + 1)
                    print(f"  [{model_name}] error (attempt {attempt+1}): "
                          f"{err_str[:100]}. Waiting {wait}s...")
                    time.sleep(wait)
    return ""


def generate_qa_for_page(text, url, category, n_questions):
    """Generate QA pairs for a single page."""
    prompt = build_prompt(text, url, category, n_questions)
    raw = call_gemini(prompt)
    if not raw:
        return []

    pairs = []
    for line in raw.split("\n"):
        line = line.strip()
        # Strip markdown code fences if model wraps output
        if line.startswith("```"):
            continue
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if "question" in obj and "answer" in obj:
                obj["url"] = url  # force correct URL
                pairs.append(obj)
        except json.JSONDecodeError:
            continue
    return pairs


# ---------------------------------------------------------------------------
# Phase 4: Post-processing
# ---------------------------------------------------------------------------

def normalize(text):
    """Lowercase, remove punctuation and extra whitespace — mirrors eval."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def answer_in_text(answer_str, page_text):
    """Check that at least one sub-answer appears in the page text."""
    norm_text = normalize(page_text)
    sub_answers = [a.strip() for a in answer_str.split("|") if a.strip()]
    found = 0
    for ans in sub_answers:
        if normalize(ans) in norm_text:
            found += 1
    # Accept if at least half the sub-answers are found (allows counting Qs)
    return found >= max(1, len(sub_answers) // 2)


def normalize_multi_answer(answer_str):
    """Strip whitespace around | separators."""
    parts = [a.strip() for a in answer_str.split("|") if a.strip()]
    return "|".join(parts)


def deduplicate(qa_list):
    """Remove near-duplicate questions (>80% word overlap)."""
    kept = []
    seen_words = []
    for item in qa_list:
        words = set(normalize(item["question"]).split())
        is_dup = False
        for prev_words in seen_words:
            if not words or not prev_words:
                continue
            overlap = len(words & prev_words) / min(len(words), len(prev_words))
            if overlap > 0.80:
                is_dup = True
                break
        if not is_dup:
            kept.append(item)
            seen_words.append(words)
    return kept


def postprocess(qa_list, page_texts):
    """Validate answers, normalize multi-answers, deduplicate."""
    valid = []
    for item in qa_list:
        answer = normalize_multi_answer(item.get("answer", ""))
        if not answer:
            continue
        # Length check: each sub-answer under 10 words
        sub_answers = answer.split("|")
        if any(len(a.split()) > 10 for a in sub_answers):
            continue
        item["answer"] = answer

        url = item.get("url", "")
        if url in page_texts and not answer_in_text(answer, page_texts[url]):
            continue
        valid.append(item)

    return deduplicate(valid)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_progress():
    """Load previously generated QA pairs and processed URLs from progress file."""
    done_urls = set()
    existing_qa = []
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    existing_qa.append(obj)
                    done_urls.add(obj.get("url", ""))
                except json.JSONDecodeError:
                    continue
        print(f"Resumed: {len(existing_qa)} QA pairs from {len(done_urls)} URLs")
    return existing_qa, done_urls


def save_progress(pairs):
    """Append QA pairs to the progress file incrementally."""
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        for item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    all_qa, done_urls = load_progress()
    page_texts = {}
    category_counts = {}

    total_categories = len(SEED_URLS)
    for cat_idx, (category, seeds) in enumerate(SEED_URLS.items(), 1):
        cap = CATEGORY_CAPS.get(category, 10)
        n_q = QUESTIONS_PER_PAGE.get(category, 2)

        print(f"\n{'='*60}")
        print(f"[{cat_idx}/{total_categories}] Category: {category} "
              f"(cap={cap} pages, {n_q} Qs/page)")
        print(f"{'='*60}")

        pages = crawl_category(category, seeds, cap)
        print(f"  Crawled {len(pages)} pages")

        cat_qa = []
        for i, (url, html) in enumerate(pages):
            if url in done_urls:
                print(f"  [{i+1}/{len(pages)}] SKIP (already done): {url}")
                continue

            text = extract_text(html)
            if len(text) < MIN_PAGE_TEXT:
                print(f"  [{i+1}/{len(pages)}] SKIP (too little text): {url}")
                continue

            page_texts[url] = text
            print(f"  [{i+1}/{len(pages)}] Generating QA: {url}")

            pairs = generate_qa_for_page(text, url, category, n_q)
            cat_qa.extend(pairs)
            save_progress(pairs)  # incremental save
            print(f"    -> {len(pairs)} QA pairs")

            time.sleep(5)  # ~12 req/min, safely under 15 RPM free tier limit

        category_counts[category] = len(cat_qa)
        all_qa.extend(cat_qa)
        print(f"  Category total (raw): {len(cat_qa)} QA pairs")

    print(f"\n{'='*60}")
    print(f"Post-processing {len(all_qa)} raw QA pairs...")
    print(f"{'='*60}")

    final_qa = postprocess(all_qa, page_texts)

    # Shuffle for variety then write
    random.shuffle(final_qa)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in final_qa:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(final_qa)} QA pairs to {OUTPUT_FILE}")
    print("\nPer-category breakdown (raw, before dedup):")
    for cat, count in category_counts.items():
        print(f"  {cat:20s}: {count}")
    print(f"\nFinal count after validation + dedup: {len(final_qa)}")


if __name__ == "__main__":
    main()
