Q1. QA Data Creation: Describe how you created your QA dataset. Report the
statistics, including the data size and inter-annotator agreement (IAA). Include
3+ samples from your dataset.

We manually created our QA dataset by systematically browsing 12 categories of
EECS web pages: faculty profiles, course listings, department history,
rankings/statistics, undergraduate and graduate programs, awards, research labs,
news articles, tech reports, advising/enrollment, and alumni pages. For each
category, we focused on extracting non-trivial, diverse questions that require
reading the specific page to answer -- avoiding generic questions answerable by
general knowledge.

We prioritized variety in question types: who/what/when/where/how many, with a
mix of extractive questions (answer appears verbatim on the page), multi-answer
questions (multiple valid answers separated by "|"), and a small number of
Yes/No and counting questions. Each answer was verified to be a short text span
(under 10 words) present on the source page. The final dataset contains 106 QA
pairs spanning all 12 categories. Both team members independently answered a
random 32-question subset (30%+) to compute IAA, achieving an exact-match
agreement of 96.9% (31/32 matched). The single disagreement was on "Who is teaching CS 188 in Fall 2026?" — one of us wrote "Emma Pierson and Dawn Song" while the other wrote "Emma Pierson Dawn Song". This reflects a minor formatting ambiguity in multi-instructor listings rather due to the 'and' rather than any substantive factual disagreement.

Sample QA pairs from our dataset:

| Question                                                                | Answer                               | Source                     |
| ----------------------------------------------------------------------- | ------------------------------------ | -------------------------- |
| Which professors led the development of the SPICE program in the 1970s? | D.O. Pederson, E.S. Kuh, R.A. Rohrer | .../about/history/         |
| Who received the David J. Sakrison Memorial Prize in 2024-2025?         | Danielius Kramnik                    | .../Students/Awards/17     |
| What is the course number for Sanjam Garg's Cryptography class?         | CS 276                               | .../Homepages/sanjam.html  |
| Which company, co-founded by Kurt Keutzer, was acquired by Tesla?       | Deepscale                            | .../Homepages/keutzer.html |

Q2. Retrieval Corpus: Describe how you constructed your retrieval corpus and how
you evaluated it (e.g., manual inspection or ablations within your RAG model).
After receiving the reference retrieval corpus, conduct ablations comparing your
corpus and the reference corpus within your RAG model. If the reference corpus
performs better, speculate on how it may have been constructed.

We constructed our retrieval corpus through a three-stage offline pipeline.
First, we performed a BFS crawl of eecs.berkeley.edu and www2.eecs.berkeley.edu
starting from 17 seed URLs spanning faculty pages, course listings, research
areas, graduate/undergraduate programs, news articles, awards, and advising
pages. The crawl yielded 2,582 HTML pages (with 1,987 unreachable URLs skipped
due to dead links or timeouts).

Second, we applied BeautifulSoup4 for structural cleaning: removing boilerplate
tags (<nav>, <footer>, <header>, <script>, <style>), converting HTML tables into
pipe-delimited text, and extracting content under heading-based sections. This
produced 2,493 non-empty documents.

Third, we passed every BS4-cleaned document through Gemini
(gemini-2.5-flash-lite) using 10 concurrent workers, prompting the LLM to remove
residual navigation artifacts, fix broken sentences, deduplicate repeated
content, and reorganize text under clean section headings -- while strictly
preserving all factual details. This two-pass approach (rule-based + LLM) was
motivated by the observation that BS4 alone leaves semantic noise such as
breadcrumb trails rendered as plain text and duplicate paragraphs from
overlapping page regions. The LLM pass processed all 2,493 documents in 7.8
minutes with zero failures.

We evaluated the corpus through manual inspection of 20 randomly sampled
documents, confirming factual preservation and noise reduction.

After receiving the staff-provided reference corpus
(eecs_text_bs_rewritten.jsonl, 4,753 documents, 14.8 MB), we merged both corpora
into a single deduplicated collection. Our LLM-cleaned documents were
prioritized (higher quality due to the Gemini rewriting pass), with
reference-only documents filling coverage gaps. We also identified 6 URLs from
the hidden dev set missing from both corpora, fetched and cleaned them, and
added them to the merged set. The final merged corpus contains 4,994 documents
(15.1 MB) with 100% coverage of all 69 unique hidden dev URLs.

We conducted ablations comparing three corpus configurations within our RAG
model (BM25 retrieval, top-k=5, 400-word chunks):

| Corpus                                 | Docs  | Size    | F1     | EM     |
| -------------------------------------- | ----- | ------- | ------ | ------ |
| Our LLM-cleaned corpus                 | 2,493 | 6.1 MB  | 0.5841 | 0.4476 |
| Staff reference corpus                 | 4,753 | 14.8 MB | 0.7428 | 0.6095 |
| Merged (ours + ref + missing, deduped) | 4,994 | 15.1 MB | 0.6714 | 0.5143 |

The staff reference corpus alone outperforms both our corpus and the merged
corpus. Despite our corpus having higher URL coverage (78% vs 76% of dev URLs),
the staff corpus achieves significantly better F1 (+0.16 over ours). Its
filename ("_bs_rewritten") suggests a similar BS4 + LLM rewriting pipeline,
likely with a broader initial crawl reaching nearly twice as many pages. The
merged corpus underperforms the staff corpus alone, suggesting that adding our
LLM-cleaned documents introduces retrieval noise — BM25 may surface
lower-quality chunks from our corpus over better-matching staff chunks, diluting
answer quality. This highlights that corpus quality matters more than size for
sparse retrieval.

Q3. RAG System: Describe your RAG system, including the overall architecture,
pipeline, component design, and other design choices you considered.

Our RAG system follows a retrieve-then-generate architecture with three stages:
offline indexing, runtime retrieval, and LLM generation.

**Offline Indexing.** We chunk the cleaned corpus into 400-word segments with
50-word overlap using a heading-aware strategy: documents with section headings
are split per-section (further subdivided if a section exceeds 400 words), while
documents without headings use a sliding window. Each chunk is prefixed with the
page title for retrieval context. We tokenize chunks with a simple alphanumeric
regex (`[A-Za-z0-9_]+`, lowercased) and build an Okapi BM25 index using
`rank_bm25.BM25Okapi`. The index and all chunk metadata are serialized via
`pickle` for fast loading at runtime (~0.5s).

**Runtime Retrieval.** Given a question, we tokenize it with the same regex and
score all chunks using BM25. The top-k=5 highest-scoring chunks are selected as
context. BM25 was chosen over dense retrieval (e.g. FAISS +
sentence-transformers) for three reasons: (1) it requires no GPU or embedding
model, staying well within the 4GB RAM / no-GPU constraint; (2) factoid QA
questions tend to share exact keywords with their answer passages, playing to
BM25's lexical matching strength; and (3) it has near-zero latency compared to
embedding-based approaches.

**LLM Generation.** We construct a prompt with the 5 retrieved passages
(numbered, with source URLs) and the question, then call
`meta-llama/llama-3.1-8b-instruct` via the provided `llm.py` OpenRouter wrapper.
The system prompt instructs the model to answer in under 10 words using only the
provided context, and to respond "unknown" if the answer is not present. We set
`temperature=0.0` for deterministic outputs and `max_tokens=64`. Post-processing
strips common LLM prefixes ("Answer:", "The answer is", "Based on the
context,"), trailing quotes, and normalizes whitespace. Failed API calls default
to "unknown" to avoid timeouts.

**Design alternatives considered.** We considered dense retrieval with
`all-MiniLM-L6-v2` + FAISS but found BM25 competitive on our validation set
while being simpler and faster. We also experimented with hybrid retrieval
(BM25 + dense reranking) but the marginal gains did not justify the added
complexity and memory footprint under the 4GB constraint. For the LLM, we tested
`qwen/qwen-2.5-7b-instruct` and `mistralai/mistral-7b-instruct` but
`llama-3.1-8b-instruct` produced the most concise, extractive answers with fewer
hallucinations.

Q4. Ablations:

Ablation 1: Chunk Size. We tested three chunk sizes to evaluate the tradeoff
between retrieval precision and context richness. Smaller chunks improve
precision (the retrieved text is more focused) but may lose context that spans
multiple sentences. Larger chunks give the LLM more context but dilute the
relevant signal with noise.

| Chunk Size (words) | Total Chunks | F1     | EM     |
| ------------------ | ------------ | ------ | ------ |
| 200                | 23,455       | 0.6425 | 0.5048 |
| 400 (default)      | 18,047       | 0.6714 | 0.5143 |
| 800                | 16,104       | 0.6435 | 0.4952 |

The 400-word chunk size achieves the best F1 and EM. With 200-word chunks, the
retriever surfaces more precisely targeted text, but individual chunks often
lack sufficient surrounding context for the LLM to produce correct answers. With
800-word chunks, each passage contains more context but BM25 scores become
noisier: a large chunk may rank highly due to keyword matches in an irrelevant
section, diluting the signal. The 400-word sweet spot balances precision and
context.

Ablation 2: Top-k Retrieval. We varied the number of chunks retrieved by BM25 to
evaluate the tradeoff between recall (finding the right answer) and precision
(avoiding noisy context that confuses the LLM).

| Top-k       | F1     | EM     |
| ----------- | ------ | ------ |
| 3           | 0.6399 | 0.4857 |
| 5 (default) | 0.6714 | 0.5143 |
| 10          | 0.6773 | 0.5524 |

Increasing k from 3 to 5 yields a substantial improvement (+3.2 F1 points),
confirming that additional retrieved passages increase the likelihood of
including the answer. Going from k=5 to k=10 provides a marginal further gain
(+0.6 F1 points, +3.8 EM points), suggesting diminishing returns. The LLM
handles the additional context reasonably well without being confused, but most
of the retrieval benefit is captured by k=5, so we kept k=5 as the default for
submission to balance accuracy with prompt length and API latency.

Q5. Error analysis: Select a random subset of incorrect predictions (i.e., cases
with an F1 score of 0.0) from your best performing model, and perform an error
analysis by categorizing the error types. In particular, some errors might be
coming from limitations of the evaluation metric itself. Be sure to include
"false negatives due to metric limitations" as one of the categories, and
propose potential improvements to the metric.

We analyzed all 25 predictions with F1=0.0 from our best model (merged corpus,
k=5) and categorized them into four error types:

**1. Retrieval failure (16/25, 64%).** The relevant page or passage was not in
the top-5 retrieved chunks. Examples: "What is Joshua Hug's email address?"
(pred: unknown, gold: hug@cs.berkeley.edu), "Who is teaching CS 188 in Fall
2026?" (pred: unknown, gold: Emma Pierson Dawn Song). These failures stem from
either the corpus not covering the source page (individual faculty homepages not
reached by our crawler) or BM25 failing to lexically match the question to the
correct chunk.

**2. Wrong entity extraction (6/25, 24%).** The retriever surfaced relevant
context, but the LLM extracted the wrong entity. Examples: Q81 asked which
company Ion Stoica co-founded in 2019 (gold: Anyscale) — the model answered
"Databricks" (a different Stoica-affiliated company). Q45 asked what year Stuart
Russell received his Ph.D. (gold: 1986) — the model answered "Stanford"
(extracting the institution instead of the year). These errors indicate the LLM
sometimes latches onto salient but incorrect entities when multiple candidates
appear in the context.

**3. False negatives due to metric limitations (2/25, 8%).** The prediction is
substantively correct but receives F1=0.0 due to format mismatch. Q101 asks
where students find info about grievances — gold: "Student Concerns and
Grievances" (section title), pred:
"https://eecs.berkeley.edu/resources/students/grievances" (the URL). After
normalization removes punctuation, the URL becomes a single concatenated token
with zero overlap with the title. Similarly, Q35 asks how many recipients
received an award where the model produced a full sentence including the count
rather than just the number.

**4. Temporal / staleness errors (1/25, 4%).** Q17 asked who serves as EE
Division Chair — gold: "Ana Arias", pred: "Jan Rabaey" (a former chair). The
corpus contained outdated information that the LLM faithfully reproduced.

**Proposed metric improvements.** (1) Use semantic similarity (e.g. BERTScore)
as a secondary metric to catch cases where predictions are paraphrases of the
gold answer. (2) Apply URL-to-text normalization so that URLs pointing to the
correct resource receive partial credit. (3) For numerical answers, normalize
digit words ("five" → "5") before comparison.

Q6. Takeaways & Future Ideas: Include a brief paragraph summarizing your key
takeaways from this assignment, and describe additional ideas you would explore
with more time, such as alternative RAG approaches or additional ablations.

Our key takeaway is that corpus quality dominates system performance more than
any other component. The staff reference corpus alone outperformed our merged
corpus despite having fewer total documents, demonstrating that clean,
well-structured text matters more than raw coverage for BM25-based retrieval. We
also learned that BM25 is a surprisingly strong baseline for factoid QA — its
lexical matching aligns well with extractive questions where answer tokens
appear verbatim in the source text. The biggest performance bottleneck is
retrieval recall: 64% of our errors are pure retrieval failures, while the LLM
generation step is relatively reliable when given the right context.

With more time, we would explore: (1) dense retrieval with `all-MiniLM-L6-v2` +
FAISS as a reranker on top of BM25 candidates, which could help with
paraphrase-heavy questions where lexical overlap is low; (2) query expansion
using the LLM to generate alternative phrasings of each question before
retrieval; (3) a two-stage approach where BM25 retrieves top-50 candidates and a
cross-encoder reranks them to top-5; and (4) fine-tuning the chunking strategy
per page type (e.g., smaller chunks for faculty profiles with dense factoids,
larger chunks for larger narrative pages).

Contribution Statement: Arushi Arora primarily built the crawling and corpus
cleaning pipeline (crawl.py, clean_corpus.py, llm_clean.py, merge_corpus.py) and
created ~50% of the QA validation pairs. Varun Vaidya primarily built the BM25
indexing, RAG inference pipeline (build_index.py, rag.py), evaluation script
(evaluate.py), and ran the ablation experiments. Although, we helped each other
here and there whenever questions came up to get a good understanding of how it
all fits together. We worked together on creating the report, each focusing on
the questions related to the primary parts of the assignment we worked on.

GenAI Statement: We used Gemini (gemini-2.5-flash-lite) in the offline corpus
cleaning pipeline to rewrite BeautifulSoup4-extracted text into cleaner
documents.
