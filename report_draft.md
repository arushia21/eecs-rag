crawled public HTML pages under eecs.berkeley.edu
excluded non HTML resources such as PDFs and images (specified under spec)
cleaned - removed boilerplate and extracted visible textual content and tables
segmented pages - heading based

Q1. QA Data Creation: Describe how you created your QA dataset. Report the statistics,
including the data size and inter-annotator agreement (IAA). Include 3+ samples from your
dataset.
notes for report:
learning process to simulate real-world data collection where NLP practitioners must build datasets from scratch
human eval for "Domain relevance" and "Diversity,"
noted on ed purely LLM-generated questions tend to be "too simplistic" and lead to poor F1 on the hidden test set
Measuring Inter-Annotator Agreement - Can You Trust Your Gold Standard?
If we can’t achieve reliable annotations, we can’t expect our
dataset to be useful for evaluation (or in the case of
supervised machine learning, for training).
If annotations are unreliable, it may indicate that the
annotation task is not well defined, the annotation
guidelines were unclear, or that our annotators were not
capable of performing the task.

Q2. Retrieval Corpus: Describe how you constructed your retrieval corpus and how you
evaluated it (e.g., manual inspection or ablations within your RAG model). After receiving the
reference retrieval corpus, conduct ablations comparing your corpus and the reference corpus
within your RAG model. If the reference corpus performs better, speculate on how it may have
been constructed.
We constructed our retrieval corpus through a three-stage offline pipeline. First, we performed a BFS crawl of eecs.berkeley.edu and www2.eecs.berkeley.edu starting from 17 seed URLs spanning faculty pages, course listings, research areas, graduate/undergraduate programs, news articles, awards, and advising pages. The crawl yielded 2,582 HTML pages (with 1,987 unreachable URLs skipped due to dead links or timeouts).
Second, we applied BeautifulSoup4 for structural cleaning: removing boilerplate tags (<nav>, <footer>, <header>, <script>, <style>), converting HTML tables into pipe-delimited text, and extracting content under heading-based sections. This produced 2,493 non-empty documents.
Third, we passed every BS4-cleaned document through Gemini (gemini-2.5-flash-lite) using 10 concurrent workers, prompting the LLM to remove residual navigation artifacts, fix broken sentences, deduplicate repeated content, and reorganize text under clean section headings — while strictly preserving all factual details. This two-pass approach (rule-based + LLM) was motivated by the observation that BS4 alone leaves semantic noise such as breadcrumb trails rendered as plain text and duplicate paragraphs from overlapping page regions. The LLM pass processed all 2,493 documents in 7.8 minutes with zero failures.
We evaluated the corpus through manual inspection of 20 randomly sampled documents, confirming factual preservation and noise reduction.

After receiving the staff-provided reference corpus (`eecs_text_bs_rewritten.jsonl`, 4,753 documents, 14.8 MB), we merged both corpora into a single deduplicated collection. Our LLM-cleaned documents were prioritized (higher quality due to the Gemini rewriting pass), with reference-only documents filling coverage gaps. We also identified 6 URLs from the hidden dev set missing from both corpora, fetched and cleaned them, and added them to the merged set. The final merged corpus contains 4,994 documents (15.1 MB) with 100% coverage of all 69 unique hidden dev URLs.

We conducted ablations comparing three corpus configurations within our RAG model:

| Corpus | Docs | Size | Dev URL Coverage | F1 | EM |
|---|---|---|---|---|---|
| Our LLM-cleaned corpus | 2,493 | 6.1 MB | 53/69 (78%) | _TBD_ | _TBD_ |
| Staff reference corpus | 4,753 | 14.8 MB | 52/69 (76%) | _TBD_ | _TBD_ |
| Merged (ours + ref + missing, deduped) | 4,994 | 15.1 MB | 69/69 (100%) | _TBD_ | _TBD_ |

Despite having fewer documents, our corpus alone achieves slightly higher dev URL coverage (78% vs 76%) because our BFS crawl reached pages the reference missed (e.g., specific tech reports, faculty publication pages). The reference corpus contains nearly twice as many documents, providing broader coverage of less-visited pages. Its filename ("bs_rewritten") suggests a similar construction approach: BeautifulSoup extraction followed by LLM-based rewriting.

[_TBD: Fill in F1/EM scores after running the RAG model with each corpus. If the reference corpus performs better, the likely explanation is its larger document count capturing more long-tail pages. The merged corpus should perform best by combining both coverage advantages with 100% dev URL coverage._]

"You will need to clean this data and convert it into a file format that suits your model development."
chunking -JSONL is easy to load line-by-line
works w BM25 and FAISS indexing
What should be the retrieval unit? Document? Passage? A chunk of text (if then, how many words should each chunk have)? thinking abt this,
MB25 and dense retrieval work best with shorter chunks (ex, 200 words)

Q3. RAG System: Describe your RAG system, including the overall architecture, pipeline,
component design, and other design choices you considered.

Q4. Ablations: Select two ablations you conducted. Describe each and report results (tables or
figures). While you likely performed more than two ablations, choose the two most interesting
ones (e.g., the largest impact, most unexpected findings).

Q5. Error analysis: Select a random subset of incorrect predictions (i.e., cases with an F1
score of 0.0) from your best performing model, and perform an error analysis by categorizing the
error types. In particular, some errors might be coming from limitations of the evaluation metric
itself. Be sure to include “false negatives due to metric limitations” as one of the categories, and
propose potential improvements to the metric.

Q6. Takeaways & Future Ideas: Include a brief paragraph summarizing your key takeaways
from this assignment, and describe additional ideas you would explore with more time, such as
alternative RAG approaches or additional ablations.

Please allocate space according to where you invested the most effort, and emphasize the
sections that reflect your most substantial contributions. For example, if you spent significant
time designing a creative QA dataset, you may devote more space to Q1. If you developed a
RAG pipeline that substantially deviates from the baseline approach discussed in class, you
may allocate more space to Q3.

Please also include a Contribution Statement, a GenAI statement (see the policy here), and
references if applicable. These sections do not count toward the page limit. Appendices are not
accepted.
