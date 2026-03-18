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
"You will need to clean this data and convert it into a file format that suits your model development."
chunking -JSONL is easy to load line-by-line
works w BM25 and FAISS indexing

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
