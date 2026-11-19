# JobMatch

Search engine over ~7,000 live job listings, pulled from public job-board APIs.
Combines keyword search with embeddings and measures whether the combination
actually helps.

Work in progress. Retrieval works; the agent layer isn't built yet.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # Linux/mac: . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add a Groq API key (only needed later)
```

## Usage

```bash
python -m jobmatch.resolve      # find each company's ATS board
python -m jobmatch.ingest       # fetch listings into data/jobs.db
python -m jobmatch.index        # embed and build the vector index
python -m jobmatch.query "backend engineer working on search" --compare
```

`--compare` runs the same query through keyword-only, embeddings-only and the
fusion so you can see where they disagree.

Other flags: `--location india`, `--remote`, `--seniority intern`,
`--mode bm25|dense|hybrid`, `--store qdrant`.

Edit `companies.txt` to change what gets indexed. The resolver tries plausible
slugs against each ATS and saves hits to `data/slugs.json`. If a company
doesn't resolve but you know its careers URL, paste the URL in instead of the
name.

## Where the data comes from

Greenhouse, Lever and Ashby publish public job-board APIs that companies use
for their own careers pages. Remotive and Arbeitnow add remote and European
listings. Nothing is scraped from boards that forbid it.

Remotive and Arbeitnow both ask that listings link back to them, so every
record keeps its source URL and the source is shown with each result.

## Ingestion

Re-running is safe. Every listing gets an id derived from its source and
source id, so a second run updates rather than duplicates. Listings that stop
appearing get marked expired instead of deleted.

The same role often appears on two boards. Matching them needs the location
strings to agree, and they don't: the same office shows up as `bengaluru`,
`Bangalore, Karnataka`, `Bengaluru, India` and `Bengaluru-VTP, India`.
`locations.py` folds those together (1,203 raw strings → 556 canonical), which
took cross-source duplicate detection from 0 matches to 9.

## Retrieval

Two indexes over the same listings:

- **SQLite FTS5** for keyword matching, with title and company weighted above
  the description body. Finds the posting that literally says "OpenSearch".
- **Embeddings** (`bge-small-en-v1.5` via fastembed, ONNX, no PyTorch) for
  semantic matching. Finds the Elasticsearch role when you searched for
  "search and ranking".

Results merge with Reciprocal Rank Fusion, which combines by rank position.
That avoids having to normalise BM25 scores against cosine similarities, which
aren't on the same scale.

Filters (location, remote, seniority, company) resolve to an id set in SQL and
apply to both retrievers.

## Vector store

Both an exact numpy implementation and Qdrant sit behind one interface.
Benchmarked on this corpus:

| store | build | query |
|---|---|---|
| numpy (exact) | ~70,000 vec/s | 0.22 ms |
| Qdrant (local, on disk) | ~216 vec/s | 1.71 ms |

Exact search is the default here. At 7k vectors a brute-force matrix multiply
is already sub-millisecond, and Qdrant's local mode ignores payload indexes so
the filtering advantage isn't available either. Qdrant stays in the codebase
and stays correct — it's the right choice against a server and a bigger index.

Embedding is the slow part, so vectors are cached against a hash of the text
that produced them. Changing what gets embedded invalidates the cache;
changing unrelated code doesn't. A re-index went from about 3 hours to under a
second.

## Evaluation

```bash
python -m jobmatch.label --seed    # judge pooled results by hand
python -m jobmatch.evaluate        # Recall@k, P@5, MRR, nDCG@10 per mode
```

Judging only what one retriever returns would decide the comparison in
advance, so candidates are pooled: the union of what each mode returns,
shuffled before display. Anything outside the pool counts as non-relevant,
which is the usual limitation of pooled judgements.

## Tests

```bash
pytest
```

70 tests. Vector stores are tested against both backends with the same suite.
Embedding is stubbed with a hashing embedder so the tests need no model
download and no network.

## Still to do

- Diversify results; a single company can take most of the top 10
- LLM agent over search with tool calls and remembered preferences
- Reranker over the fused top-k
- Scheduled re-ingestion
