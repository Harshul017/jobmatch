# JobMatch

A job-matching agent over a live index of real job listings. You say what you
are looking for; it retrieves matching roles, ranks them, and explains each
match using only the text of the listing it retrieved.

Built around three things that are easy to claim and hard to do well: a jobs
index that keeps changing, hybrid retrieval that is measured rather than
assumed, and an LLM that is kept on a short leash.

## Status

- [x] Phase 1 — ingestion, normalisation, dedupe, SQLite storage
- [x] Phase 2 — hybrid retrieval (BM25 + embeddings + RRF) and evaluation
- [ ] Phase 3 — tool-calling agent with preference memory
- [ ] Phase 4 — LLM output evals, API, Docker, CI

## How it works

**Sources.** Jobs come from the public job-board APIs that companies already
publish through: Greenhouse, Lever and Ashby, plus the Remotive and Arbeitnow
aggregator feeds. No scraping of boards that forbid it.

**Keeping the index fresh.** Ingestion is idempotent. Each listing gets a
stable id, so re-running updates in place. The same role posted to two boards
collapses to one row via a normalised `(company, title, location)` key.
Anything a source stops returning is marked expired rather than deleted.

**Search.** Each listing is indexed twice: SQLite FTS5 for keyword matching
(so "OpenSearch" finds the posting that names it) and embeddings for semantic
matching (so "AI agents" finds "LLM assistants"). The two rankings merge with
Reciprocal Rank Fusion, which combines by position and so never has to put
BM25 scores and cosine similarities on a common scale.

**Why exact search, not a vector database.** Both are implemented behind one
`VectorStore` interface and benchmarked against each other. At this corpus
size an exact numpy matrix multiply wins on both axes:

| store | build | query |
|---|---|---|
| numpy (exact) | ~70,000 vec/s | 0.22 ms |
| Qdrant (local, on disk) | ~216 vec/s | 1.71 ms |

Local Qdrant also ignores payload indexes, so the filtering advantage that
would justify it is not available either. Qdrant stays in the codebase and
stays correct -- it is the right choice against a server and a corpus large
enough that exact search stops being instant -- but defaulting to it here
would be cargo-culting a vector database into a problem that does not have
one yet.

**Embeddings are cached.** Embedding the corpus is by far the slowest step, so
vectors are stored against a hash of the exact text that produced them. A
re-index reuses everything unchanged and re-embeds only what actually moved,
which turns a multi-hour rebuild into about half a second.

**The agent.** An open-weights model gets three tools — `search_jobs`,
`get_job`, `save_preference` — and a capped loop. Every tool call is validated
against a Pydantic schema, with a retry and a fallback to plain search, and
explanations are grounded in the retrieved listing.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                           # add your Groq API key
```

## Usage

```bash
python -m jobmatch.resolve            # find each company's ATS board
python -m jobmatch.ingest             # pull listings into data/jobs.db
python -m jobmatch.index              # embed and build the vector index
python -m jobmatch.query "backend engineer working on search" --compare
python -m jobmatch.label --seed       # hand-label the evaluation queries
python -m jobmatch.evaluate           # Recall@k, MRR, nDCG per mode
pytest
```

Useful flags: `--store qdrant` on `index`/`query`/`evaluate` to use the vector
database instead of exact search, `--mode bm25|dense|hybrid` to isolate one
retriever, and `python -m jobmatch.bench_embed` to measure embedding
throughput on your own machine.

Edit `companies.txt` to change what the index covers — one company name per
line. The resolver tries plausible slugs against each ATS and saves the hits to
`data/slugs.json`. If a company fails but you know its careers URL, paste the
URL into `companies.txt` instead of the name.

If a source parses oddly, dump a raw record and compare:

```bash
python -m jobmatch.inspect_source greenhouse figma
```

## Attribution

Remotive and Arbeitnow both ask that listings taken from their APIs link back
to them. Every stored record keeps its source URL, and the source is shown with
each result.
