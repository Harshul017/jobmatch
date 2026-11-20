# JobMatch

Search engine and LLM agent over ~7,000 live job listings, pulled from public
job-board APIs. Combines keyword search with embeddings, and measures whether
the combination actually helps rather than assuming it does.

Work in progress. Retrieval, the agent, the API and the UI all work; the
relevance judgements behind the evaluation numbers are not yet verified
against a human, so no retrieval metric is quoted here yet.

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
`--mode bm25|dense|hybrid`, `--diverse 2` (cap results per company),
`--company`, `--store qdrant`.

```bash
python -m jobmatch.chat          # talk to the agent
uvicorn jobmatch.api:app         # web UI and API on localhost:8000
```

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

It also has to recognise what *isn't* a place. `In-Office` is a work
arrangement, not a city, and collapsing it naively produced `in` — which
collides with India's country code, so roles in Austin and London matched a
filter for India. The mirror image is `IN-Bengaluru`, where `IN-` is a country
prefix and the city follows; stripping at the hyphen threw the city away.
Between them those accounted for 669 wrong locations.

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
apply to both retrievers. Mid-level roles are stored with a NULL seniority, so
excluding "senior" has to keep NULL explicitly — `NOT IN` never matches NULL,
and a naive exclusion silently drops most of the index.

Titles under-report seniority. A role called "Software Engineer" that asks for
6-8 years is senior in everything but its name, and 693 listings here (9.7%)
are exactly that: no marker in the title, five or more years in the
description. So the years requirement is parsed out of the body text and
filtered on separately — "nothing senior" means both an excluded title level
and a ceiling on years. 59% of listings state a figure; the mean is 5.6 years.

Relevance ranking has no notion of variety, so one large employer with many
similar postings fills the page. `--diverse N` caps how many results one
company can take. The cap is soft: capped entries are demoted rather than
dropped, so the page still fills when few employers match. It's off for
evaluation and on for the agent, which is showing a shortlist.

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

## Agent

```bash
python -m jobmatch.chat
python -m jobmatch.chat --once "backend roles in bengaluru, nothing senior" --verbose
```

An open-weights model (gpt-oss on Groq) drives the conversation through three
tools: `search_jobs`, `get_job` and `save_preference`. Ranking stays in the
retrieval code where it can be measured. The model decides what to look for
and explains the results; it does not decide what is relevant.

Small models misbehave in predictable ways, so the loop assumes they will:

- Arguments are validated against a Pydantic schema. A bad type or a malformed
  JSON blob comes back to the model as a readable error it can correct, rather
  than raising.
- Every job mentioned must be cited by id. Ids that no tool returned get
  stripped from the reply, so an invented listing never reaches the person.
- The loop is capped. If the model keeps calling tools without answering, or
  answers with nothing, the agent runs the search itself and returns those
  results.
- Preferences persist in SQLite and are injected into the system prompt and
  applied as search filters on later turns. If filters match nothing, the
  search relaxes them and says so rather than returning an empty list.

Swapping the model is a flag; swapping the provider is one class in `llm.py`.

## API

```bash
uvicorn jobmatch.api:app --reload
curl "localhost:8000/search?q=backend+engineer&location=india&exclude_seniority=senior"
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
     -d '{"message": "remote ML roles in india"}'
```

`/health` is readiness, not liveness: an instance with an empty index is up
but cannot answer anything, so it returns 503 rather than looking healthy to
a load balancer.

A single-page UI is served at `/`, and interactive API docs at `/docs`.

## Deploying

The working index is ~122MB, nearly all of it full job descriptions that the
app never shows. `prepare_deploy` writes a trimmed copy — descriptions cut to
what the API actually returns, expired rows dropped, vacuumed, with matching
vectors.

```bash
python -m jobmatch.prepare_deploy --engineering-only --max-jobs 3000
docker build -t jobmatch .
docker run -p 8000:8000 --env-file .env jobmatch
```

The slim index is baked into the image, since most free hosting tiers have no
persistent disk. To run against the full local index instead, skip the `COPY`
and mount it: `-v $(pwd)/data:/app/data`.

The only secret is `GROQ_API_KEY`. Search works without it; the chat endpoint
returns 503 and says why.

`render.yaml` deploys it on Render's free tier: connect the repo, set
`GROQ_API_KEY` in the dashboard, done. The slim index is committed so the
image can be built from the repo alone. The free tier sleeps after about 15
minutes idle, so a first request after a quiet spell takes a while to wake.

## Evaluation

```bash
python -m jobmatch.label --seed    # judge pooled results by hand
python -m jobmatch.evaluate        # Recall@k, P@5, MRR, nDCG@10 per mode
```

Judging only what one retriever returns would decide the comparison in
advance, so candidates are pooled: the union of what each mode returns,
shuffled before display. Anything outside the pool counts as non-relevant,
which is the usual limitation of pooled judgements.

```bash
python -m jobmatch.autolabel             # a model judges the pool
python -m jobmatch.autolabel --verify 40 # spot-check it yourself
python -m jobmatch.autolabel --report
```

Hand-labelling is the gold standard and `label.py` does that. `autolabel.py`
is the fallback: a model judges the pooled candidates and a human spot-checks
a random sample, so the agreement rate is measured rather than assumed. The
figure is written into `data/eval.json` and printed by `evaluate` alongside
any metric computed from those labels, because a retrieval number is only
worth what its judgements are worth.

Retrieval metrics say nothing about whether the agent called the right tool,
honoured a constraint, or made a job up, so the agent is scored separately:

```bash
python -m jobmatch.eval_agent
```

Each scenario asserts behaviour that has to hold however the model words its
answer — searched before answering, cited only retrieved jobs, returned no
senior roles when asked for none, saved a preference only when told one. It
reports grounding rate, fallback rate, invalid tool calls and tool calls per
turn. Unlike the unit tests this calls the real model, so it costs requests.

## Tests

```bash
pytest
```

166 tests, run on 3.11 and 3.12 in CI. Vector stores are tested against both
backends with the same suite. Embedding is stubbed with a hashing embedder and the LLM with a scripted one,
so the tests need no model download, no API key and no network. The agent
tests mostly cover failure: malformed arguments, unknown tools, invented
citations, runaway tool loops.

## Known gaps

- **The evaluation labels are model-judged and unverified.** `autolabel
  --verify` exists for exactly this; until it has been run, the retrieval
  comparison is not trustworthy. An earlier run showed why: the three
  retrievers ranked in a completely different order once the judge was shown
  the role text instead of the company boilerplate at the top of each
  posting. The code did not change. The judgements did.
- Seniority is single-label, so "Senior Engineering Manager" is classified as
  management and escapes a filter that excludes senior. It should be
  multi-label: a role can be both.
- Some aggregator feeds carry agency names rather than employers, so a
  handful of listings show a recruiter where the company should be.
- Not deployed yet. `render.yaml` is ready.

## Still to do

- Reranker over the fused top-k
- Scheduled re-ingestion so the index refreshes without being asked
