"""Work out which applicant tracking system each company publishes jobs on.

You give it company names; it tries plausible slugs against the Greenhouse,
Lever and Ashby public boards and keeps whatever answers. Saves the result to
data/slugs.json so ingestion never has to guess again.

    python -m jobmatch.resolve                 # read companies.txt
    python -m jobmatch.resolve --only acme     # retry a single name
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

from .config import COMPANIES_FILE, SLUGS_FILE
from .sources import ashby, greenhouse, lever

PROVIDERS = {"greenhouse": greenhouse, "lever": lever, "ashby": ashby}

# If someone pastes a careers URL instead of a name, read the slug straight off it.
URL_PATTERNS = [
    (re.compile(r"(?:job-)?boards\.greenhouse\.io/([a-z0-9_-]+)", re.I), "greenhouse"),
    (re.compile(r"jobs\.(?:eu\.)?lever\.co/([a-z0-9_-]+)", re.I), "lever"),
    (re.compile(r"jobs\.ashbyhq\.com/([a-z0-9_.-]+)", re.I), "ashby"),
]


def candidate_slugs(name: str) -> list[str]:
    """Most boards use a squashed lowercase name; some keep the hyphens."""
    cleaned = re.sub(r"[^a-z0-9 ]", " ", name.lower()).strip()
    words = cleaned.split()
    if not words:
        return []
    squashed = "".join(words)
    hyphened = "-".join(words)
    # Drop a trailing legal suffix: "Acme Technologies Pvt Ltd" -> "acme"
    trimmed = [w for w in words if w not in {"inc", "ltd", "llc", "pvt", "limited",
                                             "technologies", "technology", "labs",
                                             "software", "systems"}]
    cands = [squashed, hyphened]
    if trimmed and trimmed != words:
        cands += ["".join(trimmed), "-".join(trimmed)]
    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def resolve_one(name: str) -> dict | None:
    """Try each provider until one returns jobs. Returns None if nothing matched."""
    for pattern, provider in URL_PATTERNS:
        m = pattern.search(name)
        if m:
            slug = m.group(1)
            count = PROVIDERS[provider].probe(slug)
            if count:
                return {"name": name, "provider": provider, "slug": slug, "jobs": count}
            return None

    for slug in candidate_slugs(name):
        for provider, module in PROVIDERS.items():
            count = module.probe(slug)
            if count:
                return {"name": name, "provider": provider, "slug": slug, "jobs": count}
    return None


def load_names(path=COMPANIES_FILE) -> list[str]:
    if not path.exists():
        raise SystemExit(
            f"No {path.name} found. Create it with one company name per line."
        )
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if l and not l.startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser(description="Find each company's ATS job board.")
    ap.add_argument("--only", help="resolve a single company name instead of the file")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    names = [args.only] if args.only else load_names()
    print(f"Probing {len(names)} companies across {len(PROVIDERS)} boards...\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(resolve_one, names))

    found = [r for r in results if r]
    missing = [n for n, r in zip(names, results) if not r]

    for r in sorted(found, key=lambda x: -x["jobs"]):
        print(f"  {r['jobs']:>5} jobs  {r['provider']:<11} {r['slug']:<24} ({r['name']})")

    if missing:
        print(f"\nNo public board found for {len(missing)}:")
        print("  " + ", ".join(missing))
        print("\n  These may use Workday/Keka/a custom site, or have a different slug.")
        print("  Paste the careers URL into companies.txt instead of the name to force it.")

    SLUGS_FILE.write_text(json.dumps(found, indent=2), encoding="utf-8")
    total = sum(r["jobs"] for r in found)
    print(f"\nResolved {len(found)}/{len(names)} companies, ~{total} jobs available.")
    print(f"Saved to {SLUGS_FILE}")


if __name__ == "__main__":
    main()
