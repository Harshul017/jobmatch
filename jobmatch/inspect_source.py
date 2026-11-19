"""Dump one raw record from a source so we can check the parser against reality.

Field names on these public feeds change without notice. If ingestion returns
empty descriptions or odd titles, run this and read what the API actually sent.

    python -m jobmatch.inspect_source greenhouse figma
    python -m jobmatch.inspect_source remotive
"""
import argparse
import json

from .sources import arbeitnow, ashby, greenhouse, lever, remotive
from .sources.base import get_json

ENDPOINTS = {
    "greenhouse": lambda s: (f"{greenhouse.BASE}/{s}/jobs", {"content": "true"}),
    "lever": lambda s: (f"{lever.BASE}/{s}", {"mode": "json"}),
    "ashby": lambda s: (f"{ashby.BASE}/{s}", {}),
    "remotive": lambda s: (remotive.URL, {"limit": 2}),
    "arbeitnow": lambda s: (arbeitnow.URL, {"page": 1}),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Show a raw record from one source.")
    ap.add_argument("provider", choices=sorted(ENDPOINTS))
    ap.add_argument("slug", nargs="?", default="", help="company slug (ATS sources)")
    ap.add_argument("--full", action="store_true", help="print the whole record")
    args = ap.parse_args()

    url, params = ENDPOINTS[args.provider](args.slug)
    data = get_json(url, params=params)

    rows = data if isinstance(data, list) else (
        data.get("jobs") or data.get("data") or []
    )
    print(f"{len(rows)} records from {url}\n")
    if not rows:
        print(json.dumps(data, indent=2)[:2000])
        return

    sample = rows[0]
    print("top-level keys:", ", ".join(sorted(sample)), "\n")
    if args.full:
        print(json.dumps(sample, indent=2)[:6000])
    else:
        for key, value in sorted(sample.items()):
            preview = json.dumps(value)[:140] if not isinstance(value, str) else repr(value[:140])
            print(f"  {key:<24} {preview}")


if __name__ == "__main__":
    main()
