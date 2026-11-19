"""Project-wide paths and settings."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "jobs.db"
QDRANT_PATH = DATA_DIR / "qdrant"
COMPANIES_FILE = ROOT / "companies.txt"
SLUGS_FILE = DATA_DIR / "slugs.json"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "")

# Be polite to the public APIs we read.
HTTP_TIMEOUT = 20.0
REQUEST_DELAY = 0.3  # seconds between calls to the same host

DATA_DIR.mkdir(exist_ok=True)
