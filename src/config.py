"""Configuration — env loading, paths, model settings."""

from pathlib import Path
from dotenv import load_dotenv

# Call once at import time; safe to call multiple times.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

