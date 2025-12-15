"""PII cleaning for service logs with technical nomenclature preservation.

Replaces emails, phone numbers, and obvious IDs with placeholders, while leaving
technical tokens (e.g., 'K 300 503 v17') unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}")
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b")
VIN_RE = re.compile(r"\b[0-9A-HJ-NPR-Za-hj-npr-z]{17}\b")

# Heuristic: preserve patterns like 'K 300 503 v17' (letters + spaced digits + version)
TECH_TOKEN_RE = re.compile(r"\b[A-Za-z](?:\s*\d{2,4}){1,3}\s*v\d{1,3}\b")


def anonymize(text: str) -> str:
    # Skip anonymization for strings that match clear technical token pattern
    if TECH_TOKEN_RE.search(text):
        return text
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = PHONE_RE.sub("<PHONE>", text)
    text = UUID_RE.sub("<UUID>", text)
    text = VIN_RE.sub("<VIN>", text)
    return text


def process_file(inp: Path, out: Path) -> None:
    with inp.open() as f_in, out.open("w") as f_out:
        for line in f_in:
            line = line.rstrip("\n")
            if not line:
                f_out.write("\n")
                continue
            try:
                obj = json.loads(line)
                for k, v in list(obj.items()):
                    if isinstance(v, str):
                        obj[k] = anonymize(v)
                f_out.write(json.dumps(obj) + "\n")
            except json.JSONDecodeError:
                f_out.write(anonymize(line) + "\n")


def main():
    ap = argparse.ArgumentParser(description="PII cleaner for logs")
    ap.add_argument("input", type=Path, help="Input JSONL or text file")
    ap.add_argument("output", type=Path, help="Output file path")
    args = ap.parse_args()
    process_file(args.input, args.output)


if __name__ == "__main__":
    main()
