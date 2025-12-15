"""Convert a plain text file of utterances into a JSONL with labels.

Labels are guessed using the rule-based classifier; you can edit the output manually.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.end_to_end import VoiceAssistantPipeline


def main():
    ap = argparse.ArgumentParser(description="Label a text dataset into JSONL")
    ap.add_argument("input", type=Path, help="Path to text file with one utterance per line")
    ap.add_argument("output", type=Path, help="Output JSONL path")
    args = ap.parse_args()

    p = VoiceAssistantPipeline(use_rag=False, classifier_type="rule")
    lines = [l.strip() for l in args.input.read_text().splitlines() if l.strip()]

    with args.output.open("w") as f:
        for text in lines:
            out = p.process_text(text)
            label = out.get("command", "unknown")
            f.write(json.dumps({"text": text, "label": label}) + "\n")

    print(f"Wrote {len(lines)} labeled samples to {args.output}")


if __name__ == "__main__":
    main()
