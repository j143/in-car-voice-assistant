"""Evaluate intent/command accuracy and OOD rate on a labeled dataset.

Input: JSONL or CSV with fields: text, label
- text: user utterance
- label: expected command string (e.g., set_temperature, open_window) or 'ood'

Outputs accuracy and simple confusion counts.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from pipeline.end_to_end import VoiceAssistantPipeline


def load_lines(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if path.suffix.lower() == ".csv":
        with path.open() as f:
            reader = csv.DictReader(f)
            return list(reader)
    raise ValueError("Unsupported file format; use .jsonl or .csv")


def evaluate(dataset: List[Dict], classifier: str, rag: str) -> Dict:
    p = VoiceAssistantPipeline(use_rag=(rag != "off"), classifier_type=classifier, rag_type=(rag if rag != "off" else "kb"))
    total = 0
    correct = 0
    confusion = Counter()
    ood_true = 0
    ood_pred = 0

    for row in dataset:
        text = str(row.get("text", ""))
        label = str(row.get("label", "unknown")).lower().strip()
        out = p.process_text(text)
        pred = str(out.get("command", "unknown")).lower()
        total += 1
        if pred == label:
            correct += 1
        confusion[(label, pred)] += 1
        if label == "ood":
            ood_true += 1
        if pred == "unknown" or pred == "ood":
            ood_pred += 1

    acc = correct / total if total else 0.0
    ood_rate = ood_pred / total if total else 0.0
    return {
        "total": total,
        "accuracy": round(acc, 4),
        "ood_rate": round(ood_rate, 4),
        "top_confusions": confusion.most_common(10),
    }


essential_commands = [
    "set_temperature",
    "open_window",
    "close_window",
    "lock_doors",
    "unlock_doors",
    "turn_on_lights",
    "turn_off_lights",
    "play_music",
    "stop_music",
    "navigate",
    "call",
    "unknown",
    "ood",
]


def main():
    ap = argparse.ArgumentParser(description="Evaluate command accuracy and OOD")
    ap.add_argument("dataset", type=Path, help="Path to JSONL/CSV dataset")
    ap.add_argument("--classifier", choices=["rule", "svm"], default="rule")
    ap.add_argument("--rag", choices=["kb", "faiss", "off"], default="off")
    args = ap.parse_args()

    data = load_lines(args.dataset)
    res = evaluate(data, classifier=args.classifier, rag=args.rag)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
