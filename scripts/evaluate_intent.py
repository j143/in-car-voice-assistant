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
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

# Add parent directory to path so pipeline module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.end_to_end import VoiceAssistantPipeline
from collections import defaultdict


def load_lines(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if path.suffix.lower() == ".csv":
        with path.open() as f:
            reader = csv.DictReader(f)
            return list(reader)
    raise ValueError("Unsupported file format; use .jsonl or .csv")


def evaluate(dataset: List[Dict], classifier: str, rag: str, adapter_path: Optional[str] = None) -> Dict:
    p = VoiceAssistantPipeline(
        use_rag=(rag != "off"), 
        classifier_type=classifier, 
        rag_type=(rag if rag != "off" else "kb"),
        adapter_path=adapter_path
    )
    total = 0
    correct = 0
    confusion = Counter()
    per_class_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    ood_true = 0
    ood_pred = 0
    confidences = []

    # Parameter extraction counters
    total_with_codes = 0
    matched_codes = 0
    total_with_terms = 0
    matched_terms = 0

    for row in dataset:
        text = str(row.get("text", ""))
        label = str(row.get("label", "unknown")).lower().strip()
        out = p.process_text(text)
        pred = str(out.get("command", "unknown")).lower()
        total += 1
        if pred == label:
            correct += 1
            per_class_counts[label]["tp"] += 1
        else:
            per_class_counts[pred]["fp"] += 1
            per_class_counts[label]["fn"] += 1
        confusion[(label, pred)] += 1
        if label == "ood":
            ood_true += 1
        if pred == "unknown" or pred == "ood":
            ood_pred += 1
        confidences.append({"label": label, "pred": pred, "score": float(out.get("confidence", 0.0) or 0.0)})

        # Parameter extraction checks (if dataset provides ground truth)
        gt_codes = row.get("error_codes") or []
        gt_terms = row.get("technical_terms") or []
        pred_params = out.get("parameters", {}) or {}
        pred_codes = pred_params.get("codes") or []
        pred_terms = pred_params.get("terms") or []

        if gt_codes:
            total_with_codes += 1
            # consider a match if any code overlaps
            if any(c in pred_codes for c in gt_codes):
                matched_codes += 1
        if gt_terms:
            total_with_terms += 1
            if any(t in pred_terms for t in gt_terms):
                matched_terms += 1

    acc = correct / total if total else 0.0
    ood_rate = ood_pred / total if total else 0.0
    # Per-class precision/recall
    per_class = {}
    for cls, m in per_class_counts.items():
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        per_class[cls] = {"precision": round(prec, 4), "recall": round(rec, 4)}

    # Simple OOD AUROC estimate: treat confidence as score, label ood vs non-ood
    try:
        import numpy as np
        from sklearn.metrics import roc_auc_score
        y_true = np.array([1 if c["label"] == "ood" else 0 for c in confidences])
        scores = np.array([1.0 - c["score"] for c in confidences])  # lower confidence => more OOD
        ood_auroc = float(roc_auc_score(y_true, scores)) if y_true.sum() and (len(y_true) - y_true.sum()) else 0.0
    except Exception:
        ood_auroc = 0.0

    return {
        "total": total,
        "accuracy": round(acc, 4),
        "ood_rate": round(ood_rate, 4),
        "ood_auroc": round(ood_auroc, 4),
        "per_class": per_class,
        "top_confusions": confusion.most_common(10),
        "parameters": {
            "codes_coverage": total_with_codes,
            "codes_match_rate": round((matched_codes / total_with_codes) if total_with_codes else 0.0, 4),
            "terms_coverage": total_with_terms,
            "terms_match_rate": round((matched_terms / total_with_terms) if total_with_terms else 0.0, 4),
        },
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
    ap.add_argument("--adapter", type=Path, default=None, help="Path to LoRA adapter checkpoint")
    args = ap.parse_args()

    data = load_lines(args.dataset)
    res = evaluate(data, classifier=args.classifier, rag=args.rag, adapter_path=str(args.adapter) if args.adapter else None)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
