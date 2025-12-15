"""Evaluate domain adaptation performance on technical nomenclature.

Metrics:
- Overall accuracy (target: >90%)
- Technical term recognition rate
- Error code recognition rate
- Latency per query (target: <100ms for text path)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

from pipeline.end_to_end import VoiceAssistantPipeline


def load_test_set(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate_domain_performance(test_path: Path, classifier: str = "rule", rag: str = "off") -> Dict:
    pipeline = VoiceAssistantPipeline(use_rag=(rag != "off"), classifier_type=classifier, rag_type=rag)
    test_samples = load_test_set(test_path)
    
    total = len(test_samples)
    correct_intent = 0
    correct_command = 0
    technical_terms_found = 0
    error_codes_found = 0
    latencies = []
    
    for sample in test_samples:
        text = sample["text"]
        expected_intent = sample["intent"]
        expected_command = sample.get("command", "unknown")
        technical_terms = sample.get("technical_terms", [])
        error_codes = sample.get("error_codes", [])
        
        # Measure latency
        t0 = time.perf_counter()
        result = pipeline.process_text(text)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
        # Check intent/command accuracy
        pred_command = result.get("command", "unknown")
        if pred_command == expected_command:
            correct_command += 1
        
        # Check technical term recognition (heuristic: command extraction implies recognition)
        if technical_terms and pred_command != "unknown":
            technical_terms_found += 1
        
        # Check error code recognition
        if error_codes and pred_command == "diagnose_error":
            error_codes_found += 1
    
    avg_latency = statistics.mean(latencies)
    p95_latency = statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 20 else max(latencies)
    
    return {
        "total_samples": total,
        "command_accuracy": round(correct_command / total, 4),
        "technical_term_recognition_rate": round(technical_terms_found / sum(1 for s in test_samples if s.get("technical_terms")) if any(s.get("technical_terms") for s in test_samples) else 0, 4),
        "error_code_recognition_rate": round(error_codes_found / sum(1 for s in test_samples if s.get("error_codes")) if any(s.get("error_codes") for s in test_samples) else 0, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "latency_target_met": avg_latency < 100.0,
        "accuracy_target_met": (correct_command / total) >= 0.90,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate domain adaptation performance")
    ap.add_argument("--test", type=Path, default=Path("data/automotive_domain_test.jsonl"))
    ap.add_argument("--classifier", default="rule", choices=["rule", "svm"])
    ap.add_argument("--rag", default="off", choices=["off", "kb", "faiss"])
    args = ap.parse_args()
    
    results = evaluate_domain_performance(args.test, args.classifier, args.rag)
    print(json.dumps(results, indent=2))
    
    # Print summary
    print("\n=== Performance Summary ===")
    print(f"Command Accuracy: {results['command_accuracy']:.1%} (Target: ≥90%)")
    print(f"Avg Latency: {results['avg_latency_ms']:.2f}ms (Target: <100ms)")
    print(f"Technical Term Recognition: {results['technical_term_recognition_rate']:.1%}")
    print(f"Error Code Recognition: {results['error_code_recognition_rate']:.1%}")
    print(f"\n✓ Accuracy Target Met: {results['accuracy_target_met']}")
    print(f"✓ Latency Target Met: {results['latency_target_met']}")


if __name__ == "__main__":
    main()
