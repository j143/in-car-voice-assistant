"""Run experiments across configurations (classifier, RAG) and collect results.

Outputs JSON with latency stats and evaluation metrics for a given dataset.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add parent directory to path so pipeline module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.end_to_end import VoiceAssistantPipeline
from scripts.evaluate_intent import load_lines, evaluate


def bench_latency(p: VoiceAssistantPipeline, text: str, runs: int) -> Dict:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = p.process_text(text)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    avg = statistics.mean(times)
    p95 = statistics.quantiles(times, n=100)[94] if len(times) >= 20 else max(times)
    return {"avg_ms": round(avg, 2), "p95_ms": round(p95, 2), "samples": [round(x, 2) for x in times]}


def run_experiment(dataset_path: Path, sample_text: str, classifier: str, rag: str, runs: int) -> Dict:
    p = VoiceAssistantPipeline(use_rag=(rag != "off"), classifier_type=classifier, rag_type=(rag if rag != "off" else "kb"))
    latency = bench_latency(p, sample_text, runs)
    eval_res = evaluate(load_lines(dataset_path), classifier=classifier, rag=rag)
    return {"classifier": classifier, "rag": rag, "latency": latency, "evaluation": eval_res}


def main():
    ap = argparse.ArgumentParser(description="Run experiments across configurations")
    ap.add_argument("dataset", type=Path, help="Path to JSONL/CSV dataset")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--text", type=str, default="Set temperature to 72")
    args = ap.parse_args()

    configs = [(c, r) for c in ["rule", "svm"] for r in ["off", "kb", "faiss"]]
    results: List[Dict] = []
    for classifier, rag in configs:
        try:
            res = run_experiment(args.dataset, args.text, classifier, rag, args.runs)
            results.append(res)
        except Exception as e:
            results.append({"classifier": classifier, "rag": rag, "error": str(e)})

    print(json.dumps({"dataset": str(args.dataset), "results": results}, indent=2))


if __name__ == "__main__":
    main()
