"""Latency benchmark for the in-car assistant (text path).
+
+Measures average latency over multiple runs to guide optimization work.
+"""
+
+from __future__ import annotations
+
+import argparse
+import statistics
+import time
+
+from pipeline.end_to_end import VoiceAssistantPipeline
+
+
+def run_once(pipeline: VoiceAssistantPipeline, text: str) -> float:
+    t0 = time.perf_counter()
+    _ = pipeline.process_text(text)
+    t1 = time.perf_counter()
+    return (t1 - t0) * 1000.0
+
+
+def main() -> None:
+    parser = argparse.ArgumentParser(description="Benchmark latency for text path")
+    parser.add_argument("--text", type=str, default="Set temperature to 72", help="Sample utterance")
+    parser.add_argument("--runs", type=int, default=10, help="Number of runs")
+    args = parser.parse_args()
+
+    pipeline = VoiceAssistantPipeline(use_rag=True)
+    latencies = [run_once(pipeline, args.text) for _ in range(args.runs)]
+    avg = statistics.mean(latencies)
+    p95 = statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 20 else max(latencies)
+    print({
+        "runs": args.runs,
+        "avg_ms": round(avg, 2),
+        "p95_ms": round(p95, 2),
+        "samples": [round(x, 2) for x in latencies],
+    })
+
+
+if __name__ == "__main__":
+    main()
