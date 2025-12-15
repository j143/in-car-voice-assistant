"""AWQ quantization/export scaffold (optional).

This script demonstrates how one would invoke AWQ (Activation-aware Weight Quantization)
to export a quantized model for edge inference. It requires optional dependencies
in requirements-awq.txt and will not run without them.
"""

from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser(description="AWQ export scaffold")
    ap.add_argument("--model", type=str, default="microsoft/phi-3-mini", help="HF model id")
    ap.add_argument("--out", type=str, default="awq-export", help="Output directory")
    args = ap.parse_args()

    try:
        from awq import AutoAWQForCausalLM  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:
        print("AWQ dependencies not installed. Install requirements-awq.txt. Error:", e)
        return

    print("Loading model", args.model)
    model = AutoAWQForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    print("Quantizing with AWQ (defaults)")
    # Example: model.quantize(...) depending on the specific API version.
    # Placeholder: actual parameters depend on AutoAWQ version and GPU availability.

    print("Saving to", args.out)
    model.save_quantized(args.out)
    tokenizer.save_pretrained(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
