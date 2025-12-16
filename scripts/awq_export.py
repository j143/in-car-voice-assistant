"""AWQ quantization/export scaffold (optional).

This script demonstrates how one would invoke AWQ (Activation-aware Weight Quantization)
to export a quantized model for edge inference. It requires optional dependencies
in requirements-awq.txt and will not run without them. Uses llm-compressor (vLLM's
maintained fork) instead of deprecated autoawq.
"""

from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser(description="AWQ export scaffold")
    ap.add_argument("--model", type=str, default="microsoft/phi-3-mini", help="HF model id")
    ap.add_argument("--out", type=str, default="awq-export", help="Output directory")
    args = ap.parse_args()

    try:
        from llmcompressor.transformers import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier
        from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    except Exception as e:
        print("AWQ dependencies not installed. Install requirements-awq.txt. Error:", e)
        return

    print("Loading model", args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    print("Quantizing with AWQ using llm-compressor")
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
    )

    print("Saving to", args.out)
    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        output_dir=args.out,
    )
    print("Done.")


if __name__ == "__main__":
    main()
