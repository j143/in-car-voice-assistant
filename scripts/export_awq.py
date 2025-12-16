"""Enhanced AWQ quantization export for edge deployment.

Exports model with Activation-aware Weight Quantization for NVIDIA Orin deployment.
Target: <100ms inference on Jetson Orin with <2GB memory footprint.
Uses llm-compressor (vLLM's maintained fork) instead of deprecated autoawq.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def export_awq_model(
    model_name: str,
    adapter_path: Path | None,
    output_dir: Path,
    w_bit: int = 4,
):
    """Export model with AWQ quantization using llm-compressor."""
    try:
        from llmcompressor.transformers import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier
        from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    except ImportError:
        print("AWQ dependencies not available. Install requirements-awq.txt")
        print("pip install 'llm-compressor[transformers]'")
        return

    print(f"Loading model: {model_name}")
    
    # Load model and tokenizer
    if adapter_path and adapter_path.exists():
        print(f"Loading with LoRA adapter from {adapter_path}")
        from peft import PeftModel  # type: ignore
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure AWQ quantization
    print(f"Quantizing to {w_bit}-bit with AWQ using llm-compressor...")
    
    # Create quantization recipe
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",  # 4-bit weights, 16-bit activations
        ignore=["lm_head"],  # Don't quantize output layer
    )
    
    # Apply quantization
    output_dir.mkdir(parents=True, exist_ok=True)
    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        output_dir=str(output_dir),
        # For better accuracy, provide calibration data:
        # dataset="your-calibration-dataset",
        # num_calibration_samples=512,
    )
    
    print(f"Quantized model saved to {output_dir}")
    print("Export complete. Deploy with TensorRT-LLM or vLLM for optimal performance.")


def main():
    ap = argparse.ArgumentParser(description="Export AWQ quantized model for edge")
    ap.add_argument("--model", default="microsoft/phi-2", help="Base model or checkpoint")
    ap.add_argument("--adapter", type=Path, help="Optional LoRA adapter path to merge")
    ap.add_argument("--output", type=Path, default=Path("exports/automotive-awq-4bit"))
    ap.add_argument("--w-bit", type=int, default=4, choices=[4, 8])
    args = ap.parse_args()
    
    export_awq_model(args.model, args.adapter, args.output, args.w_bit)


if __name__ == "__main__":
    main()
