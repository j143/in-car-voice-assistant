"""QLoRA fine-tuning pipeline for domain adaptation on automotive technical nomenclature.

Target: Adapt base LM to recognize error codes (P0420, etc.) and part numbers (K 300 503 v17).
Performance goal: >90% accuracy on technical term classification.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


def load_dataset(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def format_sample(sample: Dict) -> str:
    """Format sample for instruction tuning with rich domain context + parameters.

    Includes a structured Parameters field so the model learns to extract
    entities like error codes and technical terms (slot filling).
    """
    text = sample["text"]
    intent = sample["intent"]
    command = sample.get("command", "unknown")
    
    # Include rich metadata if available (from Bosch schema)
    technical_terms = sample.get("technical_terms", [])
    error_codes = sample.get("error_codes", [])
    subsystem = sample.get("subsystem", "")
    severity = sample.get("severity", "")
    expected = sample.get("expected", "")
    
    # Build formatted prompt with domain context
    formatted = f"User: {text}\n"
    
    if technical_terms:
        terms_str = ", ".join(technical_terms)
        formatted += f"Technical Terms: {terms_str}\n"
    
    if error_codes:
        codes_str = ", ".join(error_codes)
        formatted += f"Error Codes: {codes_str}\n"
    
    # Parameters for slot filling
    parameters: Dict[str, List[str]] = {}
    if error_codes:
        parameters["codes"] = error_codes
    if technical_terms:
        parameters["terms"] = technical_terms

    formatted += f"Assistant: Intent={intent}, Command={command}, Parameters={json.dumps(parameters, ensure_ascii=False)}"
    
    if subsystem:
        formatted += f", Subsystem={subsystem}"
    if severity:
        formatted += f", Severity={severity}"
    
    formatted += "\n"
    
    if expected:
        formatted += f"Expected: {expected}"
    
    return formatted
def split_dataset(samples: List[Dict], val_ratio: float = 0.2, seed: int = 42):
    """Deterministically shuffle and split samples into train/test lists."""
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    cut = int(len(indices) * (1 - val_ratio))
    train_idx = indices[:cut]
    test_idx = indices[cut:]
    train = [samples[i] for i in train_idx]
    test = [samples[i] for i in test_idx]
    return train, test


class DomainDataset:
    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        formatted = format_sample(self.samples[idx])
        tokenized = self.tokenizer(
            formatted, 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length",
            return_tensors="pt"
        )
        # Return as dict without extra nesting
        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": tokenized["input_ids"].squeeze(),
        }


def train_qlora(
    model_name: str,
    train_path: Path,
    test_path: Path | None,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 4,
    lora_rank: int = 16,
    use_mlflow: bool = True,
    max_length: int = 256,
    val_ratio: float = 0.2,
    create_split: bool = True,
    split_output_dir: Path | None = None,
):
    if not DEPS_AVAILABLE:
        print("Training dependencies not available. Install requirements-train.txt")
        return

    # Load dataset
    all_samples = load_dataset(train_path)
    print(f"Loaded {len(all_samples)} samples from {train_path}")

    # Train/Test split handling
    if test_path and test_path.exists():
        print(f"Using provided test set: {test_path}")
        test_samples = load_dataset(test_path)
        train_samples = all_samples
    else:
        if create_split:
            train_samples, test_samples = split_dataset(all_samples, val_ratio=val_ratio)
            print(f"Split: train={len(train_samples)}, test={len(test_samples)} (val_ratio={val_ratio})")
            # Optionally persist split for reproducibility
            if split_output_dir:
                split_output_dir.mkdir(parents=True, exist_ok=True)
                train_out = split_output_dir / (train_path.stem + ".train.jsonl")
                test_out = split_output_dir / (train_path.stem + ".test.jsonl")
                train_out.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in train_samples))
                test_out.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in test_samples))
                print(f"Saved split: {train_out} | {test_out}")
        else:
            # No test set provided and no split requested
            train_samples = all_samples
            test_samples = []

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    # Disable cache when using gradient checkpointing to save memory
    if hasattr(model, "config"):
        model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Phi-2 uses explicit q/k/v projections plus dense + MLP (fc1/fc2). Hard-code to avoid mismatches.
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "dense",  # attention output
        "fc1",
        "fc2",
    ]
    print(f"LoRA target modules (phi-2): {target_modules}")

    # Configure LoRA using detected modules
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    train_dataset = DomainDataset(train_samples, tokenizer, max_length=max_length)
    eval_dataset = DomainDataset(test_samples, tokenizer, max_length=max_length) if test_samples else None

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,  # Use bf16 instead for better stability
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_dataset is not None else "no",
        warmup_steps=10,
        optim="paged_adamw_8bit",
        report_to=["mlflow"] if use_mlflow else ["none"],
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting training...")
    trainer.train()
    
    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved adapter to {output_dir}")


def main():
    ap = argparse.ArgumentParser(description="QLoRA fine-tuning for automotive domain")
    ap.add_argument("--model", default="microsoft/phi-2", help="Base model name")
    ap.add_argument("--train", type=Path, default=Path("data/automotive_domain_train_10k.jsonl"))
    ap.add_argument("--test", type=Path, default=None, help="Optional test set. If not provided, will split from train.")
    ap.add_argument("--output", type=Path, default=Path("checkpoints/automotive-adapter"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--mlflow", action="store_true", help="Enable MLflow logging via report_to")
    ap.add_argument("--max-length", type=int, default=256, help="Max sequence length (reduce to lower memory)")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio when creating split")
    ap.add_argument("--no-split", action="store_true", help="Do not auto-split; use entire train set")
    ap.add_argument("--split-out", type=Path, default=Path("data/splits"), help="Directory to save train/test split files")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    train_qlora(
        model_name=args.model,
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        use_mlflow=args.mlflow,
        max_length=args.max_length,
        val_ratio=args.val_ratio,
        create_split=not args.no_split,
        split_output_dir=args.split_out,
    )


if __name__ == "__main__":
    main()
