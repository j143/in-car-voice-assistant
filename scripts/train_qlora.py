"""QLoRA fine-tuning pipeline for domain adaptation on automotive technical nomenclature.

Target: Adapt base LM to recognize error codes (P0420, etc.) and part numbers (K 300 503 v17).
Performance goal: >90% accuracy on technical term classification.
"""

from __future__ import annotations

import argparse
import json
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
    """Format sample for instruction tuning."""
    text = sample["text"]
    intent = sample["intent"]
    command = sample.get("command", "unknown")
    return f"User: {text}\nAssistant: Intent={intent}, Command={command}"


class DomainDataset:
    def __init__(self, samples: List[Dict], tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        formatted = format_sample(self.samples[idx])
        tokenized = self.tokenizer(
            formatted, 
            truncation=True, 
            max_length=512, 
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
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 4,
    lora_rank: int = 16,
    use_mlflow: bool = True,
):
    if not DEPS_AVAILABLE:
        print("Training dependencies not available. Install requirements-train.txt")
        return

    # Load dataset
    train_samples = load_dataset(train_path)
    print(f"Loaded {len(train_samples)} training samples")

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
    train_dataset = DomainDataset(train_samples, tokenizer)

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
    ap.add_argument("--train", type=Path, default=Path("data/automotive_domain_train.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("checkpoints/automotive-adapter"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--mlflow", action="store_true", help="Enable MLflow logging via report_to")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    train_qlora(
        model_name=args.model,
        train_path=args.train,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        use_mlflow=args.mlflow,
    )


if __name__ == "__main__":
    main()
