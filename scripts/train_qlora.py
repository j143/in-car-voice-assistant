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
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        TrainingArguments, 
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


def load_dataset(path: Path) -> List[Dict]:
    """Robustly load dataset line by line."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"Reading {len(lines)} lines from {path}...")
        for i, line in enumerate(lines):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON at line {i+1}: {e}")
    return data


def format_sample(sample: Dict) -> str:
    """Format sample for instruction tuning."""
    text = sample.get("text", "")
    intent = sample.get("intent", "unknown")
    command = sample.get("command", "unknown")
    return f"User: {text}\nAssistant: Intent={intent}, Command={command}"


class DomainDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 512):
        self.input_ids = []
        self.attention_masks = []
        
        print("Pre-tokenizing dataset for efficiency...")
        for sample in samples:
            formatted = format_sample(sample)
            # Tokenize without padding (dynamic padding handled by collator)
            tokenized = tokenizer(
                formatted, 
                truncation=True, 
                max_length=max_length, 
                padding=False,  # Important: Do not pad here
                return_tensors=None 
            )
            self.input_ids.append(tokenized["input_ids"])
            self.attention_masks.append(tokenized["attention_mask"])
            
        print(f"Tokenized {len(self.input_ids)} samples.")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # FIX: Do not return 'labels' here. 
        # The DataCollator will automatically generate them from input_ids after padding.
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx]
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
    if not train_samples:
        print("Error: No training samples loaded. Check your JSONL file.")
        return
    print(f"Successfully loaded {len(train_samples)} valid samples.")

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, # Changed to float16 for better compatibility
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        # torch_dtype=torch.float16, # Allow auto-detection
    )
    
    # SPEED OPTIMIZATION: Disable gradient checkpointing for small models (Phi-2)
    # Only enable this if you run out of VRAM (OOM error).
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

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

    # Configure LoRA
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
    # Fix potential padding side issues
    tokenizer.padding_side = "right" 

    # Dataset
    train_dataset = DomainDataset(train_samples, tokenizer)
    
    # Data collator handles dynamic padding AND label creation
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  # This ensures labels are created correctly for Causal LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        # SPEED OPTIMIZATION: Use FP16 instead of BF16 unless on Ampere (A100/3090)
        fp16=True,  
        bf16=False, 
        logging_steps=5,
        save_strategy="epoch",
        warmup_steps=10,
        optim="paged_adamw_8bit",
        report_to=["mlflow"] if use_mlflow else ["none"],
        remove_unused_columns=False,
        dataloader_num_workers=2, # Use workers to speed up data fetching
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
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
