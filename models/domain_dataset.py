"""Domain-specific dataset for automotive NLP fine-tuning.

Wraps a list of JSONL samples into a PyTorch Dataset suitable for
instruction-tuning with a causal language model.
"""

from __future__ import annotations

from typing import Dict, List, Any


def format_sample(sample: Dict[str, Any]) -> str:
    """Format a JSONL sample as an instruction-tuning string."""
    text = sample.get("text", "")
    intent = sample.get("intent", "unknown")
    command = sample.get("command", "unknown")
    return f"User: {text}\nAssistant: Intent={intent}, Command={command}"


class DomainDataset:
    """PyTorch-compatible dataset for automotive domain samples.

    Args:
        samples: List of dicts loaded from JSONL.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum token length per sample.
    """

    def __init__(self, samples: List[Dict[str, Any]], tokenizer, max_length: int = 512) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        formatted = format_sample(self.samples[idx])
        tokenized = self.tokenizer(
            formatted,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": input_ids.clone(),  # Labels == input_ids for causal LM training
        }
