"""LoRA/QLoRA fine-tuning wrapper (scaffold).

Optional module to prepare a small LM for domain adaptation with LoRA/QLoRA.
Imports are lazy to keep base environment light.
"""

from __future__ import annotations

from typing import Optional


class LoRAFinetuner:
    def __init__(self, model_name: str = "microsoft/phi-3-mini", lora_rank: int = 16) -> None:
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.model = None
        self.peft_model = None

    def load_model(self) -> None:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig  # type: ignore
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        from peft import prepare_model_for_kbit_training  # type: ignore
        self.model = prepare_model_for_kbit_training(self.model)

    def setup_peft(self) -> None:
        from peft import LoraConfig, get_peft_model  # type: ignore
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.peft_model = get_peft_model(self.model, lora_config)
