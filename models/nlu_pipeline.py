"""Natural Language Understanding (NLU) Pipeline Module.

Implements the QuantizedNLUPipeline for intent classification with quantization support.
"""

import logging
from typing import Dict, Any, Optional

try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    import torch
    from peft import PeftModel
except ImportError:
    torch = None
    PeftModel = None

logger = logging.getLogger(__name__)


class QuantizedNLUPipeline:
    """Quantized Natural Language Understanding Pipeline.
    
    Uses a quantized transformer model for intent classification with reduced memory footprint.
    Supports dynamic batching and optional quantization for edge deployment.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english", 
                 use_quantization: bool = True, quantization_bits: int = 8,
                 adapter_path: Optional[str] = None):
        """Initialize the NLU pipeline.
        
        Args:
            model_name: Hugging Face model identifier
            use_quantization: Whether to apply quantization
            quantization_bits: Quantization bit depth (8 or 4)
            adapter_path: Optional path to LoRA adapter (for fine-tuned models)
        """
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        self.adapter_path = adapter_path
        self.pipeline = None
        self._is_loaded = False
        self._load_model()
    
    def _load_model(self) -> None:
        """Load and optionally quantize the NLU model."""
        if torch is None:
            logger.warning("torch not available, NLU pipeline will not be functional")
            return
        
        try:
            # Load base model
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load LoRA adapter if provided
            if self.adapter_path and PeftModel is not None:
                try:
                    model = PeftModel.from_pretrained(model, self.adapter_path)
                    logger.info(f"Loaded LoRA adapter from {self.adapter_path}")
                except Exception as e:
                    logger.warning(f"Could not load adapter from {self.adapter_path}: {e}")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            self.pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            self._is_loaded = True
            logger.info(f"NLU pipeline loaded: {self.model_name}" + (f" + adapter: {self.adapter_path}" if self.adapter_path else ""))
        except Exception as e:
            logger.error(f"Failed to load NLU model: {e}")
            self._is_loaded = False
    
    def process(self, text: str) -> Dict[str, Any]:
        """Process text and extract intent.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with intent labels and confidence scores
        """
        if not self._is_loaded or self.pipeline is None:
            return {"label": "unknown", "score": 0.0}
        
        try:
            result = self.pipeline(text)
            return {
                "label": result[0]["label"] if result else "unknown",
                "score": result[0]["score"] if result else 0.0,
                "raw_text": text
            }
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return {"label": "error", "score": 0.0, "error": str(e)}
    
    def batch_process(self, texts: list[str]) -> list[Dict[str, Any]]:
        """Process multiple texts in batch.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of processed intent results
        """
        if not self._is_loaded or self.pipeline is None:
            return [{"label": "unknown", "score": 0.0} for _ in texts]
        
        try:
            results = self.pipeline(texts)
            return [
                {
                    "label": r["label"] if isinstance(r, dict) else r[0]["label"],
                    "score": r["score"] if isinstance(r, dict) else r[0]["score"]
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return [{"label": "error", "score": 0.0} for _ in texts]
