# Phase 1 & 2 Implementation Guide

## Remaining Code Files to Create

This document contains the code for the remaining Phase 1 & 2 modules.

## models/command_classifier.py

```python
"""SVM-based intent classifier with OOD detection."""

from typing import Tuple
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class CommandClassifier:
    """Intent classifier with out-of-distribution detection."""
    
    def __init__(
        self,
        svm_kernel: str = "rbf",
        ood_threshold: float = 0.3,
        n_intents: int = 5
    ):
        self.svm = SVC(kernel=svm_kernel, probability=True, C=1.0)
        self.ood_threshold = ood_threshold
        self.n_intents = n_intents
        self.scaler = StandardScaler()
        self.intent_labels = {
            0: "music_control",
            1: "navigation",
            2: "climate_control",
            3: "error_handling",
            4: "ood"
        }
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train classifier."""
        X_scaled = self.scaler.fit_transform(X)
        self.svm.fit(X_scaled, y)
        self.is_fitted = True
        logger.info("Classifier fitted successfully")
    
    def predict(self, embeddings: np.ndarray) -> Tuple[str, float]:
        """
        Predict intent from embeddings with OOD detection.
        
        Args:
            embeddings: Token embeddings [seq_len, hidden_size]
            
        Returns:
            (intent_label, confidence) tuple
        """
        if not self.is_fitted:
            return ("ood", 0.0)
        
        # Use mean of sequence for classification
        mean_emb = embeddings.mean(axis=0, keepdims=True)
        mean_scaled = self.scaler.transform(mean_emb)
        
        pred = self.svm.predict(mean_scaled)[0]
        proba = self.svm.predict_proba(mean_scaled).max()
        
        # OOD detection
        if proba < self.ood_threshold:
            return ("ood", proba)
        
        return (self.intent_labels[pred], proba)
```

## models/rag_component.py

```python
"""RAG component for vehicle specification retrieval."""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class VehicleRAG:
    """FAISS-indexed retrieval over vehicle specifications."""
    
    def __init__(
        self,
        pdf_paths: Optional[List[str]] = None,
        chunk_size: int = 512,
        embedding_model: str = "sentence-transformers/all-minilm-l6-v2"
    ):
        self.pdf_paths = pdf_paths or []
        self.chunk_size = chunk_size
        self.embedding_model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = None
        self.index = None
        
        if pdf_paths:
            self._build_index()
    
    def _build_index(self):
        """Build FAISS index from documents."""
        try:
            import faiss
            
            # Mock implementation - in practice load PDFs
            self.documents = [
                "Engine displacement: 2.0L turbocharged",
                "Max power: 280 HP @ 6000 RPM",
                "Transmission: 8-speed automatic",
                "Fuel tank capacity: 60L",
                "Cabin temperature range: 16-32°C"
            ]
            
            embeddings = self.embedding_model.encode(self.documents)
            self.embeddings = np.array(embeddings).astype('float32')
            
            # Create FAISS index
            d = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(self.embeddings)
            
            logger.info(f"RAG index built with {len(self.documents)} documents")
        except ImportError:
            logger.warning("faiss not installed. RAG disabled.")
    
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        """Retrieve top-k relevant documents."""
        if self.index is None:
            return []
        
        query_emb = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_emb, top_k)
        
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]
```

## scripts/prepare_synthetic.py

```python
"""Generate synthetic in-car dialog data."""

from typing import List, Dict
import random
import json

class SyntheticDataGenerator:
    """Generate synthetic training data for in-car voice assistant."""
    
    def __init__(
        self,
        num_samples: int = 10000,
        intents: List[str] = None,
        model: str = "mistral-7b",
        quantization: str = "4bit"
    ):
        self.num_samples = num_samples
        self.intents = intents or [
            "music_control", "navigation", "climate", "error_handling"
        ]
        self.model = model
        self.quantization = quantization
    
    def generate(self) -> List[Dict]:
        """Generate synthetic samples."""
        samples = []
        
        templates = {
            "music_control": [
                "Play {artist}",
                "Next song", "Pause music",
                "Set volume to {level}"
            ],
            "navigation": [
                "Navigate to {location}",
                "Show me the nearest {poi}",
                "Route planning"
            ],
            "climate": [
                "Set temperature to {temp}",
                "Increase fan speed",
                "Turn on heated seats"
            ],
            "error_handling": [
                "Engine warning",
                "Battery low",
                "Tire pressure alert"
            ]
        }
        
        for i in range(self.num_samples):
            intent = random.choice(self.intents)
            template = random.choice(templates.get(intent, ["unknown"]))
            
            samples.append({
                "id": i,
                "intent": intent,
                "text": template,
                "audio_path": f"data/synthetic/audio/{i}.wav"
            })
        
        return samples
```

## pipeline/end_to_end.py

```python
"""End-to-end inference pipeline."""

from typing import Dict, Any, Optional
from models import VoskSTTEngine, QuantizedNLUPipeline, CommandClassifier, VehicleRAG
import logging

logger = logging.getLogger(__name__)

class InCarVoiceAssistant:
    """Complete in-car voice assistant pipeline."""
    
    def __init__(
        self,
        stt_model: str = "vosk",
        nlu_model: str = "microsoft/phi-3-mini",
        quantization: str = "4bit",
        use_rag: bool = True
    ):
        self.stt = VoskSTTEngine()
        self.nlu = QuantizedNLUPipeline(nlu_model, quantization)
        self.classifier = CommandClassifier()
        self.rag = VehicleRAG() if use_rag else None
    
    def process_audio(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        End-to-end processing from audio to response.
        
        Args:
            audio_bytes: Raw audio data
            
        Returns:
            {intent, confidence, response, context}
        """
        # STT
        transcription = self.stt.transcribe_stream([audio_bytes])
        
        # Tokenization and NLU
        tokens = self.nlu.tokenize(transcription.text)
        embeddings, logits = self.nlu.forward(tokens)
        
        # Intent classification
        intent, confidence = self.classifier.predict(embeddings.detach().numpy())
        
        # RAG retrieval if enabled
        context = ""
        if self.rag and confidence > 0.5:
            context = " ".join(self.rag.retrieve(embeddings[0].detach().numpy()))
        
        return {
            "intent": intent,
            "confidence": float(confidence),
            "transcript": transcription.text,
            "response": f"Intent: {intent} (conf: {confidence:.2f})",
            "context": context
        }
```

## How to Use These Files

1. Create files in models/ directory:
   - `models/command_classifier.py`
   - `models/rag_component.py`

2. Create files in scripts/ directory:
   - `scripts/prepare_synthetic.py`

3. Create files in pipeline/ directory:
   - `pipeline/end_to_end.py`

4. Initialize pipeline:
```python
from pipeline.end_to_end import InCarVoiceAssistant

assistant = InCarVoiceAssistant(use_rag=True)
result = assistant.process_audio(audio_bytes)
print(result)
```

## Next Steps

- Phase 3: Implement fine-tuning with LoRA/QLoRA
- Phase 4: Containerization and CI/CD
- 
## Phase 3: Fine-tuning with LoRA/QLoRA

### Overview
Phase 3 focuses on parameter-efficient fine-tuning of quantized language models using LoRA (Low-Rank Adaptation) and QLoRA techniques.

#### models/lora_tuner.py
```python
"""LoRA/QLoRA fine-tuning wrapper."""
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class LoRAFinetuner:
    def __init__(self, model_name="microsoft/phi-3-mini", lora_rank=16):
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.model = None
        self.peft_model = None
    
    def load_model(self) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model = prepare_model_for_kbit_training(self.model)
    
    def setup_peft(self) -> None:
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.peft_model = get_peft_model(self.model, lora_config)
```

---

## Phase 4: Containerization and CI/CD

### Overview
Phase 4 implements Docker containerization and GitHub Actions CI/CD pipeline for automated testing and deployment.

#### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0"]
```

#### tests/test_pipeline.py
```python
import pytest
import numpy as np
from pipeline.end_to_end import InCarVoiceAssistant

@pytest.fixture
def assistant():
    return InCarVoiceAssistant(use_rag=True)

def test_stt_transcription(assistant):
    audio_bytes = b"\x00" * 16000
    result = assistant.process_audio(audio_bytes)
    assert "transcript" in result

def test_intent_classification(assistant):
    intent, confidence = assistant.classifier.predict(np.random.randn(10, 768))
    assert intent in ["music_control", "navigation", "climate_control", "ood"]
    assert 0 <= confidence <= 1
```
