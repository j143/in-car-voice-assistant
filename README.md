# In-Car Voice Assistant

An end-to-end in-car voice assistant system with Speech-to-Text (STT) → Natural Language Understanding (NLU) → Command Execution pipeline. Optimized for edge deployment with quantized LLMs, parameter-efficient fine-tuning (LoRA/QLoRA), RAG integration, and robust out-of-distribution (OOD) detection.

## Quick Start

Text-only run (no heavy deps):

```bash
pip install -r requirements-min.txt
python scripts/run_assistant.py --text "Set temperature to 72"
```

Enable NLU (Hugging Face model):

```bash
pip install -r requirements-nlu.txt
python scripts/run_assistant.py --text "Open the driver window"
```

FAISS-backed RAG (optional):

```bash
pip install -r requirements-rag.txt
```

Latency benchmark:

```bash
python scripts/benchmark_latency.py --runs 10 --text "Set temperature to 72"
```

Vosk STT: download a Vosk model to `models/vosk_models/en_us` and install `vosk` (already listed in `requirements.txt`). Then use `--audio` with raw PCM16 input.

## Experiments

Run a small evaluation and latency benchmark across configurations:

```bash
PYTHONPATH=. python scripts/experiments.py tests/sample_eval.jsonl --runs 10 --text "Set temperature to 72"
```

Label a text file quickly:

```bash
PYTHONPATH=. python scripts/label_dataset.py tests/sample_texts.txt tests/labeled.jsonl
```

Telemetry simulation:

```bash
python scripts/telemetry_sim.py
```

## Overview

This project implements a production-ready in-car voice command system inspired by real-world EV cabin scenarios. The architecture supports:

- **Lightweight STT**: Vosk-based speech recognition optimized for in-car noise
- **Quantized NLU**: 4-bit/8-bit quantized small language models (Phi-3-mini, Mistral-7B)
- **Command Classification**: SVM-based intent recognition with OOD filtering
- **RAG Component**: FAISS-indexed vehicle manuals and specs for context-aware responses
- **Parameter-Efficient Fine-Tuning**: LoRA/QLoRA adapters on 8GB RAM constraints

## Project Structure

```
in-car-voice-assistant/
├── data/
│   ├── raw/                    # AISHELL-5, voice datasets
│   ├── synthetic/              # Generated dialog JSONL
│   └── processed/              # Tokenized, stratified splits
├── models/
│   ├── stt_engine.py          # Vosk wrapper
│   ├── nlu_pipeline.py        # Quantized LLM pipeline
│   ├── command_classifier.py  # SVM + OOD detection
│   └── rag_component.py       # FAISS indexing & retrieval
├── fine_tuning/
│   ├── lora_config.py         # PEFT configurations
│   ├── train.py               # QLoRA training loop
│   └── eval.py                # Validation metrics
├── pipeline/
│   ├── end_to_end.py          # Full inference pipeline
│   └── integration_test.py
├── scripts/
│   ├── download_datasets.sh   # AISHELL-5, voice_datasets
│   ├── prepare_synthetic.py   # Generate dialog data
│   └── benchmark.py           # Latency/memory profiling
├── notebooks/
│   ├── data_exploration.ipynb
│   └── fine_tuning_demo.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Phase 1: Data Collection & Preparation

### Datasets

1. **AISHELL-5**: First open in-car multi-channel dataset
   - Real EV cabin recordings
   - Multiple driving scenarios (highway, urban, parking)
   - Download: `python scripts/download_datasets.sh`

2. **jim-schwoebel/voice_datasets**: Catalogued open voice datasets
   - In-car audio, command datasets, wake words
   - Links: https://github.com/jim-schwoebel/voice_datasets

3. **Synthetic Data**: Generated using LLM augmentation
   - Car-related queries, error dialogs, service interactions
   - Format: JSONL with speaker, intent, text, audio_path
   - Generated with diverse sentence structures matching real distribution

### Synthetic Data Generation

```python
from scripts.prepare_synthetic import SyntheticDataGenerator

generator = SyntheticDataGenerator(
    num_samples=10000,
    intents=['music_control', 'navigation', 'climate', 'error_handling'],
    model='mistral-7b',  # Or Phi-3-mini
    quantization='4bit'
)

synthetic_df = generator.generate()
synthetic_df.to_json('data/synthetic/in_car_dialogs.jsonl', orient='records', lines=True)
```

### Data Processing Pipeline

- **STT Processing**: Convert audio to transcripts using Vosk
- **Tokenization**: BPE tokenization for quantized model input
- **Stratification**: Balanced intent distribution across train/val/test
- **Normalization**: Audio resampling (16kHz), text lowercasing

Output: `data/processed/{train,val,test}_dataset.jsonl`

## Phase 2: Core Pipeline Implementation

### 1. Speech-to-Text (STT)

**File**: `models/stt_engine.py`

```python
from models.stt_engine import VoskSTTEngine

stt = VoskSTTEngine(
    model_path='models/vosk_models/en_us',
    sample_rate=16000
)

# Streaming transcription
transcript = stt.transcribe_stream(audio_chunks)
confidence = stt.get_confidence()
```

- Lightweight Vosk model (~50MB)
- Streaming support for real-time processing
- Noise-robust features for cabin acoustics

### 2. Natural Language Understanding (NLU)

**File**: `models/nlu_pipeline.py`

Quantized language model pipeline with BitsAndBytes 4-bit quantization:

```python
from models.nlu_pipeline import QuantizedNLUPipeline
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

nlu = QuantizedNLUPipeline(
    model_name='microsoft/phi-3-mini',  # 3.8B params
    quantization_config='4bit',
    lora_config=config
)

embeddings, logits = nlu.forward(transcript_tokens)
```

**Key Features**:
- 4-bit quantization via bitsandbytes
- LoRA adapters (~0.5% trainable params)
- Fits in 8GB RAM (GitHub Codespaces constraint)
- Context window: 2048 tokens

### 3. Command Classification

**File**: `models/command_classifier.py`

SVM-based intent classifier with OOD detection:

```python
from models.command_classifier import CommandClassifier
from sklearn.svm import SVC

classifier = CommandClassifier(
    svm_kernel='rbf',
    ood_threshold=0.3,  # Confidence threshold
    n_intents=5
)

intent, confidence = classifier.predict(embeddings)
is_ood = confidence < classifier.ood_threshold
```

**Intents** (in-car focused):
- Music Control (play, pause, volume)
- Navigation (route, destination)
- Climate Control (temp, fan, seats)
- Error Handling (vehicle alerts, diagnostics)
- OOD Fallback (unknown queries)

### 4. RAG Component

**File**: `models/rag_component.py`

FAISS-indexed retrieval over vehicle specifications:

```python
from models.rag_component import VehicleRAG

rag = VehicleRAG(
    pdf_paths=['docs/vehicle_specs.pdf', 'docs/manual.pdf'],
    chunk_size=512,
    embedding_model='sentence-transformers/all-minilm-l6-v2'
)

# Retrieve context for query
context = rag.retrieve(query_embedding, top_k=3)
```

- Indexes public vehicle documentation PDFs
- Sentence-Transformers for semantic search
- Context injection into LLM for grounding

## Phase 3: Edge-Optimized Fine-Tuning (Coming Soon)

- QLoRA training loop on 8GB RAM
- Synthetic dataset augmentation strategies
- Evaluation metrics: Intent accuracy, latency, memory

## Phase 4: Repository & Documentation (Coming Soon)

- GitHub Actions CI/CD
- Docker containerization
- Deployment guides for edge devices

## Installation

```bash
# Clone repository
git clone https://github.com/j143/in-car-voice-assistant.git
cd in-car-voice-assistant

# Install dependencies
pip install -r requirements.txt

# Download models and datasets
python scripts/download_datasets.sh

# Generate synthetic data
python scripts/prepare_synthetic.py
```

## Usage

### End-to-End Inference (selectors)

```python
from pipeline.end_to_end import VoiceAssistantPipeline

assistant = VoiceAssistantPipeline(
    use_rag=True,
    nlu_model_name='distilbert-base-uncased-finetuned-sst-2-english',
    classifier_type='rule',   # or 'svm'
    rag_type='kb'             # or 'faiss'
)

# Process text (fast path)
print(assistant.process_text("Set temperature to 72"))
```

### Training with LoRA

```python
from fine_tuning.train import LoRATrainer

trainer = LoRATrainer(
    model='microsoft/phi-3-mini',
    dataset_path='data/processed/train_dataset.jsonl',
    output_dir='checkpoints/',
    num_epochs=3,
    batch_size=8  # On 8GB RAM
)

trainer.train()
```

## Benchmarks

Target metrics on GitHub Codespaces (8GB RAM):
- **Inference Latency**: < 500ms per query (STT + NLU + retrieval)
- **Memory Usage**: < 7GB (model + data)
- **Intent Accuracy**: > 90% on synthetic test set
- **OOD Detection F1**: > 0.85

## References

- AISHELL-5: https://github.com/wenet-e2e/aishell5
- Voice Datasets Catalog: https://github.com/jim-schwoebel/voice_datasets
- Digital Voice Assistant Pipeline: https://github.com/search?q=digital-voice-assistant-in-car
- PEFT/LoRA: https://github.com/huggingface/peft
- BitsAndBytes: https://github.com/TimDettmers/bitsandbytes

## License

MIT

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.
