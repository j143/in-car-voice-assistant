# In-Car Voice Assistant: Edge-Optimized SLM Deployment

**Strategic Goal:** Deploy a domain-adapted Small Language Model (SLM) for in-vehicle voice assistants on NVIDIA Orin edge hardware, achieving:
- **Latency:** <100ms per query
- **Accuracy:** >90% on automotive technical nomenclature (error codes, part numbers)
- **Memory:** <2GB footprint for edge deployment

An end-to-end system implementing Speech-to-Text (STT) → Natural Language Understanding (NLU) → Command Execution pipeline, optimized for edge deployment with QLoRA fine-tuning, AWQ quantization, and automotive domain adaptation.

## Quick Start

**Baseline evaluation** (rule-based, no ML dependencies):

```bash
pip install -r requirements-min.txt
PYTHONPATH=. python scripts/evaluate_domain.py --test data/automotive_domain_test.jsonl
```

**Current baseline:** 20% accuracy on technical terms (Target: >90%)  
**Gap:** Need domain adaptation via QLoRA fine-tuning

Text command processing:

```bash
PYTHONPATH=. python scripts/run_assistant.py --text "Error code P0420 detected on ECU"
```

Run experiments across configurations:

```bash
PYTHONPATH=. python scripts/experiments.py data/automotive_domain_test.jsonl --runs 5
```

## Experiments

### Domain Adaptation Workflow

**1. Baseline Evaluation**
```bash
PYTHONPATH=. python scripts/evaluate_domain.py --test data/automotive_domain_test.jsonl
```
*Expected: ~20% accuracy (rule-based) → Need fine-tuning*

**2. QLoRA Fine-Tuning** (requires GPU + `requirements-train.txt`)
```bash
pip install -r requirements-train.txt
python scripts/train_qlora.py \
  --model microsoft/phi-2 \
  --train data/automotive_domain_train.jsonl \
  --output checkpoints/automotive-adapter \
  --epochs 3 \
  --batch-size 4
```

**3. AWQ Export for Edge** (requires `requirements-awq.txt`)
```bash
pip install -r requirements-awq.txt
python scripts/export_awq.py \
  --model microsoft/phi-2 \
  --adapter checkpoints/automotive-adapter \
  --output exports/automotive-awq-4bit \
  --w-bit 4
```

**4. Performance Tracking**

Open `notebooks/edge_optimization.ipynb` to track:
- Baseline vs QLoRA vs AWQ performance
- Latency profiling (<100ms target)
- Memory footprint (<2GB target)
- Technical term recognition rate (>90% target)

### Quick Experiments

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

This project implements an **edge-optimized voice command system** for automotive environments, targeting **NVIDIA Jetson Orin** deployment. The architecture follows a research-to-production workflow:

### Current Status (Phase 1-2)
- ✅ **Data:** Domain-specific automotive dataset with error codes (P0420, P0171), part numbers (K 300 503 v17), technical terms (ECU, TPMS, ABS)
- ✅ **Baseline:** Rule-based classifier achieving <100ms latency (✓) but 20% accuracy (❌ target: >90%)
- ✅ **Pipeline:** Modular STT → NLU → Classification → RAG with selector flags
- ✅ **Evaluation:** Domain-specific metrics for technical term and error code recognition

### Roadmap (Phase 3-4)
- 🔄 **QLoRA Fine-Tuning:** Domain adaptation on 25+ automotive samples → target >90% accuracy
- 🔄 **AWQ Quantization:** 4-bit export for <2GB memory footprint on Orin
- 🔄 **TensorRT-LLM:** Optimize inference to <100ms with AWQ kernels
- 📋 **Edge Deployment:** Docker container with health checks for Jetson Orin

### Architecture Components

- **Lightweight STT**: Vosk-based speech recognition (optional, lazy-loaded)
- **Quantized NLU**: 4-bit quantized small language models (Phi-2, Phi-3-mini)
- **Command Classification**: Rule-based (baseline) + SVM + optional ML classifier
- **Domain Adaptation**: QLoRA fine-tuning on automotive technical nomenclature
- **Edge Quantization**: AWQ for activation-aware 4-bit deployment
- **RAG Component**: Optional FAISS-indexed vehicle specs for context

## Project Structure

```
in-car-voice-assistant/
├── data/
│   ├── automotive_domain_train.jsonl   # 25 training samples (error codes, part numbers)
│   ├── automotive_domain_test.jsonl    # 15 test samples
│   └── bosch_dataset_seed.jsonl        # 99 seed samples for synthesis
├── models/
│   ├── stt_engine.py                  # Vosk wrapper (lazy-loaded)
│   ├── nlu_pipeline.py                # Quantized LLM pipeline (Phi-2 / DistilBERT)
│   ├── command_classifier.py          # Rule-based intent → command mapper
│   ├── command_classifier_svm.py      # Optional SVM classifier with OOD detection
│   ├── rag_component.py               # Knowledge-base RAG
│   ├── rag_faiss.py                   # Optional FAISS-indexed RAG
│   ├── lora_tuner.py                  # LoRA/QLoRA fine-tuning wrapper
│   ├── domain_dataset.py              # PyTorch Dataset for JSONL samples
│   └── telemetry.py                   # Mock vehicle telemetry adapter
├── pipeline/
│   └── end_to_end.py                  # Full STT → NLU → Classifier → RAG pipeline
├── scripts/
│   ├── run_assistant.py               # CLI: run the assistant on text or audio
│   ├── evaluate_domain.py             # Domain evaluation (accuracy, latency)
│   ├── experiments.py                 # Sweep all classifier × RAG configs
│   ├── benchmark_latency.py           # Latency profiler
│   ├── train_qlora.py                 # QLoRA fine-tuning (requires GPU)
│   ├── export_awq.py                  # AWQ 4-bit export for edge (requires GPU)
│   ├── synthesize_dataset.py          # Expand seed data to 10k samples (OpenAI API)
│   ├── prepare_synthetic.py           # Simple synthetic data generator (no API)
│   ├── label_dataset.py               # Label a plain-text file to JSONL
│   ├── clean_logs.py                  # Log cleanup utility
│   └── telemetry_sim.py               # Telemetry simulation
├── tests/
│   ├── test_pipeline.py               # Pipeline unit tests
│   └── sample_eval.jsonl              # Small evaluation fixture
├── notebooks/
│   └── edge_optimization.ipynb        # Track baseline → QLoRA → AWQ metrics
├── config/
│   └── assistant.yaml                 # Default component selections
├── Dockerfile                         # Slim image for local/CI use
├── Dockerfile.orin                    # NVIDIA Jetson Orin deployment image
├── requirements.txt                   # Full dependencies
├── requirements-min.txt               # Minimal (text-only, no model downloads)
├── requirements-nlu.txt               # NLU pipeline dependencies
├── requirements-rag.txt               # RAG/FAISS dependencies
├── requirements-train.txt             # QLoRA training dependencies
├── requirements-awq.txt               # AWQ export dependencies
└── README.md
```

## Phase 1: Data

**Datasets (included):**
- `data/automotive_domain_train.jsonl` — 25 samples with error codes (P0420, P0171, P0300), part numbers (K 300 503 v17), technical terms
- `data/automotive_domain_test.jsonl` — 15 evaluation samples
- `data/bosch_dataset_seed.jsonl` — 99 seed samples for LLM-based expansion

**Expand training data (requires OpenAI key):**
```bash
python scripts/synthesize_dataset.py \
  --seed-file data/bosch_dataset_seed.jsonl \
  --output-file data/automotive_domain_train_10k.jsonl \
  --multiplier 50
```

**Generate synthetic data without API:**
```bash
PYTHONPATH=. python scripts/prepare_synthetic.py
```

## Phase 2: Pipeline

**Components (all implemented):**
- `models/stt_engine.py` — Vosk STT, lazy-loaded, streaming
- `models/nlu_pipeline.py` — DistilBERT (default) or Phi-2 with LoRA adapter
- `models/command_classifier.py` — Rule-based intent → command mapper
- `models/command_classifier_svm.py` — Optional SVM with OOD detection
- `models/rag_component.py` — Knowledge-base RAG (no extra deps)
- `models/rag_faiss.py` — FAISS semantic RAG (optional, requires `requirements-rag.txt`)

**Run the pipeline:**
```bash
PYTHONPATH=. python scripts/run_assistant.py --text "Error code P0420 detected on ECU"
# With SVM classifier and FAISS RAG:
PYTHONPATH=. python scripts/run_assistant.py --text "Set temperature to 72" --classifier svm --rag faiss
```

## Phase 3: QLoRA Fine-Tuning

```bash
pip install -r requirements-train.txt
python scripts/train_qlora.py \
  --model microsoft/phi-2 \
  --train data/automotive_domain_train.jsonl \
  --output checkpoints/automotive-adapter \
  --epochs 3 --batch-size 4
```

Evaluate with adapter:
```bash
PYTHONPATH=. python scripts/evaluate_domain.py \
  --test data/automotive_domain_test.jsonl \
  --adapter checkpoints/automotive-adapter
```

## Phase 4: AWQ Export & Edge Deployment

```bash
pip install -r requirements-awq.txt
python scripts/export_awq.py \
  --model microsoft/phi-2 \
  --adapter checkpoints/automotive-adapter \
  --output exports/automotive-awq-4bit \
  --w-bit 4
```

See `DEPLOYMENT.md` for Jetson Orin Docker setup.

## Installation

```bash
git clone https://github.com/j143/in-car-voice-assistant.git
cd in-car-voice-assistant

# Minimal (text path only, no model download)
pip install -r requirements-min.txt

# Full dependencies
pip install -r requirements.txt
```

## Usage

```python
from pipeline.end_to_end import VoiceAssistantPipeline

assistant = VoiceAssistantPipeline(
    use_rag=True,
    nlu_model_name='distilbert-base-uncased-finetuned-sst-2-english',
    classifier_type='rule',   # or 'svm'
    rag_type='kb'             # or 'faiss'
)

print(assistant.process_text("Set temperature to 72"))
```

## Benchmarks

**Performance Targets (NVIDIA Orin Edge)**

| Metric | Target | Baseline | QLoRA | AWQ 4-bit |
|--------|--------|----------|-------|-----------|
| **Accuracy** (technical terms) | >90% | 20% ❌ | TBD | TBD |
| **Latency** (avg, text path) | <100ms | 0.01ms ✅ | TBD | TBD |
| **Memory** (inference) | <2GB | <0.1GB ✅ | ~2.5GB | <2GB |
| **Error Code Recognition** | >85% | 0% ❌ | TBD | TBD |

**Test Set:** 15 automotive domain samples with:
- Error codes: P0420, P0171, P0300, C1234, B1318, U0101
- Technical terms: ECU, TPMS, ABS, ESC, DTC, hydraulic pump motor
- Part numbers: K 300 503 v17, A 000 420 17 20

**Run benchmarks:**
```bash
PYTHONPATH=. python scripts/evaluate_domain.py --test data/automotive_domain_test.jsonl
```

## References

- AISHELL-5: https://github.com/wenet-e2e/aishell5
- Voice Datasets Catalog: https://github.com/jim-schwoebel/voice_datasets
- Digital Voice Assistant Pipeline: https://github.com/search?q=digital-voice-assistant-in-car
- PEFT/LoRA: https://github.com/huggingface/peft
- BitsAndBytes: https://github.com/TimDettmers/bitsandbytes

## Disclaimer

_"Not affiliated with Bosch"_, this repository is for research and experimentation purpose.
