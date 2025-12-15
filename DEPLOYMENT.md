# Edge Deployment Guide

## NVIDIA Jetson Orin Deployment

### Prerequisites
- NVIDIA Jetson Orin (8GB+ RAM)
- JetPack 5.1+ with CUDA 11.8+
- Docker with NVIDIA Container Runtime

### Build Container

```bash
# Build Orin-optimized image
docker build -f Dockerfile.orin -t automotive-assistant:orin .

# Run container with GPU support
docker run --runtime nvidia --gpus all \
  -v $(pwd)/exports:/app/models \
  automotive-assistant:orin \
  python scripts/run_assistant.py --text "Error code P0420 detected"
```

### Deploy AWQ Model

```bash
# Export AWQ model (on development machine with GPU)
python scripts/export_awq.py \
  --model microsoft/phi-2 \
  --adapter checkpoints/automotive-adapter \
  --output exports/automotive-awq-4bit

# Copy to Orin
scp -r exports/automotive-awq-4bit jetson@orin:/data/models/

# Run with TensorRT-LLM (optional, for <100ms inference)
# docker run --runtime nvidia --gpus all \
#   -v /data/models:/app/models \
#   automotive-assistant:orin-trt \
#   python scripts/inference_trt.py --model /app/models/automotive-awq-4bit
```

### Performance Optimization

1. **Memory:** AWQ 4-bit quantization reduces from ~7GB to <2GB
2. **Latency:** TensorRT-LLM optimizes AWQ kernels for <100ms inference
3. **Accuracy:** QLoRA adapters maintain >90% on technical terms

### Health Monitoring

```bash
# Health check endpoint
docker exec automotive-assistant python -c "from pipeline.end_to_end import VoiceAssistantPipeline; p = VoiceAssistantPipeline(use_rag=False); print(p.process_text('Error P0420'))"
```

## GitHub Codespaces / Local Development

For rapid iteration without edge hardware:

```bash
# Minimal text-only path
pip install -r requirements-min.txt
PYTHONPATH=. python scripts/run_assistant.py --text "TPMS alert front left"

# With NLU (requires GPU or CPU fallback)
pip install -r requirements-nlu.txt
PYTHONPATH=. python scripts/evaluate_domain.py
```

## CI/CD Pipeline

```yaml
# .github/workflows/edge-deploy.yml
- name: Train QLoRA
  run: python scripts/train_qlora.py --epochs 1
  
- name: Export AWQ
  run: python scripts/export_awq.py
  
- name: Evaluate
  run: python scripts/evaluate_domain.py
  
- name: Deploy to Orin
  run: scp -r exports/* orin:/data/models/
```
