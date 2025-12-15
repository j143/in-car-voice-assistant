# Strategic Alignment: Bosch Edge AI Research

## Project Goal

Deploy a **domain-adapted Small Language Model (SLM)** for in-vehicle voice assistants on **NVIDIA Orin** edge hardware.

### Success Criteria
- **Latency:** <100ms per query (real-time responsiveness in moving vehicle)
- **Accuracy:** >90% on automotive technical nomenclature
  - Error codes: P0420, P0171, P0300, C1234, etc.
  - Part numbers: K 300 503 v17, A 000 420 17 20
  - Technical terms: ECU, TPMS, ABS, ESC, DTC
- **Memory:** <2GB footprint (shared with autonomous driving features)

## Current Status

### ✅ Phase 1-2: Foundation (Complete)
- Domain-specific dataset: 25 training + 15 test samples
- Modular pipeline with component selectors
- Baseline evaluation framework
- **Baseline Performance:**
  - Accuracy: 20% ❌ (rule-based, no ML)
  - Latency: 0.01ms ✅ (text path)
  - Memory: <0.1GB ✅

### 🔄 Phase 3: Domain Adaptation (Ready to Execute)
**The Fix:** QLoRA fine-tuning on automotive vocabulary

**Implementation:**
```bash
# Train LoRA adapters (4-bit base model, ~100MB trainable params)
python scripts/train_qlora.py \
  --model microsoft/phi-2 \
  --train data/automotive_domain_train.jsonl \
  --epochs 3
```

**Expected Outcome:**
- Accuracy: 20% → >90% on technical terms
- Model learns to "speak Bosch" (error codes, part numbers)
- Memory: ~2.5GB (training), <2GB (after AWQ export)

### 📋 Phase 4: Edge Optimization (Next)
**The Fix:** AWQ quantization for NVIDIA Orin deployment

**Implementation:**
```bash
# Export 4-bit AWQ model
python scripts/export_awq.py \
  --model microsoft/phi-2 \
  --adapter checkpoints/automotive-adapter \
  --output exports/automotive-awq-4bit
```

**Expected Outcome:**
- Memory: 2.5GB → <2GB (4-bit quantization)
- Latency: Optimize with TensorRT-LLM for <100ms
- Preserves accuracy via activation-aware quantization

## Research-Grade Technical Details

### Why QLoRA?
Generic models don't know Bosch error codes. By training **LoRA adapters** (adding just ~100MB of trainable parameters) on internal domain data, we force the model to recognize automotive vocabulary without retraining 3.8B parameters.

### Why AWQ?
Standard weight clipping destroys accuracy. **Activation-aware Weight Quantization** protects the 1% of "salient" weights that matter most for technical queries, while shrinking the rest to 4-bit. Result: 4x smaller, same intelligence.

### Weekly Sprint Alignment (from role.md)

**Monday:** ✅ Literature review (QLoRA, AWQ papers)  
**Tuesday:** ✅ Baseline benchmarking → 20% accuracy identified  
**Wednesday:** 🔄 Implement QLoRA training pipeline  
**Thursday:** 📋 AWQ export + Orin deployment test  
**Friday:** 📋 Tech talk prep + code refactoring for production

## Next Actions

### Immediate (This Week)
1. **Run QLoRA training** on automotive dataset (requires GPU)
2. **Evaluate fine-tuned model** → target >90% accuracy
3. **Export AWQ model** for Orin deployment

### Commands to Execute
```bash
# 1. Train (requires GPU + requirements-train.txt)
pip install -r requirements-train.txt
python scripts/train_qlora.py --model microsoft/phi-2 --epochs 3

# 2. Evaluate
PYTHONPATH=. python scripts/evaluate_domain.py --test data/automotive_domain_test.jsonl

# 3. Export for edge
pip install -r requirements-awq.txt
python scripts/export_awq.py --model microsoft/phi-2 --output exports/automotive-awq-4bit
```

### Validation Criteria
- [ ] Accuracy >90% on test set
- [ ] Error code recognition >85%
- [ ] Latency <100ms (with TensorRT-LLM)
- [ ] Memory <2GB (AWQ export)

## Deliverables

### Technical Report (for team)
- CSV with experiment logs (baseline, QLoRA, AWQ)
- Latency/memory comparison table
- Failed edge cases analysis

### Production Artifact
- Docker image for Jetson Orin
- AWQ model checkpoint
- Deployment guide with health checks

---

**Status:** Foundation complete. Ready to execute QLoRA training and edge optimization.
