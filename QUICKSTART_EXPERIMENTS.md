# Quick Start: Edge AI Experiments

## You are here: Ready to execute domain adaptation

**Goal:** Improve from 20% → >90% accuracy on automotive technical terms

---

## 1. Verify Baseline (No GPU needed)

```bash
pip install -r requirements-min.txt
PYTHONPATH=. python scripts/evaluate_domain.py --test data/automotive_domain_test.jsonl
```

**Expected output:**
```
Command Accuracy: 20.0% (Target: ≥90%)
Avg Latency: 0.00ms (Target: <100ms)
✓ Latency Target Met: True
✓ Accuracy Target Met: False  ← Need to fix this
```

---

## 2. Train QLoRA Adapters (Requires GPU)

### Option A: Local GPU / Codespace with GPU
```bash
pip install -r requirements-train.txt
python scripts/train_qlora.py \
  --model microsoft/phi-2 \
  --train data/automotive_domain_train.jsonl \
  --output checkpoints/automotive-adapter \
  --epochs 3 \
  --batch-size 4
```

### Option B: Google Colab (Free GPU)
Upload `data/automotive_domain_train.jsonl` and `scripts/train_qlora.py`, then run in notebook.

### Expected Training Time
- ~10-15 minutes on T4 GPU
- ~25 samples × 3 epochs = 75 training steps
- Checkpoint saved to `checkpoints/automotive-adapter/`

---

## 3. Evaluate Fine-Tuned Model

After training, re-evaluate (update pipeline to load adapter):

```bash
PYTHONPATH=. python scripts/evaluate_domain.py
```

**Target:**
```
Command Accuracy: >90.0% ✓
Technical Term Recognition: >90% ✓
Error Code Recognition: >85% ✓
```

---

## 4. Export for Edge Deployment

```bash
pip install -r requirements-awq.txt
python scripts/export_awq.py \
  --model microsoft/phi-2 \
  --adapter checkpoints/automotive-adapter \
  --output exports/automotive-awq-4bit \
  --w-bit 4
```

**Result:** 4-bit quantized model <2GB, ready for NVIDIA Orin

---

## 5. Track Experiments

Open Jupyter notebook:
```bash
jupyter notebook notebooks/edge_optimization.ipynb
```

Compare baseline → QLoRA → AWQ performance in table format.

---

## Quick Commands Reference

| Task | Command |
|------|---------|
| Baseline eval | `PYTHONPATH=. python scripts/evaluate_domain.py` |
| Train QLoRA | `python scripts/train_qlora.py --model microsoft/phi-2 --epochs 3` |
| Export AWQ | `python scripts/export_awq.py --output exports/automotive-awq-4bit` |
| Run experiments | `PYTHONPATH=. python scripts/experiments.py data/automotive_domain_test.jsonl` |
| Test command | `PYTHONPATH=. python scripts/run_assistant.py --text "Error P0420 detected"` |

---

## Dataset Overview

- **Training:** 25 samples with error codes (P0420, P0171, P0300), part numbers (K 300 503 v17)
- **Test:** 15 samples with diverse technical terms (ECU, TPMS, ABS, ESC)
- **Format:** JSONL with `text`, `intent`, `command`, `technical_terms`, `error_codes`

---

## Troubleshooting

**No GPU available:**
- Use Google Colab for training (free T4 GPU)
- Or use Codespaces with GPU upgrade ($0.36/hour)

**Out of memory:**
- Reduce `--batch-size` to 2 or 1
- Use gradient accumulation (already set to 4)

**Low accuracy after training:**
- Increase `--epochs` to 5-10
- Check if technical terms are being tokenized correctly
- Add more diverse training samples

---

**Next:** Run baseline eval, then proceed to QLoRA training when GPU is available.
