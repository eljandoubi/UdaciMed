# UdaciMed — Pneumonia Detection Model Optimization

End-to-end optimization pipeline for a ResNet-18 chest X-ray pneumonia detector, from baseline profiling to hardware-accelerated ONNX production deployment. Developed as part of the **Udacity Machine Learning Engineer Nanodegree** course on model optimization for clinical AI.

---

## Project Overview

UdaciMed is a medical AI startup whose radiologists use a binary pneumonia detection model to triage chest X-rays. Before the model can be approved for production, it must meet **UdaciMed's Universal Performance Standard**:

| Target | Threshold |
|--------|-----------|
| FLOP reduction vs baseline | > 80 % |
| Peak memory footprint | < 100 MB |
| Single-image inference latency | < 3 ms |
| Batch throughput | > 2 000 samples/sec |
| Clinical sensitivity (recall) | > 98 % |

The project follows a three-notebook pipeline:

```
Notebook 01  →  Baseline profiling & optimization analysis
Notebook 02  →  Architecture optimization & fine-tuning
Notebook 03  →  Hardware-accelerated ONNX deployment
```

---

## Results Summary

All experiments run on **NVIDIA GeForce RTX 3070 Laptop GPU (8 GB VRAM)**. Notebook 03 uses FP16 ONNX + ONNX Runtime CUDA EP.

| Metric | Baseline | Arch Optimized | Deployed (ONNX FP16) | Target | Status |
|--------|----------|---------------|----------------------|--------|--------|
| Parameters | 11.2 M (42.6 MB) | 1.45 M (5.5 MB) | — | — | — |
| FLOP reduction | 0 % | **98.5 %** | — | > 80 % | ✅ |
| Peak memory | 318.2 MB | 211.1 MB | **2.77 MB** (file) | < 100 MB | ✅ |
| Latency (BS=1) | 3.61 ms | 4.59 ms | **0.543 ms** | < 3 ms | ✅ |
| Throughput (BS=32) | 1 940 s/s | **6 761 s/s** | **6 779 s/s** | > 2 000 s/s | ✅ |
| Sensitivity | 100.0 % (thr=0.4) | **98.2 %** (thr=0.3) | **98.72 %** (thr=0.3) | > 98 % | ✅ |

**5 / 5 production targets met** after hardware acceleration.

---

## Dataset

**[PneumoniaMNIST](https://medmnist.com/)** — a binary classification subset of the MedMNIST benchmark.

| Split | Samples |
|-------|---------|
| Train | 4 708 |
| Validation | 524 |
| Test | 624 |

- **Classes:** `0 = Normal`, `1 = Pneumonia`
- **Resolution used:** 64 × 64 pixels
- **Preprocessing:** RGB conversion + ImageNet normalization

---

## Repository Structure

```
UdaciMed/
├── notebooks/
│   ├── 01_baseline_analysis.ipynb          # Profiling + optimization opportunity analysis
│   ├── 02_architecture_optimization.ipynb  # Apply optimizations, fine-tune, evaluate
│   └── 03_deployment_acceleration.ipynb    # ONNX export + ONNX Runtime benchmarking
├── utils/
│   ├── model.py                   # ResNetBaseline, create_baseline_model, train
│   ├── data_loader.py             # PneumoniaMNIST loaders
│   ├── architecture_optimization.py   # All 7 optimization implementations
│   ├── evaluation.py              # evaluate_with_multiple_thresholds
│   ├── profiling.py               # PerformanceProfiler (timing, FLOPs, memory)
│   ├── visualization.py           # Plotting helpers
│   └── __init__.py
├── results/
│   ├── best_baseline_model.pth    # Saved baseline weights
│   ├── optimized_model.pth        # Saved fine-tuned optimized weights
│   ├── baseline_results.pkl       # Baseline profiling dict
│   └── optimization_results_*.pkl # Per-experiment results
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA (RTX 3070 used; CPU fallback available for NB01/NB02)

### Install

```bash
cd UdaciMed
pip install -r requirements.txt
# or
uv sync
```

> **Note:** `tensorrt-cu12` requires CUDA 12, cuDNN 8.9.2, and NVIDIA driver ≥ 550. Remove it from `requirements.txt` if running CPU-only.

---

## Running the Notebooks

Run in order — each notebook saves artefacts consumed by the next.

```bash
jupyter notebook
```

| Step | Notebook | Produces |
|------|----------|----------|
| 1 | `01_baseline_analysis.ipynb` | `results/baseline_results.pkl`, `results/best_baseline_model.pth` |
| 2 | `02_architecture_optimization.ipynb` | `results/optimized_model.pth`, `results/optimization_results_<name>.pkl` |
| 3 | `03_deployment_acceleration.ipynb` | ONNX model, benchmark results, 5/5 targets verified |

---

## Implemented Optimizations (`utils/architecture_optimization.py`)

All seven optimization strategies are implemented:

| Optimization | Key Idea | Measured / Estimated Gain |
|---|---|---|
| **Interpolation Removal** | Skip 64→224 upscaling; run at native 64×64 | 47.9 % FLOP reduction (est.), drives most of the 98.5 % combined gain |
| **Depthwise Separable Convolutions** | Replace 3×3 conv with depthwise + pointwise | 88.6 % param reduction across 16 candidate layers |
| **Grouped Convolutions** | Split channels into parallel groups | 50 % FLOP reduction per layer (groups=2) |
| **Inverted Residual Blocks** | MobileNetV2-style expand→depthwise→project | 60 % FLOP reduction per block (est.) |
| **Low-Rank Factorization** | SVD decomposition of linear layers | 0 gain here — only 1 FC layer with 1,026 params |
| **Channel Optimization** | Channels-last memory layout + in-place ReLU | ~1.2× GPU speedup, zero accuracy cost |
| **Parameter Sharing** | Share weights across same-shape layers | ~10.7 MB memory saving (est.) |

The `create_optimized_model()` function applies techniques in dependency order:

```
interpolation_removal → inverted_residuals → depthwise_separable
  → grouped_conv → channel_optimization → lowrank_factorization → parameter_sharing
```

### Configuration Used

```python
OPTIMIZATION_CONFIG = {
    'interpolation_removal': True,    # biggest FLOP win
    'depthwise_separable':   True,    # biggest parameter win
    'grouped_conv':          False,
    'channel_optimization':  False,
    'inverted_residuals':    False,
    'lowrank_factorization': False,
    'parameter_sharing':     False,
    'memory_format': torch.preserve_format,
    'use_amp': False,
}
```

---

## Baseline Model

**Architecture:** ResNet-18 with an adaptive input wrapper that bilinearly upscales any input to 224×224 before the backbone. Classification head: `Dropout(0.2) → Linear(512, 2)`.

**Key baseline numbers (measured, RTX 3070):**

| | |
|---|---|
| Parameters | 11,177,538 (42.6 MB) |
| Peak inference memory | 318.2 MB (activations: 212.9 MB, weights: 42.6 MB) |
| Single-sample latency | 3.61 ms |
| Batch throughput (BS=32) | 1 940 samples/sec |
| Sensitivity (threshold=0.7) | 99.5 % |

**Primary bottleneck:** the 64→224 bilinear upscale inflates every downstream feature map by ~12×, causing 66.9 % of peak memory to be activation overhead.

---

## Optimized Model

After applying interpolation removal + depthwise separable convolutions and fine-tuning for 15 epochs (lr=1e-4):

| | |
|---|---|
| Parameters | 1,449,986 (5.5 MB) — **−87 %** |
| Peak inference memory | 211.1 MB (activations: 137.4 MB) |
| Batch throughput (BS=32) | 6 761 samples/sec — **+3.5×** |
| Single-sample latency | 4.59 ms (regressed on GPU; closed by ONNX export) |
| FLOP reduction | 98.5 % |
| Sensitivity (threshold=0.3) | 98.2 % — **target met** |
| Weight transfer | 46/110 layers compatible; 64 modified/new layers retrained |

---

## Deployment Pipeline (Notebook 3)

```
Optimized PyTorch model
        │
        ▼  torch.onnx.export (FP16, dynamic axes)
FP16 ONNX model (2.77 MB)
        │
        ▼  ort.InferenceSession (CUDAExecutionProvider + CPU fallback)
ONNX Runtime inference
        │
        ▼  benchmark_performance() — BS = 1, 8, 16, 32
Measured: 0.543 ms @ BS=1 | 6 779 s/s @ BS=32 | 18.09 MB GPU memory
        │
        ▼  validate_clinical_performance() on 624 test samples
Sensitivity: 98.72 % at threshold=0.3
```

**Final scorecard:**

| Metric | Target | Achieved |
|--------|--------|----------|
| Memory | < 100 MB | **2.77 MB** ✅ |
| Latency | < 3 ms | **0.543 ms** ✅ |
| Throughput | > 2 000 s/s | **6 779 s/s** ✅ |
| FLOP reduction | > 80 % | **98.5 %** ✅ |
| Sensitivity | > 98 % | **98.72 %** ✅ |

---

## Cross-Platform Deployment Guide

| Target | Recommended Stack | Key Notes |
|--------|------------------|-----------|
| GPU server (cloud) | ONNX Runtime + TensorRT EP | 2–5× over CUDA EP; compile per GPU model |
| Multi-tenant GPU service | Triton + TensorRT backend | Dynamic batching; highest DevOps cost |
| Intel CPU workstation | ONNX Runtime + OpenVINO EP | 2–4× over PyTorch on Intel; single ONNX file |
| Intel CPU (max perf) | Native OpenVINO IR | Deepest fusions; Intel lock-in |
| iOS | Core ML (via `coremltools`) | Apple Neural Engine; lowest battery draw on iPhone |
| Android | LiteRT (TFLite) | Smallest binary; widest Android HW support |
| Cross-platform mobile launch | ONNX Runtime Mobile | One model file for iOS + Android |

---

## Training Details

- **Optimizer:** AdamW with weight decay
- **Scheduler:** StepLR (step=7, gamma=0.1 for fine-tuning)
- **Loss:** CrossEntropyLoss
- **Early stopping:** monitors validation accuracy (patience=5)
- **Baseline:** 10 epochs, lr=3e-4 → best val acc: see NB01 training cell
- **Fine-tuning optimized model:** 15 epochs, lr=1e-4 → best val acc 95.6 % (epoch 11)

---

## License

This project is provided for educational purposes as part of the Udacity Machine Learning Engineer Nanodegree program.
