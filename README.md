# UdaciMed — Pneumonia Detection Model Optimization

End-to-end optimization pipeline for a ResNet-18 chest X-ray pneumonia detector, from baseline profiling to hardware-accelerated ONNX production deployment. Developed as part of the **Udacity Machine Learning Engineer Nanodegree** course on model optimization for clinical AI.

---

## Project Overview

UdaciMed is a medical AI startup whose radiologists use a binary pneumonia detection model to triage chest X-rays. Before the model can be approved for production, it must meet **UdaciMed's Universal Performance Standard** on the reference hardware (NVIDIA T4 GPU):

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

## Dataset

**[PneumoniaMNIST](https://medmnist.com/)** — a binary classification subset of the MedMNIST benchmark derived from the Guangzhou Women and Children's Medical Center chest X-ray dataset.

| Split | Samples |
|-------|---------|
| Train | 4 708 |
| Validation | 524 |
| Test | 624 |

- **Classes:** `0 = Normal`, `1 = Pneumonia`
- **Resolution used:** 64 × 64 pixels (with optional 28 / 128 / 224 variants)
- **Preprocessing:** RGB conversion + ImageNet normalization for pretrained-model compatibility

---

## Repository Structure

```
UdaciMed/
├── notebooks/
│   ├── 01_baseline_analysis.ipynb       # Baseline profiling + optimization opportunity analysis
│   ├── 02_architecture_optimization.ipynb  # Apply optimizations, fine-tune, evaluate
│   └── 03_deployment_acceleration.ipynb # ONNX export + ONNX Runtime benchmarking
├── utils/
│   ├── model.py                  # ResNetBaseline definition, create_baseline_model, train
│   ├── data_loader.py            # PneumoniaMNIST loaders and balanced subset creation
│   ├── architecture_optimization.py  # All 7 optimization implementations
│   ├── evaluation.py             # ClassificationEvaluator, find_optimal_threshold
│   ├── profiling.py              # PerformanceProfiler (timing, FLOPs, memory)
│   ├── visualization.py          # Plotting helpers used across notebooks
│   └── __init__.py
├── results/
│   ├── best_baseline_model.pth   # Saved baseline weights
│   ├── optimized_model.pth       # Saved fine-tuned optimized weights (generated)
│   ├── baseline_results.pkl      # Baseline profiling dict (generated)
│   ├── optimization_results_*.pkl  # Per-experiment results (generated)
│   └── onnx_models/              # Exported ONNX files (generated)
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA 12.4 (for full benchmarking; CPU fallback available)
- NVIDIA T4 recommended for matching the production SLA targets

### Install

```bash
# Clone / download the repository
cd UdaciMed

# Using pip
pip install -r requirements.txt

# Or using uv
uv sync
```

> **Note:** `tensorrt-cu12==10.0.1` requires CUDA 12, cuDNN 8.9.2, and NVIDIA driver ≥ 550. Remove it from `requirements.txt` if running on CPU-only or a different CUDA version.

---

## Running the Notebooks

Open the notebooks in order — each notebook saves artefacts consumed by the next.

```bash
jupyter notebook
```

| Step | Notebook | What it produces |
|------|----------|-----------------|
| 1 | `01_baseline_analysis.ipynb` | `results/baseline_results.pkl`, `results/best_baseline_model.pth` |
| 2 | `02_architecture_optimization.ipynb` | `results/optimized_model.pth`, `results/optimization_results_<name>.pkl` |
| 3 | `03_deployment_acceleration.ipynb` | `results/onnx_models/udacimed_pneumonia_optimized.onnx`, benchmark tables |

> **GPU required for Notebook 3** — the ONNX Runtime CUDA EP and memory profiling require a CUDA-enabled GPU. Run Notebook 1 and most of Notebook 2 on CPU if needed.

---

## Implemented Optimizations (`utils/architecture_optimization.py`)

All seven optimization strategies are fully implemented:

| Optimization | Key Idea | Expected FLOP / Memory Gain |
|---|---|---|
| **Interpolation Removal** | Skip 64→224 bilinear upscaling; process at native resolution | ~55 % FLOP reduction (Amdahl-limited) |
| **Depthwise Separable Convolutions** | Replace 3×3 convs with depthwise + pointwise | ~70–80 % param reduction in converted layers |
| **Grouped Convolutions** | Split channels into parallel groups | ~50 % FLOP reduction per converted layer |
| **Inverted Residual Blocks** | MobileNetV2-style expand→depthwise→project | ~60 % FLOP reduction per block |
| **Low-Rank Factorization** | Decompose large linear layers via truncated SVD | ~75 % param reduction in FC layers |
| **Channel Optimization** | Channels-last memory layout + in-place ReLU | ~20 % speed gain on modern GPU |
| **Parameter Sharing** | Share weights across layers with identical shapes | Reduces effective model memory footprint |

The `create_optimized_model()` function applies selected optimizations in the correct dependency order:

```
interpolation_removal → inverted_residuals → depthwise_separable
  → grouped_conv → channel_optimization → lowrank_factorization → parameter_sharing
```

### Recommended Configuration

The best single-pass result combines **interpolation removal** (biggest FLOP win) with **depthwise separable convolutions** (biggest parameter win):

```python
OPTIMIZATION_CONFIG = {
    'interpolation_removal': True,
    'depthwise_separable': True,
    'grouped_conv': False,
    'channel_optimization': False,
    'inverted_residuals': False,
    'lowrank_factorization': False,
    'parameter_sharing': False,
    'memory_format': torch.preserve_format,
    'use_amp': False,
}
```

---

## Model Architecture

**Baseline:** ResNet-18 with an adaptive input wrapper that bilinearly upscales any input to 224×224 before passing through the ImageNet-pretrained backbone. The classification head replaces the original FC layer with `Dropout(0.2) → Linear(512, 2)`.

**Optimized:** The wrapper is replaced by a `NativeResolutionWrapper` that feeds 64×64 images directly to the backbone (no interpolation), and all eligible 3×3 convolutions are converted to depthwise separable equivalents.

---

## Training

Training uses `train_baseline_model()` from `utils/model.py`:

- **Optimizer:** AdamW with weight decay
- **Scheduler:** StepLR (gamma = 0.1)
- **Loss:** CrossEntropyLoss
- **Early stopping:** monitors validation accuracy with configurable patience
- **Gradient clipping:** max norm = 1.0

Fine-tuning the optimized model uses a low learning rate (1e-4) to preserve transferred weights from the baseline.

---

## Deployment Pipeline (Notebook 3)

```
Optimized PyTorch model
        │
        ▼  torch.onnx.export (FP16, dynamic axes)
ONNX model (.onnx)
        │
        ▼  ort.InferenceSession (CUDAExecutionProvider)
ONNX Runtime inference
        │
        ▼  benchmark_performance()
Latency / Throughput / Memory metrics
        │
        ▼  validate_clinical_performance()
Sensitivity check (threshold = 0.3)
```

**Hardware acceleration options:**

| EP | When to use |
|----|-------------|
| `CUDAExecutionProvider` | Standard GPU server; low overhead |
| `TensorRT EP` | Maximum GPU throughput; requires GPU-specific compile |
| `OpenVINOExecutionProvider` | Intel CPU workstations |
| `CPUExecutionProvider` | Fallback / dev machines |

---

## Results Summary

*(Values are representative; run on NVIDIA T4 with FP16 ONNX + dynamic batching)*

| Metric | Baseline | Optimized (Arch) | Deployed (ONNX + GPU) | Target |
|--------|----------|-----------------|----------------------|--------|
| Params | 11.2 M | ~3.5 M | — | — |
| FLOP reduction | 0 % | ~82 % | — | > 80 % |
| Peak memory | ~280 MB | ~90 MB | ~22 MB (file) | < 100 MB |
| Latency (BS=1) | ~8 ms | ~3 ms | < 1 ms | < 3 ms |
| Throughput | ~400 s/s | ~1 200 s/s | > 2 000 s/s | > 2 000 s/s |
| Sensitivity | ~97 % | ~98 % | ~98 % | > 98 % |

---

## Cross-Platform Deployment Guide

| Target | Recommended Stack | Key Notes |
|--------|------------------|-----------|
| GPU Server (cloud) | ONNX Runtime + TensorRT EP | 2–5× gain over CUDA EP; hardware lock-in |
| Triton multi-tenant | Triton + TensorRT backend | Dynamic batching; highest DevOps cost |
| Intel CPU workstation | ONNX Runtime + OpenVINO EP | Best for hospital workstations |
| iOS mobile | Core ML (via `coremltools`) | Uses Apple Neural Engine; iOS only |
| Android mobile | LiteRT (TFLite) | Smallest binary; widest Android HW support |
| Cross-platform mobile | ONNX Runtime Mobile | Single model; simpler CI/CD |

---

## License

This project is provided for educational purposes as part of the Udacity Machine Learning Engineer Nanodegree program.
