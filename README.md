# Model Inference Optimizer

A benchmarking and experimentation suite for understanding **ML model inference performance tradeoffs** across common production optimization paths.

This project measures how the same trained model behaves when moved from **research-friendly execution** to **production-oriented inference engines**, focusing on latency, throughput, and operational constraints rather than model accuracy.

---

## Motivation

Models trained in PyTorch are optimized for **flexibility and iteration speed**, not deployment performance.  
In production, inference performance directly impacts:

- Latency-sensitive systems (real-time inference)
- Infrastructure cost (GPU/CPU utilization)
- Scalability under load

This project explores **how much performance can be gained** by applying standard inference optimizations, and **what tradeoffs are introduced at each step**.

---

## What This Project Does

Given a pre-trained model (ResNet50 by default), the system:

1. Runs baseline inference in PyTorch
2. Converts the model through progressively optimized formats
3. Benchmarks each format under consistent conditions
4. Compares latency, throughput, and speedup

### Optimization Path

- **PyTorch (Baseline)** – Flexible, debuggable, slower
- **TorchScript** – Removes Python overhead via JIT compilation
- **ONNX Runtime** – Graph-level and kernel optimizations
- **TensorRT (optional)** – GPU-specific fusion and precision optimizations

The result is a **side-by-side comparison** of inference behavior across engines.

---

## Example Results (Illustrative)

| Format          | Latency (ms) | Throughput (img/sec) | Speedup |
|-----------------|--------------|----------------------|---------|
| PyTorch         | 15.2         | 65                   | 1.0×    |
| TorchScript     | 12.1         | 82                   | 1.25×   |
| ONNX Runtime    | 8.7          | 115                  | 1.75×   |
| TensorRT (FP16) | 3.2          | 312                  | 4.75×   |

*Actual results vary by hardware, batch size, and configuration.*

---

## Repository Structure

```
model-inference-optimizer/
├── benchmarks/              # Inference backends (PyTorch, TorchScript, ONNX, TensorRT)
├── models/                  # Saved optimized models
├── results/                 # Benchmark outputs and visualizations
├── examples/                # Minimal usage examples
├── benchmark_all.py         # Benchmark orchestration
├── visualize_results.py     # Result visualization
├── config.yaml              # Benchmark configuration
├── QUICKSTART.md            # Setup and execution details
└── README.md                # This file
```

---

## Running the Benchmarks

Basic usage:

```bash
python benchmark_all.py
```

Custom batch sizes or configs:

```bash
python benchmark_all.py --batch-size 1 4 8
python benchmark_all.py --config my_config.yaml
```

Visualization:

```bash
python visualize_results.py --input results/benchmark_results.csv
```

Detailed setup instructions are in **`QUICKSTART.md`**.

---

## Design Focus

This project prioritizes:

- **Measurement over abstraction**
- **Explicit tradeoffs over automation**
- **Inference behavior, not training performance**

It is intentionally minimal to keep system behavior **inspectable and reproducible**.

---

## Known Limitations

- Not designed for distributed or multi-GPU benchmarking
- No autoscaling or serving layer included
- TensorRT support requires NVIDIA GPU + compatible drivers
- Benchmarks focus on inference speed, not accuracy drift

These constraints are intentional to isolate inference-engine effects.

---

## Use Cases

- Evaluating inference optimization paths before production deployment
- Understanding latency vs throughput tradeoffs
- Comparing CPU vs GPU inference behavior
- Learning how model formats affect execution characteristics

---

## Scope Clarification

This project is **no
