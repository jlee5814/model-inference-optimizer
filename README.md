# Model Inference Optimizer

A comprehensive benchmarking suite that demonstrates how different optimization techniques can dramatically improve ML model inference performance. This project takes a pre-trained model (ResNet50 by default) and converts it through progressively optimized formats, measuring the speedup at each step.

## The Problem

When you train a model in PyTorch, it's optimized for flexibility (easy to experiment with). But in production, you want **speed**. Different optimization tools can make the same model run **2-10x faster** without changing its predictions.

This project demonstrates exactly how much faster your models can run using production-grade optimization techniques.

## What This Project Does

Takes a pre-trained deep learning model and:

1. **Baseline**: Runs standard PyTorch inference
2. **TorchScript**: Optimizes using PyTorch's JIT compiler (removes Python overhead)
3. **ONNX Runtime**: Converts to ONNX format with cross-platform optimizations
4. **TensorRT** (optional): Uses NVIDIA's GPU optimizer for maximum performance

Then generates detailed benchmarks showing latency, throughput, and speedup for each format.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/model-inference-optimizer.git
cd model-inference-optimizer

# Install dependencies
pip install -r requirements.txt
```

### Run Benchmarks

```bash
# Run all benchmarks (uses config.yaml settings)
python benchmark_all.py

# Run with specific batch size
python benchmark_all.py --batch-size 1 4 8

# Use custom config
python benchmark_all.py --config my_config.yaml
```

### Visualize Results

```bash
# Generate comparison charts and tables
python visualize_results.py --input results/benchmark_results_TIMESTAMP.csv
```

## Example Results

Here's what you can expect to see:

```
Format          | Latency (ms) | Throughput (img/sec) | Speedup
----------------|--------------|----------------------|--------
PyTorch         | 15.2         | 65                   | 1.0x
TorchScript     | 12.1         | 82                   | 1.25x
ONNX Runtime    | 8.7          | 115                  | 1.75x
TensorRT (FP16) | 3.2          | 312                  | 4.75x
```

*Actual results vary based on hardware, model, and batch size*

## Project Structure

```
model-inference-optimizer/
├── benchmarks/              # Inference engines for each format
│   ├── pytorch_baseline.py  # Standard PyTorch
│   ├── torchscript_inference.py  # TorchScript (JIT)
│   ├── onnx_inference.py    # ONNX Runtime
│   └── tensorrt_inference.py  # TensorRT (optional)
├── models/                  # Saved optimized models
├── results/                 # Benchmark results and charts
├── benchmark_all.py         # Main orchestration script
├── visualize_results.py     # Visualization generator
├── config.yaml             # Configuration file
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Optimization Techniques Explained

### 1. PyTorch (Baseline)

Standard PyTorch inference with Python overhead. This is your starting point.

```python
model = torchvision.models.resnet50(pretrained=True)
output = model(image)
```

**Pros**: Flexible, easy to debug
**Cons**: Slower due to Python interpreter overhead

### 2. TorchScript

PyTorch's JIT (Just-In-Time) compiler removes Python overhead and optimizes the computation graph.

```python
scripted_model = torch.jit.trace(model, example_input)
scripted_model = torch.jit.optimize_for_inference(scripted_model)
```

**Optimizations**:
- Removes Python interpreter overhead
- Graph-level optimizations (operator fusion, constant folding)
- Can run without Python

**Expected speedup**: 1.2-1.5x

### 3. ONNX Runtime

Cross-platform inference engine with extensive optimizations.

```python
torch.onnx.export(model, example_input, "model.onnx")
session = ort.InferenceSession("model.onnx")
```

**Optimizations**:
- Platform-specific kernel optimizations
- Graph-level optimizations (layer fusion, constant folding)
- Memory layout optimizations
- Works across frameworks

**Expected speedup**: 1.5-2.5x

### 4. TensorRT (NVIDIA GPUs only)

NVIDIA's high-performance inference optimizer. Requires NVIDIA GPU and TensorRT installation.

```python
# Converts ONNX → TensorRT with FP16 precision
engine = build_engine_from_onnx("model.onnx", fp16_mode=True)
```

**Optimizations**:
- GPU kernel fusion (combines multiple operations)
- FP16/INT8 precision (faster with minimal accuracy loss)
- Layer-specific optimizations
- Memory optimization

**Expected speedup**: 3-10x (GPU only)

## Configuration

Edit `config.yaml` to customize benchmarking:

```yaml
model:
  name: "resnet50"  # or "resnet18", "mobilenet_v2"
  pretrained: true
  input_shape: [1, 3, 224, 224]

benchmark:
  warmup_iterations: 10
  benchmark_iterations: 100
  batch_sizes: [1, 4, 8, 16]
  device: "cuda"  # or "cpu"

optimization:
  enable_torchscript: true
  enable_onnx: true
  enable_tensorrt: false  # Requires NVIDIA GPU + TensorRT
```

## Hardware Requirements

### Minimum
- CPU: Any modern x86_64 processor
- RAM: 8GB
- OS: Linux, macOS, or Windows

### Recommended
- GPU: NVIDIA GPU with CUDA support (for TensorRT)
- RAM: 16GB
- CUDA: 11.x or later (for GPU acceleration)

### TensorRT Requirements (Optional)
- NVIDIA GPU (RTX 20xx series or newer recommended)
- CUDA Toolkit 11.x+
- cuDNN 8.x+
- TensorRT 8.x+

## Benchmarking Best Practices

1. **Warmup**: Always run warmup iterations (model loading, cache warming)
2. **Consistency**: Close other applications to reduce noise
3. **Multiple runs**: Run benchmarks multiple times for statistical significance
4. **Batch size**: Test multiple batch sizes (production workloads vary)
5. **Device**: Test on target hardware (development vs production)

## Use Cases

### Edge Deployment (Tesla/Automotive)
- Optimize for **latency** (single-image inference)
- Use **TorchScript** or **ONNX Runtime** (CPU/edge device friendly)
- Consider **quantization** for further speedup

### Cloud Inference (xAI/LLM Serving)
- Optimize for **throughput** (batch processing)
- Use **TensorRT** or **ONNX Runtime** with GPU
- Maximize batch size for efficiency

### Mobile Deployment
- Use **ONNX Runtime** with mobile optimizations
- Consider **quantization** (INT8) for smaller models
- Test on actual mobile hardware

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `config.yaml`
- Use CPU device: `device: "cpu"`

### TensorRT not available
- TensorRT is optional. Set `enable_tensorrt: false` in config
- Install TensorRT: Follow [NVIDIA's installation guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/)

### ONNX conversion fails
- Some PyTorch operations aren't supported in ONNX
- Try using `torch.jit.trace` instead of `torch.jit.script`
- Check ONNX opset version compatibility

## Further Optimizations

This project demonstrates the basics. Advanced techniques include:

- **Quantization**: INT8/FP16 precision (2-4x speedup)
- **Pruning**: Remove unimportant weights (smaller, faster models)
- **Distillation**: Train smaller model to mimic larger one
- **Dynamic batching**: Combine multiple requests for efficiency
- **Model compilation**: XLA, TVM for specialized hardware

## Why This Matters

**For Tesla/Automotive**:
- Cars need real-time inference on edge devices
- Every millisecond matters for safety-critical decisions
- Optimized models enable deployment on cost-effective hardware

**For xAI/LLM Companies**:
- Serving costs scale with latency × requests
- 2x speedup = 50% cost reduction
- Enables serving more users with same infrastructure

**For Any ML Engineer**:
- Production deployment isn't just about training
- Optimization is a critical skill for ML engineering
- Understanding performance trade-offs is essential

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new optimizations
4. Submit a pull request

## Acknowledgments

- PyTorch Team for TorchScript
- Microsoft for ONNX Runtime
- NVIDIA for TensorRT
- HuggingFace for optimization inspiration

## References

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/tune-performance.html)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)

---

Built to demonstrate production ML optimization techniques for edge deployment and cloud inference.
