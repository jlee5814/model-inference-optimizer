# Quick Start Guide

Get your first benchmark running in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch, ONNX Runtime, and visualization tools. TensorRT is optional.

## Step 2: Run Your First Benchmark

### Option A: CPU-Only (Works Anywhere)

Edit `config.yaml` and set:
```yaml
benchmark:
  device: "cpu"
  batch_sizes: [1]  # Start with batch size 1

optimization:
  enable_tensorrt: false  # Disable TensorRT on CPU
```

Then run:
```bash
python benchmark_all.py --batch-size 1
```

### Option B: GPU (NVIDIA CUDA)

If you have an NVIDIA GPU with CUDA:

```bash
# Default config uses CUDA
python benchmark_all.py --batch-size 1
```

## Step 3: View Results

The script will output results to the console and save them to `results/`.

Look for output like:
```
BENCHMARK SUMMARY
================================================================================
format        batch_size  mean_latency_ms  throughput_samples_per_sec
PyTorch       1          12.34             81.0
TorchScript   1          10.21             98.0
ONNX Runtime  1          7.89              126.7
```

## Step 4: Visualize Results

```bash
# Find your results file (timestamp will vary)
ls results/

# Generate charts
python visualize_results.py --input results/benchmark_results_20240101_120000.csv
```

Charts will be saved to `results/charts/`:
- `latency_comparison.png` - Latency across formats
- `throughput_comparison.png` - Throughput comparison
- `speedup_comparison.png` - Speedup vs PyTorch baseline

## What's Happening?

1. **PyTorch**: Loads ResNet50 and runs baseline inference
2. **TorchScript**: Converts model to TorchScript and benchmarks
3. **ONNX Runtime**: Exports to ONNX format and benchmarks
4. **TensorRT** (optional): Builds optimized GPU engine

Each format is tested with warmup iterations (to load caches) followed by benchmark iterations.

## Common Issues

### "CUDA out of memory"
- Use CPU: `python benchmark_all.py --batch-size 1` with `device: "cpu"` in config
- Or reduce batch size

### "TensorRT not found"
- It's optional! Set `enable_tensorrt: false` in config.yaml

### "Slow on first run"
- First run downloads ResNet50 weights (~100MB)
- Models are cached for future runs

## Next Steps

1. **Try different batch sizes**: `python benchmark_all.py --batch-size 1 4 8`
2. **Test different models**: Edit `config.yaml` and change `model.name` to `"resnet18"` or `"mobilenet_v2"`
3. **Run on your own model**: Modify the inference scripts to load your model
4. **Add quantization**: Implement INT8/FP16 quantization for further speedup

## Understanding the Output

### Latency (ms)
Time to process one batch. **Lower is better**.
- Important for: Real-time applications, edge devices

### Throughput (samples/sec)
How many samples processed per second. **Higher is better**.
- Important for: Batch processing, cloud inference

### Speedup
How much faster vs PyTorch baseline. **Higher is better**.
- 2.0x = twice as fast
- 0.5x = twice as slow (shouldn't happen!)

## Minimal Example (Copy-Paste)

```bash
# Install
pip install torch torchvision onnx onnxruntime numpy matplotlib pandas tabulate

# Run
python benchmark_all.py --batch-size 1

# Visualize (use your actual filename)
python visualize_results.py --input results/benchmark_results_*.csv
```

That's it! You now have hard data on how much faster optimized inference can be.

## Tips for Best Results

1. **Close other applications** - Reduces noise in measurements
2. **Run multiple times** - First run includes model download
3. **Test on target hardware** - Results vary by CPU/GPU
4. **Use realistic batch sizes** - Match your production workload

Happy optimizing! 🚀
