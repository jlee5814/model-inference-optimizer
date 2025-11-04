"""

Interactive Walkthrough Demo

 

This script demonstrates each optimization step with detailed explanations.

You can run this to see the Model Inference Optimizer in action!

 

Usage:

    python demo_walkthrough.py

"""

 

import torch

import torchvision.models as models

import time

import numpy as np

import os

 

print("="*80)

print("MODEL INFERENCE OPTIMIZER - INTERACTIVE WALKTHROUGH")

print("="*80)

print("\nThis demo will show you how different optimizations speed up inference.")

print("We'll use ResNet50 (a popular image classification model) as an example.")

print("\nNote: First run will download model weights (~100MB)")

 

# Create models directory

os.makedirs("models", exist_ok=True)

 

# Check device

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n✓ Using device: {device}")

if device == "cpu":

    print("  (For faster results, use an NVIDIA GPU with CUDA)")

 

input_shape = (1, 3, 224, 224)  # 1 image, 3 RGB channels, 224x224 pixels

warmup_iters = 5

benchmark_iters = 20

 

print("\n" + "="*80)

print("STEP 1: PyTorch Baseline (Standard Inference)")

print("="*80)

print("\nWhat we're doing:")

print("  - Loading pre-trained ResNet50")

print("  - Running standard PyTorch inference")

print("  - This is your baseline (1.0x speed)")

 

# Load model

print("\nLoading model...")

model = models.resnet50(pretrained=True)

model.to(device)

model.eval()

print("✓ Model loaded")

 

# Create dummy input (random tensor)

dummy_input = torch.randn(input_shape).to(device)

print(f"✓ Created dummy input: {dummy_input.shape}")

 

# Warmup

print(f"\nWarming up ({warmup_iters} iterations)...")

with torch.no_grad():

    for _ in range(warmup_iters):

        _ = model(dummy_input)

if device == "cuda":

    torch.cuda.synchronize()

print("✓ Warmup complete")

 

# Benchmark

print(f"\nBenchmarking ({benchmark_iters} iterations)...")

latencies_pytorch = []

with torch.no_grad():

    for i in range(benchmark_iters):

        if device == "cuda":

            torch.cuda.synchronize()

        start = time.perf_counter()

 

        output = model(dummy_input)

 

        if device == "cuda":

            torch.cuda.synchronize()

        end = time.perf_counter()

 

        latencies_pytorch.append((end - start) * 1000)

 

pytorch_mean = np.mean(latencies_pytorch)

pytorch_throughput = 1000 / pytorch_mean

 

print(f"\n{'='*40}")

print(f"PyTorch Results:")

print(f"  Mean Latency: {pytorch_mean:.2f} ms")

print(f"  Throughput: {pytorch_throughput:.1f} images/sec")

print(f"  Speedup: 1.00x (baseline)")

print(f"{'='*40}")

 

print("\n" + "="*80)

print("STEP 2: TorchScript (JIT Compilation)")

print("="*80)

print("\nWhat we're doing:")

print("  - Converting PyTorch model to TorchScript")

print("  - TorchScript removes Python interpreter overhead")

print("  - Performs graph optimizations (operator fusion, etc.)")

print("  - Expected speedup: 1.2-1.5x")

 

# Convert to TorchScript

print("\nConverting to TorchScript...")

scripted_model = torch.jit.trace(model, dummy_input)

scripted_model = torch.jit.optimize_for_inference(scripted_model)

print("✓ TorchScript conversion complete")

 

# Save (optional)

torch.jit.save(scripted_model, "models/demo_torchscript.pt")

print("✓ Saved to models/demo_torchscript.pt")

 

# Warmup

print(f"\nWarming up ({warmup_iters} iterations)...")

with torch.no_grad():

    for _ in range(warmup_iters):

        _ = scripted_model(dummy_input)

if device == "cuda":

    torch.cuda.synchronize()

print("✓ Warmup complete")

 

# Benchmark

print(f"\nBenchmarking ({benchmark_iters} iterations)...")

latencies_torchscript = []

with torch.no_grad():

    for i in range(benchmark_iters):

        if device == "cuda":

            torch.cuda.synchronize()

        start = time.perf_counter()

 

        output = scripted_model(dummy_input)

 

        if device == "cuda":

            torch.cuda.synchronize()

        end = time.perf_counter()

 

        latencies_torchscript.append((end - start) * 1000)

 

torchscript_mean = np.mean(latencies_torchscript)

torchscript_throughput = 1000 / torchscript_mean

torchscript_speedup = pytorch_mean / torchscript_mean

 

print(f"\n{'='*40}")

print(f"TorchScript Results:")

print(f"  Mean Latency: {torchscript_mean:.2f} ms")

print(f"  Throughput: {torchscript_throughput:.1f} images/sec")

print(f"  Speedup: {torchscript_speedup:.2f}x")

print(f"{'='*40}")

 

print("\n" + "="*80)

print("STEP 3: ONNX Runtime (Cross-Platform Optimization)")

print("="*80)

print("\nWhat we're doing:")

print("  - Exporting PyTorch model to ONNX format")

print("  - ONNX Runtime applies platform-specific optimizations")

print("  - Works across frameworks (PyTorch, TensorFlow, etc.)")

print("  - Expected speedup: 1.5-2.5x")

 

try:

    import onnx

    import onnxruntime as ort

 

    # Export to ONNX

    print("\nExporting to ONNX...")

    onnx_path = "models/demo_resnet50.onnx"

    torch.onnx.export(

        model,

        dummy_input.cpu(),

        onnx_path,

        export_params=True,

        opset_version=14,

        do_constant_folding=True,

        input_names=['input'],

        output_names=['output'],

        dynamic_axes={

            'input': {0: 'batch_size'},

            'output': {0: 'batch_size'}

        }

    )

    print(f"✓ Exported to {onnx_path}")

 

    # Verify

    onnx_model = onnx.load(onnx_path)

    onnx.checker.check_model(onnx_model)

    print("✓ ONNX model verified")

 

    # Create ONNX Runtime session

    print("\nCreating ONNX Runtime session...")

    providers = []

    if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():

        providers.append("CUDAExecutionProvider")

    providers.append("CPUExecutionProvider")

 

    sess_options = ort.SessionOptions()

    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

 

    session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)

    print(f"✓ Session created with providers: {session.get_providers()}")

 

    # Get input/output names

    input_name = session.get_inputs()[0].name

    output_name = session.get_outputs()[0].name

 

    # Prepare input

    dummy_input_np = np.random.randn(*input_shape).astype(np.float32)

 

    # Warmup

    print(f"\nWarming up ({warmup_iters} iterations)...")

    for _ in range(warmup_iters):

        _ = session.run([output_name], {input_name: dummy_input_np})

    print("✓ Warmup complete")

 

    # Benchmark

    print(f"\nBenchmarking ({benchmark_iters} iterations)...")

    latencies_onnx = []

    for i in range(benchmark_iters):

        start = time.perf_counter()

 

        output = session.run([output_name], {input_name: dummy_input_np})

 

        end = time.perf_counter()

 

        latencies_onnx.append((end - start) * 1000)

 

    onnx_mean = np.mean(latencies_onnx)

    onnx_throughput = 1000 / onnx_mean

    onnx_speedup = pytorch_mean / onnx_mean

 

    print(f"\n{'='*40}")

    print(f"ONNX Runtime Results:")

    print(f"  Mean Latency: {onnx_mean:.2f} ms")

    print(f"  Throughput: {onnx_throughput:.1f} images/sec")

    print(f"  Speedup: {onnx_speedup:.2f}x")

    print(f"{'='*40}")

 

    onnx_available = True

 

except ImportError:

    print("\n⚠ ONNX Runtime not installed. Skipping ONNX benchmark.")

    print("  Install with: pip install onnx onnxruntime")

    onnx_available = False

 

print("\n" + "="*80)

print("FINAL COMPARISON")

print("="*80)

 

print(f"\n{'Format':<20} {'Latency (ms)':<15} {'Throughput (img/s)':<20} {'Speedup':<10}")

print("-" * 80)

print(f"{'PyTorch':<20} {pytorch_mean:<15.2f} {pytorch_throughput:<20.1f} {'1.00x':<10}")

print(f"{'TorchScript':<20} {torchscript_mean:<15.2f} {torchscript_throughput:<20.1f} {f'{torchscript_speedup:.2f}x':<10}")

if onnx_available:

    print(f"{'ONNX Runtime':<20} {onnx_mean:<15.2f} {onnx_throughput:<20.1f} {f'{onnx_speedup:.2f}x':<10}")

 

print("\n" + "="*80)

print("KEY TAKEAWAYS")

print("="*80)

print("\n1. TorchScript removes Python overhead")

print("   - Uses JIT compilation")

print("   - Graph-level optimizations")

print(f"   - Achieved: {torchscript_speedup:.2f}x speedup in this demo")

 

if onnx_available:

    print("\n2. ONNX Runtime adds platform-specific optimizations")

    print("   - Kernel fusion and memory optimization")

    print("   - Cross-platform compatibility")

    print(f"   - Achieved: {onnx_speedup:.2f}x speedup in this demo")

 

print("\n3. In production, you'd also consider:")

print("   - TensorRT (3-10x on NVIDIA GPUs)")

print("   - Quantization (INT8/FP16 for 2-4x additional speedup)")

print("   - Model pruning and distillation")

 

print("\n4. Why this matters:")

print("   - 2x speedup = 50% cost reduction in cloud inference")

print("   - Faster inference = better user experience")

print("   - Enables deployment on edge devices (cars, phones)")

 

print("\n" + "="*80)

print("NEXT STEPS")

print("="*80)

print("\n1. Run the full benchmark suite:")

print("   python benchmark_all.py --batch-size 1")

print("\n2. Try different models:")

print("   Edit config.yaml and change model name to 'resnet18' or 'mobilenet_v2'")

print("\n3. Visualize results:")

print("   python visualize_results.py --input results/benchmark_results_*.csv")

print("\n4. Experiment with batch sizes:")

print("   python benchmark_all.py --batch-size 1 4 8 16")

 

print("\n✓ Demo complete! Models saved to models/ directory")

print("="*80)
