"""
Simple Benchmark Example

This is a minimal example showing how to benchmark a single optimization format.
Perfect for understanding the basics before running the full benchmark suite.

Usage:
    python examples/simple_benchmark.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.pytorch_baseline import PyTorchInference
from benchmarks.torchscript_inference import TorchScriptInference
from benchmarks.onnx_inference import ONNXInference


def simple_pytorch_example():
    """Run a simple PyTorch benchmark"""
    print("\n" + "="*80)
    print("EXAMPLE 1: PyTorch Baseline")
    print("="*80)

    # Create inference engine
    inference = PyTorchInference(model_name="resnet50", device="cpu")

    # Load model
    inference.load_model(pretrained=True)

    # Warmup
    inference.warmup(input_shape=(1, 3, 224, 224), iterations=5)

    # Benchmark
    results = inference.benchmark(input_shape=(1, 3, 224, 224), iterations=20)

    print("\nResults:")
    print(f"  Mean Latency: {results['mean_latency_ms']:.2f} ms")
    print(f"  Throughput: {results['throughput_samples_per_sec']:.1f} images/sec")


def simple_torchscript_example():
    """Run a simple TorchScript benchmark"""
    print("\n" + "="*80)
    print("EXAMPLE 2: TorchScript Optimization")
    print("="*80)

    # Create inference engine
    inference = TorchScriptInference(model_name="resnet50", device="cpu")

    # Convert to TorchScript
    inference.convert_to_torchscript(
        pretrained=True,
        save_path="models/resnet50_torchscript.pt"
    )

    # Warmup
    inference.warmup(input_shape=(1, 3, 224, 224), iterations=5)

    # Benchmark
    results = inference.benchmark(input_shape=(1, 3, 224, 224), iterations=20)

    print("\nResults:")
    print(f"  Mean Latency: {results['mean_latency_ms']:.2f} ms")
    print(f"  Throughput: {results['throughput_samples_per_sec']:.1f} images/sec")


def simple_onnx_example():
    """Run a simple ONNX Runtime benchmark"""
    print("\n" + "="*80)
    print("EXAMPLE 3: ONNX Runtime Optimization")
    print("="*80)

    # Create inference engine
    inference = ONNXInference(model_name="resnet50", device="cpu")

    # Convert to ONNX
    onnx_path = inference.convert_to_onnx(
        pretrained=True,
        save_path="models/resnet50.onnx"
    )

    # Load ONNX model
    inference.load_model(onnx_path)

    # Warmup
    inference.warmup(input_shape=(1, 3, 224, 224), iterations=5)

    # Benchmark
    results = inference.benchmark(input_shape=(1, 3, 224, 224), iterations=20)

    print("\nResults:")
    print(f"  Mean Latency: {results['mean_latency_ms']:.2f} ms")
    print(f"  Throughput: {results['throughput_samples_per_sec']:.1f} images/sec")


def compare_all():
    """Run all examples and compare results"""
    print("\n" + "#"*80)
    print("RUNNING ALL EXAMPLES")
    print("#"*80)

    all_results = []

    # PyTorch
    print("\nRunning PyTorch baseline...")
    inference = PyTorchInference(model_name="resnet50", device="cpu")
    inference.load_model(pretrained=True)
    inference.warmup(input_shape=(1, 3, 224, 224), iterations=5)
    results = inference.benchmark(input_shape=(1, 3, 224, 224), iterations=20)
    all_results.append(results)

    # TorchScript
    print("\nRunning TorchScript...")
    inference = TorchScriptInference(model_name="resnet50", device="cpu")
    if not os.path.exists("models/resnet50_torchscript.pt"):
        inference.convert_to_torchscript(
            pretrained=True,
            save_path="models/resnet50_torchscript.pt"
        )
    else:
        inference.load_model("models/resnet50_torchscript.pt")
    inference.warmup(input_shape=(1, 3, 224, 224), iterations=5)
    results = inference.benchmark(input_shape=(1, 3, 224, 224), iterations=20)
    all_results.append(results)

    # ONNX Runtime
    print("\nRunning ONNX Runtime...")
    inference = ONNXInference(model_name="resnet50", device="cpu")
    if not os.path.exists("models/resnet50.onnx"):
        inference.convert_to_onnx(
            pretrained=True,
            save_path="models/resnet50.onnx"
        )
    inference.load_model("models/resnet50.onnx")
    inference.warmup(input_shape=(1, 3, 224, 224), iterations=5)
    results = inference.benchmark(input_shape=(1, 3, 224, 224), iterations=20)
    all_results.append(results)

    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    pytorch_latency = all_results[0]['mean_latency_ms']

    print(f"\n{'Format':<20} {'Latency (ms)':<15} {'Throughput (img/s)':<20} {'Speedup':<10}")
    print("-" * 80)

    for result in all_results:
        speedup = pytorch_latency / result['mean_latency_ms']
        print(f"{result['format']:<20} {result['mean_latency_ms']:<15.2f} "
              f"{result['throughput_samples_per_sec']:<20.1f} {speedup:<10.2f}x")

    print("\n" + "="*80)
    print("Key Takeaways:")
    print("="*80)
    print("1. TorchScript removes Python overhead (1.2-1.5x speedup)")
    print("2. ONNX Runtime adds platform-specific optimizations (1.5-2.5x speedup)")
    print("3. Results vary by hardware - test on your target device!")
    print("4. For GPU: Re-run with device='cuda' for even better speedups")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple benchmark examples")
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3],
        help='Run specific example (1=PyTorch, 2=TorchScript, 3=ONNX)'
    )

    args = parser.parse_args()

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    if args.example == 1:
        simple_pytorch_example()
    elif args.example == 2:
        simple_torchscript_example()
    elif args.example == 3:
        simple_onnx_example()
    else:
        # Run all examples and compare
        compare_all()

    print("\n✓ Done! Try running the full benchmark suite with: python benchmark_all.py")
