"""
TorchScript Inference
Optimized PyTorch inference using TorchScript (JIT compilation)
"""

import torch
import torchvision.models as models
import time
import numpy as np
from typing import Tuple, Dict
import os


class TorchScriptInference:
    """TorchScript optimized inference engine"""

    def __init__(self, model_name: str = "resnet50", device: str = "cuda"):
        """
        Initialize TorchScript model

        Args:
            model_name: Name of the torchvision model
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scripted_model = None

    def convert_to_torchscript(self, pretrained: bool = True, save_path: str = None):
        """
        Convert PyTorch model to TorchScript

        Args:
            pretrained: Whether to use pretrained weights
            save_path: Path to save the TorchScript model
        """
        print(f"Converting {self.model_name} to TorchScript...")

        # Load original PyTorch model
        if self.model_name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
        elif self.model_name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
        elif self.model_name == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        model.to(self.device)
        model.eval()

        # Create example input for tracing
        example_input = torch.randn(1, 3, 224, 224).to(self.device)

        # Convert to TorchScript using trace
        # Note: We use trace instead of script for better compatibility with complex models
        with torch.no_grad():
            self.scripted_model = torch.jit.trace(model, example_input)

        # Optimize the TorchScript model
        self.scripted_model = torch.jit.optimize_for_inference(self.scripted_model)

        print(f"✓ Model converted to TorchScript")

        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.jit.save(self.scripted_model, save_path)
            print(f"✓ TorchScript model saved to {save_path}")

        return self

    def load_model(self, model_path: str):
        """Load a pre-converted TorchScript model"""
        print(f"Loading TorchScript model from {model_path}...")
        self.scripted_model = torch.jit.load(model_path, map_location=self.device)
        self.scripted_model.eval()
        print(f"✓ TorchScript model loaded on {self.device}")
        return self

    def warmup(self, input_shape: Tuple[int, int, int, int], iterations: int = 10):
        """
        Warmup the model with dummy inputs

        Args:
            input_shape: Shape of input tensor (batch, channels, height, width)
            iterations: Number of warmup iterations
        """
        print(f"Warming up for {iterations} iterations...")
        dummy_input = torch.randn(input_shape).to(self.device)

        with torch.no_grad():
            for _ in range(iterations):
                _ = self.scripted_model(dummy_input)

        # Synchronize CUDA if using GPU
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        print("✓ Warmup complete")

    def benchmark(self, input_shape: Tuple[int, int, int, int], iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark the TorchScript model

        Args:
            input_shape: Shape of input tensor
            iterations: Number of benchmark iterations

        Returns:
            Dictionary with performance metrics
        """
        print(f"\nBenchmarking TorchScript (batch_size={input_shape[0]})...")

        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)

        latencies = []

        with torch.no_grad():
            for i in range(iterations):
                # Start timing
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

                # Inference
                output = self.scripted_model(dummy_input)

                # End timing
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.perf_counter()

                latencies.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        latencies = np.array(latencies)
        batch_size = input_shape[0]

        results = {
            "format": "TorchScript",
            "device": str(self.device),
            "batch_size": batch_size,
            "mean_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "throughput_samples_per_sec": 1000 * batch_size / np.mean(latencies),
        }

        print(f"  Mean Latency: {results['mean_latency_ms']:.2f} ms")
        print(f"  Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")

        return results


if __name__ == "__main__":
    # Quick test
    inference = TorchScriptInference(model_name="resnet50", device="cuda")
    inference.convert_to_torchscript(pretrained=True, save_path="models/resnet50_torchscript.pt")
    inference.warmup(input_shape=(1, 3, 224, 224), iterations=10)
    results = inference.benchmark(input_shape=(1, 3, 224, 224), iterations=100)

    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")
