"""
PyTorch Baseline Inference
Standard PyTorch model inference without optimizations
"""

import torch
import torchvision.models as models
import time
import numpy as np
from typing import Tuple, Dict


class PyTorchInference:
    """Baseline PyTorch inference engine"""

    def __init__(self, model_name: str = "resnet50", device: str = "cuda"):
        """
        Initialize PyTorch model

        Args:
            model_name: Name of the torchvision model
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self, pretrained: bool = True):
        """Load the PyTorch model"""
        print(f"Loading {self.model_name} model (PyTorch baseline)...")

        # Load pre-trained model
        if self.model_name == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
        elif self.model_name == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
        elif self.model_name == "mobilenet_v2":
            self.model = models.mobilenet_v2(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        # Move to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        print(f"✓ Model loaded on {self.device}")
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
                _ = self.model(dummy_input)

        # Synchronize CUDA if using GPU
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        print("✓ Warmup complete")

    def benchmark(self, input_shape: Tuple[int, int, int, int], iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark the model

        Args:
            input_shape: Shape of input tensor
            iterations: Number of benchmark iterations

        Returns:
            Dictionary with performance metrics
        """
        print(f"\nBenchmarking PyTorch (batch_size={input_shape[0]})...")

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
                output = self.model(dummy_input)

                # End timing
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.perf_counter()

                latencies.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        latencies = np.array(latencies)
        batch_size = input_shape[0]

        results = {
            "format": "PyTorch",
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

    def save_model(self, save_path: str):
        """Save the PyTorch model"""
        torch.save(self.model.state_dict(), save_path)
        print(f"✓ Model saved to {save_path}")


if __name__ == "__main__":
    # Quick test
    inference = PyTorchInference(model_name="resnet50", device="cuda")
    inference.load_model(pretrained=True)
    inference.warmup(input_shape=(1, 3, 224, 224), iterations=10)
    results = inference.benchmark(input_shape=(1, 3, 224, 224), iterations=100)

    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")
