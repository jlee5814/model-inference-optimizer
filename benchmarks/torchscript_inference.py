"""
TorchScript Inference
Optimized PyTorch inference using TorchScript (JIT compilation)
"""

import os
import torch
import time
import numpy as np
from typing import Tuple, Dict
from ultralytics import YOLO

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

    def convert_to_torchscript(self, pretrained=True, save_path="models/torchscript_model.pt"):
        """
        Convert YOLOv8 model to TorchScript format using Ultralytics exporter.
        """
        if "yolov8" in self.model_name:
            print(f"Converting {self.model_name} to TorchScript using Ultralytics export...")
            yolo = YOLO(f"{self.model_name}.pt")
            yolo.export(format="torchscript", dynamic=False, simplify=True, imgsz=640)
            print("✅ YOLOv8 TorchScript model exported successfully.")

            # Load the exported TorchScript model back into memory for benchmarking
            torchscript_path = f"{self.model_name}.torchscript"
            if not os.path.exists(torchscript_path):
                torchscript_path = os.path.join(os.getcwd(), torchscript_path)

            self.scripted_model = torch.jit.load(torchscript_path, map_location=self.device)
            self.scripted_model.eval()
            print(f"✅ Loaded TorchScript model from {torchscript_path}")
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

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
