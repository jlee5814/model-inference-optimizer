"""
ONNX Runtime Inference
Cross-platform optimized inference using ONNX Runtime
"""

import torch
import torchvision.models as models
import onnx
import onnxruntime as ort
import time
import numpy as np
from typing import Tuple, Dict
import os


class ONNXInference:
    """ONNX Runtime optimized inference engine"""

    def __init__(self, model_name: str = "resnet50", device: str = "cuda"):
        """
        Initialize ONNX Runtime session

        Args:
            model_name: Name of the torchvision model
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.session = None
        self.input_name = None
        self.output_name = None

    def convert_to_onnx(
        self,
        pretrained: bool = True,
        save_path: str = None,
        opset_version: int = 14,
        optimize: bool = True
    ):
        """
        Convert YOLOv8 model to ONNX format using Ultralytics exporter.
        Falls back to torchvision models if YOLOv8 is not detected.
        """
        import os
        import onnx
        from ultralytics import YOLO

        print(f"Converting {self.model_name} to ONNX...")

        # YOLOv8 export path
        if "yolov8" in self.model_name.lower():
            print(f"Detected YOLOv8 model: {self.model_name}")
            yolo = YOLO(f"{self.model_name}.pt")
            yolo.export(format="onnx", dynamic=False, simplify=True, imgsz=640)
            print("✅ YOLOv8 ONNX model exported successfully.")

            onnx_path = f"{self.model_name}.onnx"
            if not os.path.exists(onnx_path):
                onnx_path = os.path.join(os.getcwd(), onnx_path)

            # Load & verify exported model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print(f"✅ Verified ONNX model: {onnx_path}")

            self.onnx_model_path = onnx_path
            return onnx_path

        # Fallback: standard torchvision model
        import torch
        import torchvision.models as models

        if self.model_name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
        elif self.model_name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
        elif self.model_name == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        if save_path is None:
            save_path = f"models/{self.model_name}.onnx"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        print(f"✅ Model exported to ONNX format: {save_path}")
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verified")

        return save_path

    def load_model(self, model_path: str):
        """
        Load ONNX model and create runtime session

        Args:
            model_path: Path to the ONNX model
        """
        print(f"Loading ONNX model from {model_path}...")

        # Set up providers (execution providers)
        providers = []
        if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
            print("  Using CUDA Execution Provider")
        providers.append("CPUExecutionProvider")

        # Session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1  # Number of threads for parallel ops

        # Create inference session
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"✓ ONNX Runtime session created")
        print(f"  Providers: {self.session.get_providers()}")

        return self

    def warmup(self, input_shape: Tuple[int, int, int, int], iterations: int = 10):
        """
        Warmup the model with dummy inputs

        Args:
            input_shape: Shape of input tensor (batch, channels, height, width)
            iterations: Number of warmup iterations
        """
        print(f"Warming up for {iterations} iterations...")
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        for _ in range(iterations):
            _ = self.session.run([self.output_name], {self.input_name: dummy_input})

        print("✓ Warmup complete")

    def benchmark(self, input_shape: Tuple[int, int, int, int], iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark the ONNX model

        Args:
            input_shape: Shape of input tensor
            iterations: Number of benchmark iterations

        Returns:
            Dictionary with performance metrics
        """
        print(f"\nBenchmarking ONNX Runtime (batch_size={input_shape[0]})...")

        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        latencies = []

        for i in range(iterations):
            # Start timing
            start_time = time.perf_counter()

            # Inference
            output = self.session.run([self.output_name], {self.input_name: dummy_input})

            # End timing
            end_time = time.perf_counter()

            latencies.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        latencies = np.array(latencies)
        batch_size = input_shape[0]

        results = {
            "format": "ONNX Runtime",
            "device": self.session.get_providers()[0],
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
    inference = ONNXInference(model_name="resnet50", device="cuda")

    # Convert and save
    onnx_path = inference.convert_to_onnx(pretrained=True, save_path="models/resnet50.onnx")

    # Load and benchmark
    inference.load_model(onnx_path)
    inference.warmup(input_shape=(1, 3, 224, 224), iterations=10)
    results = inference.benchmark(input_shape=(1, 3, 224, 224), iterations=100)

    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")
