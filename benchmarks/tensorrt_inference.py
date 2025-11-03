"""
TensorRT Inference (Optional)
High-performance NVIDIA GPU inference using TensorRT

Note: Requires NVIDIA GPU and TensorRT installation
"""

import time
import numpy as np
from typing import Tuple, Dict
import os

# TensorRT imports (will fail gracefully if not installed)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("Warning: TensorRT not available. Install with: pip install nvidia-tensorrt pycuda")


class TensorRTInference:
    """TensorRT optimized inference engine (NVIDIA GPUs only)"""

    def __init__(self, model_name: str = "resnet50"):
        """
        Initialize TensorRT inference

        Args:
            model_name: Name of the model
        """
        if not TRT_AVAILABLE:
            raise ImportError(
                "TensorRT is not available. Please install:\n"
                "  pip install nvidia-tensorrt pycuda\n"
                "Note: Requires NVIDIA GPU and CUDA toolkit"
            )

        self.model_name = model_name
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

    def build_engine_from_onnx(
        self,
        onnx_path: str,
        engine_path: str = None,
        fp16_mode: bool = True,
        max_workspace_size: int = 1 << 30  # 1GB
    ):
        """
        Build TensorRT engine from ONNX model

        Args:
            onnx_path: Path to ONNX model
            engine_path: Path to save TensorRT engine
            fp16_mode: Enable FP16 precision (faster, slightly less accurate)
            max_workspace_size: Maximum workspace size in bytes
        """
        print(f"Building TensorRT engine from {onnx_path}...")

        # Create builder and network
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse ONNX model
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print("ERROR: Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        print("✓ ONNX model parsed successfully")

        # Create builder config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

        # Enable FP16 if requested and supported
        if fp16_mode and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✓ FP16 mode enabled")
        else:
            print("  Using FP32 mode")

        # Build engine
        print("Building TensorRT engine (this may take a few minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            print("ERROR: Failed to build TensorRT engine")
            return None

        # Save engine
        if engine_path is None:
            engine_path = f"models/{self.model_name}_tensorrt.trt"

        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)

        print(f"✓ TensorRT engine saved to {engine_path}")

        # Create runtime and deserialize engine
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()

        print("✓ TensorRT engine ready")

        return self

    def load_engine(self, engine_path: str):
        """Load a pre-built TensorRT engine"""
        print(f"Loading TensorRT engine from {engine_path}...")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        print("✓ TensorRT engine loaded")

        return self

    def allocate_buffers(self, input_shape: Tuple[int, int, int, int]):
        """Allocate GPU buffers for input and output"""
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                shape = input_shape
            else:
                shape = self.context.get_tensor_shape(tensor_name)

            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem})
                self.context.set_input_shape(tensor_name, input_shape)
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def warmup(self, input_shape: Tuple[int, int, int, int], iterations: int = 10):
        """Warmup the TensorRT engine"""
        print(f"Warming up for {iterations} iterations...")

        self.allocate_buffers(input_shape)
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        for _ in range(iterations):
            np.copyto(self.inputs[0]['host'], dummy_input.ravel())
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()

        print("✓ Warmup complete")

    def benchmark(self, input_shape: Tuple[int, int, int, int], iterations: int = 100) -> Dict[str, float]:
        """Benchmark TensorRT inference"""
        print(f"\nBenchmarking TensorRT (batch_size={input_shape[0]})...")

        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        latencies = []

        for _ in range(iterations):
            # Copy input to device
            np.copyto(self.inputs[0]['host'], dummy_input.ravel())

            # Start timing
            start_time = time.perf_counter()

            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()

            # End timing
            end_time = time.perf_counter()

            latencies.append((end_time - start_time) * 1000)

        # Calculate statistics
        latencies = np.array(latencies)
        batch_size = input_shape[0]

        results = {
            "format": "TensorRT",
            "device": "CUDA",
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
    if TRT_AVAILABLE:
        # Quick test - requires ONNX model to exist
        inference = TensorRTInference(model_name="resnet50")
        inference.build_engine_from_onnx(
            onnx_path="models/resnet50.onnx",
            engine_path="models/resnet50_tensorrt.trt",
            fp16_mode=True
        )
        inference.warmup(input_shape=(1, 3, 224, 224), iterations=10)
        results = inference.benchmark(input_shape=(1, 3, 224, 224), iterations=100)

        print("\nResults:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    else:
        print("TensorRT not available - skipping test")
