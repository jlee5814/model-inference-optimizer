"""
Model Inference Optimizer - Main Benchmark Script

This script runs comprehensive benchmarks across all supported optimization formats:
- PyTorch (baseline)
- TorchScript
- ONNX Runtime
- TensorRT (optional, requires NVIDIA GPU)

Usage:
    python benchmark_all.py [--config config.yaml] [--batch-size 1]
"""

import argparse
import yaml
import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List
import torch

from benchmarks.pytorch_baseline import PyTorchInference
from benchmarks.torchscript_inference import TorchScriptInference
from benchmarks.onnx_inference import ONNXInference

# Optional TensorRT import
try:
    from benchmarks.tensorrt_inference import TensorRTInference
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False


class ModelInferenceOptimizer:
    """Main orchestrator for model optimization and benchmarking"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the optimizer

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Create output directories
        os.makedirs(self.config['output']['models_dir'], exist_ok=True)
        os.makedirs(self.config['output']['results_dir'], exist_ok=True)

        self.results = []

    def run_pytorch_baseline(self, batch_size: int):
        """Run PyTorch baseline benchmark"""
        print("\n" + "="*80)
        print("PYTORCH BASELINE")
        print("="*80)

        device = self.config['benchmark']['device']
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            print("Warning: CUDA not available, using CPU")

        inference = PyTorchInference(
            model_name=self.config['model']['name'],
            device=device
        )

        inference.load_model(pretrained=self.config['model']['pretrained'])

        input_shape = tuple(self.config['model']['input_shape'])
        input_shape = (batch_size, *input_shape[1:])

        inference.warmup(
            input_shape=input_shape,
            iterations=self.config['benchmark']['warmup_iterations']
        )

        results = inference.benchmark(
            input_shape=input_shape,
            iterations=self.config['benchmark']['benchmark_iterations']
        )

        self.results.append(results)
        return results

    def run_torchscript(self, batch_size: int):
        """Run TorchScript benchmark"""
        print("\n" + "="*80)
        print("TORCHSCRIPT")
        print("="*80)

        if not self.config['optimization']['enable_torchscript']:
            print("TorchScript optimization disabled in config")
            return None

        device = self.config['benchmark']['device']
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        inference = TorchScriptInference(
            model_name=self.config['model']['name'],
            device=device
        )

        model_path = os.path.join(
            self.config['output']['models_dir'],
            f"{self.config['model']['name']}_torchscript.pt"
        )

        # Convert or load
        if not os.path.exists(model_path):
            inference.convert_to_torchscript(
                pretrained=self.config['model']['pretrained'],
                save_path=model_path
            )
        else:
            inference.load_model(model_path)

        input_shape = tuple(self.config['model']['input_shape'])
        input_shape = (batch_size, *input_shape[1:])

        inference.warmup(
            input_shape=input_shape,
            iterations=self.config['benchmark']['warmup_iterations']
        )

        results = inference.benchmark(
            input_shape=input_shape,
            iterations=self.config['benchmark']['benchmark_iterations']
        )

        self.results.append(results)
        return results

    def run_onnx(self, batch_size: int):
        """Run ONNX Runtime benchmark"""
        print("\n" + "="*80)
        print("ONNX RUNTIME")
        print("="*80)

        if not self.config['optimization']['enable_onnx']:
            print("ONNX optimization disabled in config")
            return None

        device = self.config['benchmark']['device']
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        inference = ONNXInference(
            model_name=self.config['model']['name'],
            device=device
        )

        model_path = os.path.join(
            self.config['output']['models_dir'],
            f"{self.config['model']['name']}.onnx"
        )

        # Convert or load
        if not os.path.exists(model_path):
            inference.convert_to_onnx(
                pretrained=self.config['model']['pretrained'],
                save_path=model_path,
                opset_version=self.config['optimization']['onnx']['opset_version'],
                optimize=self.config['optimization']['onnx']['optimize_graph']
            )

        inference.load_model(model_path)

        input_shape = tuple(self.config['model']['input_shape'])
        input_shape = (batch_size, *input_shape[1:])

        inference.warmup(
            input_shape=input_shape,
            iterations=self.config['benchmark']['warmup_iterations']
        )

        results = inference.benchmark(
            input_shape=input_shape,
            iterations=self.config['benchmark']['benchmark_iterations']
        )

        self.results.append(results)
        return results

    def run_tensorrt(self, batch_size: int):
        """Run TensorRT benchmark (optional)"""
        print("\n" + "="*80)
        print("TENSORRT")
        print("="*80)

        if not self.config['optimization']['enable_tensorrt']:
            print("TensorRT optimization disabled in config")
            return None

        if not TENSORRT_AVAILABLE:
            print("TensorRT not available - skipping")
            return None

        inference = TensorRTInference(model_name=self.config['model']['name'])

        onnx_path = os.path.join(
            self.config['output']['models_dir'],
            f"{self.config['model']['name']}.onnx"
        )

        engine_path = os.path.join(
            self.config['output']['models_dir'],
            f"{self.config['model']['name']}_tensorrt.trt"
        )

        # Build or load engine
        if not os.path.exists(engine_path):
            if not os.path.exists(onnx_path):
                print("Error: ONNX model not found. Run ONNX benchmark first.")
                return None

            inference.build_engine_from_onnx(
                onnx_path=onnx_path,
                engine_path=engine_path,
                fp16_mode=self.config['optimization']['tensorrt']['fp16_mode'],
                max_workspace_size=self.config['optimization']['tensorrt']['max_workspace_size']
            )
        else:
            inference.load_engine(engine_path)

        input_shape = tuple(self.config['model']['input_shape'])
        input_shape = (batch_size, *input_shape[1:])

        inference.warmup(
            input_shape=input_shape,
            iterations=self.config['benchmark']['warmup_iterations']
        )

        results = inference.benchmark(
            input_shape=input_shape,
            iterations=self.config['benchmark']['benchmark_iterations']
        )

        self.results.append(results)
        return results

    def run_all_benchmarks(self, batch_sizes: List[int] = None):
        """Run all benchmarks for all batch sizes"""
        if batch_sizes is None:
            batch_sizes = self.config['benchmark']['batch_sizes']

        print("\n" + "#"*80)
        print(f"# MODEL INFERENCE OPTIMIZER")
        print(f"# Model: {self.config['model']['name']}")
        print(f"# Device: {self.config['benchmark']['device']}")
        print(f"# Batch sizes: {batch_sizes}")
        print("#"*80)

        all_results = []

        for batch_size in batch_sizes:
            print(f"\n\n{'='*80}")
            print(f"BATCH SIZE: {batch_size}")
            print(f"{'='*80}")

            self.results = []

            # Run all benchmarks
            self.run_pytorch_baseline(batch_size)
            self.run_torchscript(batch_size)
            self.run_onnx(batch_size)
            self.run_tensorrt(batch_size)

            all_results.extend(self.results)

        return all_results

    def save_results(self, results: List[Dict], output_path: str = None):
        """Save benchmark results to JSON and CSV"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.config['output']['results_dir'],
                f"benchmark_results_{timestamp}"
            )

        # Save as JSON
        json_path = f"{output_path}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {json_path}")

        # Save as CSV
        csv_path = f"{output_path}.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print(f"✓ Results saved to {csv_path}")

        return df

    def print_summary(self, results: List[Dict]):
        """Print a summary table of results"""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)

        df = pd.DataFrame(results)

        # Group by format and batch size
        summary = df.groupby(['format', 'batch_size']).agg({
            'mean_latency_ms': 'mean',
            'throughput_samples_per_sec': 'mean'
        }).round(2)

        print("\n" + summary.to_string())

        # Calculate speedup relative to PyTorch baseline
        print("\n" + "="*80)
        print("SPEEDUP vs PYTORCH BASELINE")
        print("="*80)

        for batch_size in df['batch_size'].unique():
            print(f"\nBatch Size: {batch_size}")
            batch_df = df[df['batch_size'] == batch_size]

            pytorch_latency = batch_df[batch_df['format'] == 'PyTorch']['mean_latency_ms'].values
            if len(pytorch_latency) == 0:
                continue

            pytorch_latency = pytorch_latency[0]

            for _, row in batch_df.iterrows():
                speedup = pytorch_latency / row['mean_latency_ms']
                print(f"  {row['format']:20s}: {speedup:.2f}x speedup "
                      f"({row['mean_latency_ms']:.2f} ms)")


def main():
    parser = argparse.ArgumentParser(description="Model Inference Optimizer")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        nargs='+',
        help='Batch size(s) to benchmark (overrides config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for results (without extension)'
    )

    args = parser.parse_args()

    # Create optimizer
    optimizer = ModelInferenceOptimizer(config_path=args.config)

    # Run benchmarks
    batch_sizes = args.batch_size if args.batch_size else None
    results = optimizer.run_all_benchmarks(batch_sizes=batch_sizes)

    # Save and display results
    df = optimizer.save_results(results, output_path=args.output)
    optimizer.print_summary(results)

    print("\n" + "="*80)
    print("BENCHMARKING COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Check the results directory for detailed metrics")
    print("  2. Run visualize_results.py to generate comparison charts")
    print("  3. Analyze the speedup and choose the best format for your use case")


if __name__ == "__main__":
    main()
