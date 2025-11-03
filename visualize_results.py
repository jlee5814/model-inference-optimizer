"""
Visualization Script for Benchmark Results

This script generates comparison charts and tables from benchmark results.

Usage:
    python visualize_results.py --input results/benchmark_results_20240101_120000.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tabulate import tabulate
import numpy as np


class BenchmarkVisualizer:
    """Visualize and compare benchmark results"""

    def __init__(self, results_path: str):
        """
        Initialize visualizer

        Args:
            results_path: Path to benchmark results CSV file
        """
        self.results_path = results_path
        self.df = pd.read_csv(results_path)
        self.output_dir = os.path.join(os.path.dirname(results_path), "charts")
        os.makedirs(self.output_dir, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)

    def create_latency_comparison(self):
        """Create latency comparison bar chart"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Group by format and batch_size
        pivot_data = self.df.pivot(
            index='batch_size',
            columns='format',
            values='mean_latency_ms'
        )

        pivot_data.plot(kind='bar', ax=ax, width=0.8)

        ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Inference Latency Comparison Across Optimization Formats',
                     fontsize=14, fontweight='bold')
        ax.legend(title='Format', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.xticks(rotation=0)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'latency_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Latency comparison saved to {output_path}")

        plt.close()

    def create_throughput_comparison(self):
        """Create throughput comparison bar chart"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Group by format and batch_size
        pivot_data = self.df.pivot(
            index='batch_size',
            columns='format',
            values='throughput_samples_per_sec'
        )

        pivot_data.plot(kind='bar', ax=ax, width=0.8)

        ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Throughput (samples/sec)', fontsize=12, fontweight='bold')
        ax.set_title('Inference Throughput Comparison Across Optimization Formats',
                     fontsize=14, fontweight='bold')
        ax.legend(title='Format', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.xticks(rotation=0)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'throughput_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Throughput comparison saved to {output_path}")

        plt.close()

    def create_speedup_chart(self):
        """Create speedup comparison chart (relative to PyTorch baseline)"""
        # Calculate speedup for each batch size
        speedup_data = []

        for batch_size in self.df['batch_size'].unique():
            batch_df = self.df[self.df['batch_size'] == batch_size]

            # Get PyTorch baseline
            pytorch_row = batch_df[batch_df['format'] == 'PyTorch']
            if len(pytorch_row) == 0:
                continue

            pytorch_latency = pytorch_row['mean_latency_ms'].values[0]

            # Calculate speedup for each format
            for _, row in batch_df.iterrows():
                speedup = pytorch_latency / row['mean_latency_ms']
                speedup_data.append({
                    'batch_size': batch_size,
                    'format': row['format'],
                    'speedup': speedup
                })

        speedup_df = pd.DataFrame(speedup_data)

        # Create chart
        fig, ax = plt.subplots(figsize=(12, 6))

        pivot_data = speedup_df.pivot(
            index='batch_size',
            columns='format',
            values='speedup'
        )

        pivot_data.plot(kind='bar', ax=ax, width=0.8)

        ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup vs PyTorch Baseline', fontsize=12, fontweight='bold')
        ax.set_title('Inference Speedup Comparison (Higher is Better)',
                     fontsize=14, fontweight='bold')
        ax.legend(title='Format', fontsize=10)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
        ax.grid(axis='y', alpha=0.3)

        plt.xticks(rotation=0)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'speedup_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Speedup comparison saved to {output_path}")

        plt.close()

    def create_latency_distribution(self):
        """Create latency distribution box plot"""
        fig, ax = plt.subplots(figsize=(14, 6))

        # Select relevant columns for distribution
        dist_cols = ['format', 'batch_size', 'min_latency_ms', 'p50_latency_ms',
                     'p95_latency_ms', 'p99_latency_ms', 'max_latency_ms']

        # Create a more detailed visualization
        formats = self.df['format'].unique()
        batch_sizes = sorted(self.df['batch_size'].unique())

        x_pos = np.arange(len(formats))
        width = 0.8 / len(batch_sizes)

        for i, batch_size in enumerate(batch_sizes):
            batch_df = self.df[self.df['batch_size'] == batch_size]
            p50_values = batch_df.set_index('format')['p50_latency_ms'].reindex(formats).values

            ax.bar(x_pos + i * width, p50_values, width, label=f'Batch {batch_size}', alpha=0.8)

        ax.set_xlabel('Optimization Format', fontsize=12, fontweight='bold')
        ax.set_ylabel('P50 Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Median Latency Distribution by Format and Batch Size',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos + width * (len(batch_sizes) - 1) / 2)
        ax.set_xticklabels(formats)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'latency_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Latency distribution saved to {output_path}")

        plt.close()

    def create_summary_table(self):
        """Create and save a formatted summary table"""
        print("\n" + "="*100)
        print("DETAILED BENCHMARK RESULTS")
        print("="*100)

        # Create summary for each batch size
        for batch_size in sorted(self.df['batch_size'].unique()):
            batch_df = self.df[self.df['batch_size'] == batch_size]

            print(f"\n{'='*100}")
            print(f"Batch Size: {batch_size}")
            print(f"{'='*100}")

            # Calculate speedup
            pytorch_row = batch_df[batch_df['format'] == 'PyTorch']
            if len(pytorch_row) > 0:
                pytorch_latency = pytorch_row['mean_latency_ms'].values[0]
            else:
                pytorch_latency = None

            table_data = []
            for _, row in batch_df.iterrows():
                if pytorch_latency:
                    speedup = pytorch_latency / row['mean_latency_ms']
                else:
                    speedup = 1.0

                table_data.append([
                    row['format'],
                    f"{row['mean_latency_ms']:.2f}",
                    f"{row['std_latency_ms']:.2f}",
                    f"{row['p50_latency_ms']:.2f}",
                    f"{row['p95_latency_ms']:.2f}",
                    f"{row['p99_latency_ms']:.2f}",
                    f"{row['throughput_samples_per_sec']:.1f}",
                    f"{speedup:.2f}x"
                ])

            headers = ['Format', 'Mean (ms)', 'Std (ms)', 'P50 (ms)', 'P95 (ms)',
                       'P99 (ms)', 'Throughput (img/s)', 'Speedup']

            print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))

        # Save to file
        output_path = os.path.join(self.output_dir, 'summary_table.txt')
        with open(output_path, 'w') as f:
            for batch_size in sorted(self.df['batch_size'].unique()):
                batch_df = self.df[self.df['batch_size'] == batch_size]

                f.write(f"\nBatch Size: {batch_size}\n")
                f.write("="*100 + "\n")

                pytorch_row = batch_df[batch_df['format'] == 'PyTorch']
                if len(pytorch_row) > 0:
                    pytorch_latency = pytorch_row['mean_latency_ms'].values[0]
                else:
                    pytorch_latency = None

                table_data = []
                for _, row in batch_df.iterrows():
                    if pytorch_latency:
                        speedup = pytorch_latency / row['mean_latency_ms']
                    else:
                        speedup = 1.0

                    table_data.append([
                        row['format'],
                        f"{row['mean_latency_ms']:.2f}",
                        f"{row['std_latency_ms']:.2f}",
                        f"{row['throughput_samples_per_sec']:.1f}",
                        f"{speedup:.2f}x"
                    ])

                headers = ['Format', 'Mean Latency (ms)', 'Std (ms)',
                           'Throughput (img/s)', 'Speedup']

                f.write(tabulate(table_data, headers=headers, tablefmt='grid'))
                f.write("\n\n")

        print(f"\n✓ Summary table saved to {output_path}")

    def generate_all_visualizations(self):
        """Generate all visualizations and reports"""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80 + "\n")

        self.create_latency_comparison()
        self.create_throughput_comparison()
        self.create_speedup_chart()
        self.create_latency_distribution()
        self.create_summary_table()

        print("\n" + "="*80)
        print("VISUALIZATION COMPLETE!")
        print("="*80)
        print(f"\nAll charts and tables saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to benchmark results CSV file'
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return

    visualizer = BenchmarkVisualizer(args.input)
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
