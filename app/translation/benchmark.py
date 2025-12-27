"""
Quantization Benchmark Module

Compares performance between FP32 and INT8 quantized models.
- Single query latency
- Batch throughput
- Memory usage
- Generates comparison reports
"""

import time
import logging
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import MarianMTModel, MarianTokenizer

# Add parent to path for imports when running as module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import settings
from app.models import BenchmarkResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    model_name: str = "Helsinki-NLP/opus-mt-fr-en"
    num_warmup: int = 3
    num_runs: int = 10
    batch_sizes: tuple = (1, 4, 8, 16)
    test_queries: tuple = (
        "yogurt liberté logo",
        "je veux acheter du fromage",
        "papier toilette royale",
        "pêche fraîche du marché",
        "gomme pour l'école",
    )


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def benchmark_model(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    config: BenchmarkConfig,
    device: str = "cpu"
) -> dict:
    """
    Benchmark a model's performance.
    
    Returns dict with latency and throughput metrics.
    """
    model.to(device)
    model.eval()
    
    results = {
        "single_query_latencies": [],
        "batch_latencies": {},
        "memory_mb": get_memory_usage()
    }
    
    # Warmup
    for _ in range(config.num_warmup):
        inputs = tokenizer(config.test_queries[0], return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(**inputs, max_length=128)
    
    # Single query benchmark
    for query in config.test_queries:
        for _ in range(config.num_runs):
            inputs = tokenizer(query, return_tensors="pt").to(device)
            
            start = time.perf_counter()
            with torch.no_grad():
                model.generate(**inputs, max_length=128, num_beams=5)
            latency = (time.perf_counter() - start) * 1000
            
            results["single_query_latencies"].append(latency)
    
    # Batch benchmark
    for batch_size in config.batch_sizes:
        batch_queries = (config.test_queries * (batch_size // len(config.test_queries) + 1))[:batch_size]
        batch_latencies = []
        
        for _ in range(config.num_runs):
            inputs = tokenizer(
                list(batch_queries), 
                return_tensors="pt", 
                padding=True
            ).to(device)
            
            start = time.perf_counter()
            with torch.no_grad():
                model.generate(**inputs, max_length=128, num_beams=5)
            latency = (time.perf_counter() - start) * 1000
            
            batch_latencies.append(latency)
        
        results["batch_latencies"][batch_size] = batch_latencies
    
    return results


def run_comparison_benchmark(output_path: Optional[Path] = None) -> dict:
    """
    Run full comparison benchmark between FP32 and INT8 models.
    
    Returns comparison results.
    """
    config = BenchmarkConfig()
    device = "cpu"  # Quantization works best on CPU
    
    print("=" * 60)
    print("QUANTIZATION PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"\nModel: {config.model_name}")
    print(f"Device: {device}")
    print(f"Test queries: {len(config.test_queries)}")
    print(f"Runs per test: {config.num_runs}")
    print()
    
    # Load tokenizer (shared)
    print("Loading tokenizer...")
    tokenizer = MarianTokenizer.from_pretrained(config.model_name)
    
    # ========================================
    # FP32 Model Benchmark
    # ========================================
    print("\n[1/2] Benchmarking FP32 model...")
    
    initial_memory = get_memory_usage()
    model_fp32 = MarianMTModel.from_pretrained(config.model_name)
    fp32_results = benchmark_model(model_fp32, tokenizer, config, device)
    fp32_results["memory_mb"] = get_memory_usage() - initial_memory
    
    # Calculate stats
    fp32_avg_latency = sum(fp32_results["single_query_latencies"]) / len(fp32_results["single_query_latencies"])
    fp32_throughput = 1000 / fp32_avg_latency  # queries per second
    
    print(f"  Average latency: {fp32_avg_latency:.1f} ms")
    print(f"  Throughput: {fp32_throughput:.1f} q/s")
    print(f"  Memory: {fp32_results['memory_mb']:.1f} MB")
    
    # Clean up
    del model_fp32
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ========================================
    # INT8 Quantized Model Benchmark
    # ========================================
    print("\n[2/2] Benchmarking INT8 quantized model...")
    
    initial_memory = get_memory_usage()
    model_int8 = MarianMTModel.from_pretrained(config.model_name)
    
    # Apply dynamic quantization
    model_int8 = torch.quantization.quantize_dynamic(
        model_int8,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    int8_results = benchmark_model(model_int8, tokenizer, config, device)
    int8_results["memory_mb"] = get_memory_usage() - initial_memory
    
    # Calculate stats
    int8_avg_latency = sum(int8_results["single_query_latencies"]) / len(int8_results["single_query_latencies"])
    int8_throughput = 1000 / int8_avg_latency  # queries per second
    
    print(f"  Average latency: {int8_avg_latency:.1f} ms")
    print(f"  Throughput: {int8_throughput:.1f} q/s")
    print(f"  Memory: {int8_results['memory_mb']:.1f} MB")
    
    # Clean up
    del model_int8
    
    # ========================================
    # Comparison Summary
    # ========================================
    speedup = fp32_avg_latency / int8_avg_latency
    memory_reduction = 1 - (int8_results["memory_mb"] / fp32_results["memory_mb"])
    throughput_gain = int8_throughput / fp32_throughput
    
    comparison = {
        "fp32": {
            "average_latency_ms": round(fp32_avg_latency, 2),
            "throughput_qps": round(fp32_throughput, 2),
            "memory_mb": round(fp32_results["memory_mb"], 2),
            "batch_latencies": {
                str(k): round(sum(v) / len(v), 2) 
                for k, v in fp32_results["batch_latencies"].items()
            }
        },
        "int8": {
            "average_latency_ms": round(int8_avg_latency, 2),
            "throughput_qps": round(int8_throughput, 2),
            "memory_mb": round(int8_results["memory_mb"], 2),
            "batch_latencies": {
                str(k): round(sum(v) / len(v), 2) 
                for k, v in int8_results["batch_latencies"].items()
            }
        },
        "improvements": {
            "latency_speedup": f"{speedup:.2f}x",
            "latency_reduction_percent": f"{(1 - 1/speedup) * 100:.1f}%",
            "throughput_gain": f"{throughput_gain:.2f}x",
            "memory_reduction_percent": f"{memory_reduction * 100:.1f}%"
        }
    }
    
    # Print summary table
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'FP32':<15} {'INT8':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Latency (ms)':<25} {fp32_avg_latency:<15.1f} {int8_avg_latency:<15.1f} {comparison['improvements']['latency_reduction_percent']:<15}")
    print(f"{'Throughput (q/s)':<25} {fp32_throughput:<15.1f} {int8_throughput:<15.1f} {comparison['improvements']['throughput_gain']:<15}")
    print(f"{'Memory (MB)':<25} {fp32_results['memory_mb']:<15.1f} {int8_results['memory_mb']:<15.1f} {comparison['improvements']['memory_reduction_percent']:<15}")
    print()
    
    # Save results if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Results saved to: {output_path}")
    
    return comparison


def main():
    """Run benchmark from command line."""
    logging.basicConfig(level=logging.INFO)
    
    output_path = Path(__file__).parent.parent.parent / "benchmark_results.json"
    results = run_comparison_benchmark(output_path)
    
    return results


if __name__ == "__main__":
    main()
