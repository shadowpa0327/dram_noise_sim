import torch
import time
from typing import Callable, List, Tuple
from dram_error_simulation import dram_bitflip, dram_bitflip_triton
from test_correctness import run_all_correctness_tests


def warmup_gpu(device: str = "cuda", iterations: int = 5):
    """Warm up the GPU to get consistent timing measurements."""
    print("🔥 Warming up GPU...")
    
    for _ in range(iterations):
        # Some dummy operations to warm up GPU
        x = torch.randn(1000, 1000, dtype=torch.float16, device=device)
        y = torch.matmul(x, x.T)
        z = torch.sum(y)
        torch.cuda.synchronize()
    
    print("✅ GPU warmup complete")


def benchmark_function(
    func: Callable,
    args: tuple,
    kwargs: dict,
    iterations: int = 10,
    warmup_iterations: int = 3,
    name: str = "Function"
) -> Tuple[float, float]:
    """
    Benchmark a function with proper warmup.
    
    Returns
    -------
    Tuple[float, float]
        (mean_time, std_time) in seconds
    """
    print(f"  Benchmarking {name}...")
    
    # Warmup iterations
    for _ in range(warmup_iterations):
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
    
    # Actual timing
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start_time = time.time()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return mean_time, std_time


def benchmark_scalar_probability():
    """Benchmark scalar probability case."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Scalar Probability (p=1e-3)")
    print("=" * 60)
    
    # Test configurations
    test_sizes = [
        (4, 1024),      # Small
        (16, 2048),     # Medium  
        (64, 4096),     # Large
    ]
    
    p = 1e-3
    results = []
    
    for batch_size, seq_len in test_sizes:
        print(f"\nTensor size: ({batch_size}, {seq_len})")
        print("-" * 40)
        
        # Create test data
        x_original = torch.randn(batch_size, seq_len, dtype=torch.float16, device="cuda")
        
        # Benchmark PyTorch version
        def pytorch_version():
            torch.manual_seed(42)  # Reset seed for fair comparison
            x = x_original.clone()
            y, _, _, _ = dram_bitflip(x, p)  # Unpack tuple (4 elements now)
            return y
        
        pytorch_mean, pytorch_std = benchmark_function(
            pytorch_version, (), {}, name="PyTorch"
        )
        
        # Benchmark Triton version
        def triton_version():
            torch.manual_seed(42)  # Reset seed for fair comparison
            x = x_original.clone()
            y, _, _ = dram_bitflip_triton(x, p)  # Now returns (tensor, bits_flipped, bit_position_flips)
            return y
        
        triton_mean, triton_std = benchmark_function(
            triton_version, (), {}, name="Triton"
        )
        
        # Calculate speedup
        speedup = pytorch_mean / triton_mean
        
        print(f"  PyTorch: {pytorch_mean:.6f} ± {pytorch_std:.6f} seconds")
        print(f"  Triton:  {triton_mean:.6f} ± {triton_std:.6f} seconds")
        print(f"  Speedup: {speedup:.2f}x")
        
        results.append({
            'size': f"{batch_size}x{seq_len}",
            'pytorch_time': pytorch_mean,
            'triton_time': triton_mean,
            'speedup': speedup
        })
    
    return results


def benchmark_vector_probability():
    """Benchmark vector probability case."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Vector Probability (1024 elements)")
    print("=" * 60)
    
    # Test configurations
    test_sizes = [
        (4, 1024),      # Small
        (16, 2048),     # Medium
        (64, 4096),     # Large
    ]
    
    # Create probability vector with varying probabilities
    p_vector = torch.full((1024,), 1e-4, device="cuda")
    p_vector[::100] = 1e-2  # Higher probability every 100th bit
    
    results = []
    
    for batch_size, seq_len in test_sizes:
        print(f"\nTensor size: ({batch_size}, {seq_len})")
        print("-" * 40)
        
        # Create test data
        x_original = torch.randn(batch_size, seq_len, dtype=torch.float16, device="cuda")
        
        # Benchmark PyTorch version
        def pytorch_version():
            torch.manual_seed(42)
            x = x_original.clone()
            y, _, _, _ = dram_bitflip(x, p_vector)  # Unpack tuple (4 elements now)
            return y
        
        pytorch_mean, pytorch_std = benchmark_function(
            pytorch_version, (), {}, name="PyTorch"
        )
        
        # Benchmark Triton version
        def triton_version():
            torch.manual_seed(42)
            x = x_original.clone()
            y, _, _ = dram_bitflip_triton(x, p_vector)  # Now returns (tensor, bits_flipped, bit_position_flips)
            return y
        
        triton_mean, triton_std = benchmark_function(
            triton_version, (), {}, name="Triton"
        )
        
        # Calculate speedup
        speedup = pytorch_mean / triton_mean
        
        print(f"  PyTorch: {pytorch_mean:.6f} ± {pytorch_std:.6f} seconds")
        print(f"  Triton:  {triton_mean:.6f} ± {triton_std:.6f} seconds")
        print(f"  Speedup: {speedup:.2f}x")
        
        results.append({
            'size': f"{batch_size}x{seq_len}",
            'pytorch_time': pytorch_mean,
            'triton_time': triton_mean,
            'speedup': speedup
        })
    
    return results


def print_benchmark_summary(scalar_results: List[dict], vector_results: List[dict]):
    """Print a summary of all benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    print("\nScalar Probability Results:")
    print("-" * 50)
    print(f"{'Size':<12} {'PyTorch (s)':<12} {'Triton (s)':<12} {'Speedup':<10}")
    print("-" * 50)
    for result in scalar_results:
        print(f"{result['size']:<12} {result['pytorch_time']:<12.6f} "
              f"{result['triton_time']:<12.6f} {result['speedup']:<10.2f}x")
    
    print("\nVector Probability Results:")
    print("-" * 50)
    print(f"{'Size':<12} {'PyTorch (s)':<12} {'Triton (s)':<12} {'Speedup':<10}")
    print("-" * 50)
    for result in vector_results:
        print(f"{result['size']:<12} {result['pytorch_time']:<12.6f} "
              f"{result['triton_time']:<12.6f} {result['speedup']:<10.2f}x")
    
    # Calculate average speedups
    scalar_avg_speedup = sum(r['speedup'] for r in scalar_results) / len(scalar_results)
    vector_avg_speedup = sum(r['speedup'] for r in vector_results) / len(vector_results)
    
    print(f"\nAverage Speedups:")
    print(f"  Scalar probability: {scalar_avg_speedup:.2f}x")
    print(f"  Vector probability: {vector_avg_speedup:.2f}x")
    print(f"  Overall average: {(scalar_avg_speedup + vector_avg_speedup) / 2:.2f}x")


def run_benchmarks():
    """Run all benchmarks."""
    print("🚀 Starting DRAM Error Simulation Benchmarks")
    print("=" * 80)
    
    # First ensure correctness
    print("Step 1: Running correctness tests...")
    correctness_passed = run_all_correctness_tests()
    
    if not correctness_passed:
        print("❌ Correctness tests failed! Aborting benchmarks.")
        return
    
    print("\n✅ Correctness tests passed! Proceeding with benchmarks...")
    
    # Step 2: Warm up GPU
    warmup_gpu()
    
    # Step 3: Run benchmarks
    scalar_results = benchmark_scalar_probability()
    vector_results = benchmark_vector_probability()
    
    # Step 4: Print summary
    print_benchmark_summary(scalar_results, vector_results)
    
    print("\n🎉 Benchmarks completed successfully!")


if __name__ == "__main__":
    run_benchmarks() 