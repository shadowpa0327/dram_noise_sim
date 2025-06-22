"""
Simple demo of DRAM error simulation functionality.

For comprehensive testing, run: python test_correctness.py
For benchmarking, run: python benchmark.py
"""

import torch
from dram_error_simulation import dram_bitflip, dram_bitflip_triton


def simple_demo():
    """Simple demonstration of DRAM error simulation."""
    print("DRAM Error Simulation - Simple Demo")
    print("=" * 40)
    
    torch.manual_seed(42)
    
    # Create a small test tensor
    x = torch.randn(2, 128, dtype=torch.float16, device="cuda") 
    x_original = x.clone()
    
    print(f"Original tensor shape: {x.shape}")
    print(f"Original values (first 6): {x.flatten()[:6]}")
    
    # Test with scalar probability
    print("\n1. Testing with scalar probability (p=0.01):")
    y_pytorch, dmg_mask = dram_bitflip(x.clone(), p=0.01)
    y_triton = dram_bitflip_triton(x.clone(), p=0.01)
    
    pytorch_changes = (x_original != y_pytorch).sum().item()
    triton_changes = (x_original != y_triton).sum().item()
    
    print(f"   PyTorch version changed {pytorch_changes} elements")
    print(f"   Triton version changed {triton_changes} elements")
    print(f"   PyTorch result (first 6): {y_pytorch.flatten()[:6]}")
    print(f"   Triton result (first 6): {y_triton.flatten()[:6]}")
    
    # Test with vector probability
    print("\n2. Testing with vector probability (1024 elements):")
    p_vector = torch.full((1024,), 0.001, device="cuda")
    p_vector[::100] = 0.05  # Higher probability every 100th bit
    
    y_pytorch_vec, _ = dram_bitflip(x.clone(), p=p_vector)
    y_triton_vec = dram_bitflip_triton(x.clone(), p=p_vector)
    
    pytorch_vec_changes = (x_original != y_pytorch_vec).sum().item()
    triton_vec_changes = (x_original != y_triton_vec).sum().item()
    
    print(f"   PyTorch version changed {pytorch_vec_changes} elements")
    print(f"   Triton version changed {triton_vec_changes} elements")
    print(f"   PyTorch result (first 6): {y_pytorch_vec.flatten()[:6]}")
    print(f"   Triton result (first 6): {y_triton_vec.flatten()[:6]}")
    
    print("\n✅ Demo completed!")
    print("\nFor more comprehensive testing and benchmarking:")
    print("  - Run correctness tests: python test_correctness.py")
    print("  - Run benchmarks: python benchmark.py")


if __name__ == "__main__":
    simple_demo()


