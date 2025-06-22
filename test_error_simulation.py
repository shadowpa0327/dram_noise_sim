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
    print("\n1. Testing with scalar probability (p=0.1):")
    y_pytorch, dmg_mask = dram_bitflip(x.clone(), p=0.1)
    
    pytorch_changes = (x_original != y_pytorch).sum().item()
    
    print(f"   PyTorch version changed {pytorch_changes} elements")
    print(f"   PyTorch result (first 6): {y_pytorch.flatten()[:6]}")
    
    # Test with vector probability
    print("\n2. Testing with vector probability (1024 elements):")
    p_vector = torch.full((1024,), 0.001, device="cuda")
    p_vector[::100] = 0.05  # Higher probability every 100th bit
    
    y_pytorch_vec, _ = dram_bitflip(x.clone(), p=p_vector)
    
    pytorch_vec_changes = (x_original != y_pytorch_vec).sum().item()
    
    print(f"   PyTorch version changed {pytorch_vec_changes} elements")
    print(f"   PyTorch result (first 6): {y_pytorch_vec.flatten()[:6]}")
    
    # print the binary representation of the original and corrupted tensors
    print("\n" + "=" * 60)
    print("BINARY REPRESENTATION ANALYSIS")
    print("=" * 60)
    
    def show_binary_comparison(original, corrupted, title, max_elements=4):
        """Show binary representation comparison between original and corrupted tensors."""
        print(f"\n{title}:")
        print("-" * 40)
        
        # Convert to int16 view to see raw bits
        orig_int16 = original.view(torch.int16).flatten()[:max_elements]
        corr_int16 = corrupted.view(torch.int16).flatten()[:max_elements]
        xor_mask = orig_int16 ^ corr_int16
        
        for i in range(len(orig_int16)):
            orig_val = orig_int16[i].item()
            corr_val = corr_int16[i].item()
            xor_val = xor_mask[i].item()
            if xor_val == 0:
                continue
            # Convert to unsigned 16-bit for display
            orig_unsigned = orig_val & 0xFFFF
            corr_unsigned = corr_val & 0xFFFF
            xor_unsigned = xor_val & 0xFFFF
            
            orig_binary = format(orig_unsigned, '016b')
            corr_binary = format(corr_unsigned, '016b')
            xor_binary = format(xor_unsigned, '016b')
            
            print(f"Element [{i}] had something flipped:")
            print(f"  Original: {orig_binary} ({orig_val:6d}) -> {original.flatten()[i]:.6f}")
            print(f"  Corrupt:  {corr_binary} ({corr_val:6d}) -> {corrupted.flatten()[i]:.6f}")
            print(f"  XOR mask: {xor_binary} ({xor_val:6d})")
            
            # Show which bit positions were flipped
            flipped_positions = [pos for pos, bit in enumerate(xor_binary) if bit == '1']
            if flipped_positions:
                print(f"  Flipped bit positions: {flipped_positions}")
            else:
                print(f"  No bits flipped")
            print()
    
    # Show binary comparison for scalar probability case
    show_binary_comparison(x_original, y_pytorch, "SCALAR PROBABILITY (p=0.1)", max_elements=-1)
    
    # Show binary comparison for vector probability case  
    show_binary_comparison(x_original, y_pytorch_vec, "VECTOR PROBABILITY (1024 elements)", max_elements=-1)
    

if __name__ == "__main__":
    simple_demo()


