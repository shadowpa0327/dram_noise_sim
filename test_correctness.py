import torch
from dram_error_simulation import pack_and_xor_triton, dram_bitflip, dram_bitflip_triton


def pack_and_xor_pytorch(x_int: torch.Tensor, rand_bits: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of bit packing and XOR operation.
    
    Parameters
    ----------
    x_int : torch.Tensor
        Int16 tensor to be modified, shape (N, 64)
    rand_bits : torch.Tensor  
        Boolean tensor with random bits, shape (N, 64, 16)
    
    Returns
    -------
    torch.Tensor
        The damage mask that was applied
    """
    N, W = x_int.shape
    assert W == 64, "W must be 64"
    
    # Bit‑position indices 0…15 → shift amounts
    bit_shifts = torch.arange(16, device=x_int.device, dtype=torch.int16).view(1, 1, 16)
    dmg_mask = (rand_bits.to(torch.int16) << bit_shifts).sum(dim=2).to(torch.int16)
    
    # Apply XOR
    x_int ^= dmg_mask
    
    return dmg_mask


def test_pack_and_xor_correctness():
    """
    Test correctness of Triton vs PyTorch implementations using identical inputs.
    """
    print()
    print("=" * 70)
    print("CORRECTNESS TEST: PyTorch vs Triton Pack-and-XOR")
    print("=" * 70)
    
    torch.manual_seed(12345)
    
    # Create test data
    N_bursts = 8
    test_data = torch.randn(N_bursts, 64, dtype=torch.float16, device="cuda")
    test_data_int = test_data.view(torch.int16)
    
    # Generate identical random bits for both implementations
    p = 0.01  # Higher probability to get more flips for testing
    rand_uniform = torch.rand((N_bursts, 64*16), device="cuda")
    rand_bits_bool = rand_uniform < p
    
    print(f"Test tensor shape: {test_data.shape}")
    print(f"Test data as int16 shape: {test_data_int.shape}")
    print(f"Random bits shape: {rand_bits_bool.shape}")
    print(f"Flip probability: {p}")
    print(f"Expected flips: ~{rand_bits_bool.sum().item()} bits")
    
    # ------------------------------------------------------------------
    # Test PyTorch implementation
    # ------------------------------------------------------------------
    pytorch_data = test_data_int.clone()
    rand_bits_pytorch = rand_bits_bool.view(N_bursts, 64, 16)  # Shape for PyTorch version
    
    pytorch_dmg_mask = pack_and_xor_pytorch(pytorch_data, rand_bits_pytorch)
    
    # ------------------------------------------------------------------
    # Test Triton implementation
    # ------------------------------------------------------------------
    triton_data = test_data_int.clone()
    rand_bits_triton = rand_bits_bool.view(-1, 16).to(torch.uint8)  # Shape for Triton version
    triton_data_flat = triton_data.view(-1).contiguous()
    
    pack_and_xor_triton(triton_data_flat, rand_bits_triton)
    
    # Reshape triton result back to (N_bursts, 64) for comparison
    triton_data = triton_data_flat.view(N_bursts, 64)
    
    # ------------------------------------------------------------------
    # Compare results
    # ------------------------------------------------------------------
    print()
    print("COMPARISON RESULTS:")
    print("-" * 40)
    
    # Check if the final results are identical
    results_match = torch.equal(pytorch_data, triton_data)
    print(f"Final results identical: {results_match}")
    
    if not results_match:
        diff_mask = pytorch_data != triton_data
        num_differences = diff_mask.sum().item()
        print(f"Number of different elements: {num_differences}")
        
        if num_differences <= 10:  # Show details for small number of differences
            diff_positions = torch.where(diff_mask)
            print("Differences found at positions:")
            for i in range(min(10, num_differences)):
                pos_n, pos_w = diff_positions[0][i].item(), diff_positions[1][i].item()
                pytorch_val = pytorch_data[pos_n, pos_w].item()
                triton_val = triton_data[pos_n, pos_w].item()
                print(f"  [{pos_n}, {pos_w}]: PyTorch={pytorch_val}, Triton={triton_val}")
    
    # Compare damage masks (only available for PyTorch version)
    print(f"PyTorch damage mask shape: {pytorch_dmg_mask.shape}")
    print(f"PyTorch damage mask sample: {pytorch_dmg_mask.flatten()[:8]}")
    
    # Verify the XOR operation by reconstructing original data
    pytorch_reconstructed = pytorch_data ^ pytorch_dmg_mask
    
    original_matches_pytorch = torch.equal(test_data_int, pytorch_reconstructed)
    print(f"PyTorch reconstruction matches original: {original_matches_pytorch}")
    
    # Show some sample values in binary for detailed inspection
    if not results_match or not original_matches_pytorch:
        print()
        print("BINARY INSPECTION (first 4 elements):")
        print("-" * 50)
        
        def show_binary(tensor, name, max_show=4):
            flat_tensor = tensor.flatten()[:max_show]
            print(f"{name}:")
            for i, val in enumerate(flat_tensor):
                unsigned_val = val.item() & 0xFFFF
                binary_str = format(unsigned_val, '016b')
                print(f"  [{i}]: {binary_str} ({val.item()})")
        
        show_binary(test_data_int, "Original")
        show_binary(pytorch_data, "PyTorch result")
        show_binary(triton_data, "Triton result")
        show_binary(pytorch_dmg_mask, "PyTorch damage mask")
    
    return results_match


def test_dram_bitflip_correctness():
    """
    Test correctness of dram_bitflip vs dram_bitflip_triton using same random seed.
    """
    print()
    print("=" * 70)
    print("CORRECTNESS TEST: dram_bitflip vs dram_bitflip_triton")
    print("=" * 70)
    
    # Test data
    x_original = torch.randn(4, 256, dtype=torch.float16, device="cuda")
    print(f"Test tensor shape: {x_original.shape}")
    
    test_cases = [
        ("Scalar probability", 0.001),
        ("Vector probability", torch.full((1024,), 0.0005, device="cuda"))
    ]
    
    all_passed = True
    
    for test_name, p in test_cases:
        print(f"\nTesting {test_name}:")
        print("-" * 30)
        
        # Test with same random seed
        torch.manual_seed(42)
        x_pytorch = x_original.clone()
        y_pytorch, dmg_mask = dram_bitflip(x_pytorch, p)
        
        torch.manual_seed(42)
        x_triton = x_original.clone()
        y_triton = dram_bitflip_triton(x_triton, p)
        
        # Compare results
        results_match = torch.allclose(y_pytorch, y_triton, rtol=1e-3, atol=1e-6)
        elements_different = (y_pytorch != y_triton).sum().item()
        
        print(f"  Results approximately equal: {results_match}")
        print(f"  Elements different: {elements_different}")
        print(f"  PyTorch changed elements: {(x_original != y_pytorch).sum().item()}")
        print(f"  Triton changed elements: {(x_original != y_triton).sum().item()}")
        
        if not results_match:
            all_passed = False
            max_diff = (y_pytorch - y_triton).abs().max().item()
            print(f"  Maximum difference: {max_diff}")
    
    return all_passed


def run_all_correctness_tests():
    """Run all correctness tests."""
    print("Running DRAM Error Simulation Correctness Tests")
    print("=" * 80)
    
    # Test 1: Pack and XOR operations
    pack_xor_passed = test_pack_and_xor_correctness()
    
    # Test 2: Full DRAM bitflip functions
    dram_bitflip_passed = test_dram_bitflip_correctness()
    
    # Summary
    print()
    print("=" * 80)
    print("CORRECTNESS TEST SUMMARY")
    print("=" * 80)
    print(f"Pack-and-XOR test: {'✅ PASSED' if pack_xor_passed else '❌ FAILED'}")
    print(f"DRAM bitflip test: {'✅ PASSED' if dram_bitflip_passed else '❌ FAILED'}")
    
    overall_passed = pack_xor_passed and dram_bitflip_passed
    print(f"Overall: {'✅ ALL TESTS PASSED' if overall_passed else '❌ SOME TESTS FAILED'}")
    
    return overall_passed


if __name__ == "__main__":
    run_all_correctness_tests() 