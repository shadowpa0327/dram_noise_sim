import torch
import torch.nn as nn
import torch.nn.functional as F
from monkey_patch import patch_linear_module, patch_model_linear_layers

def create_simple_model(device='cuda'):
    """Create a simple DNN model for testing."""
    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64)
    )
    return model.to(device).half()  # Move to GPU and convert to fp16

def test_basic_patching():
    """Test basic monkey patching functionality."""
    print("=" * 60)
    print("TEST 1: Basic Monkey Patching")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_simple_model(device)
    input_data = torch.randn(16, 128, device=device, dtype=torch.float16)  # Batch size 16, input dim 128
    
    # Get original output
    with torch.no_grad():
        original_output = model(input_data)
    
    # Patch the model with a relatively high error rate for visible effects
    patch_model_linear_layers(
        model,
        error_prob=1e-4  # Higher error rate to see effects
    )
    
    # Get output with DRAM errors
    with torch.no_grad():
        corrupted_output = model(input_data)
    
    # Check that outputs are different (errors were injected)
    diff = torch.abs(original_output - corrupted_output).mean()
    print(f"Mean absolute difference: {diff:.6f}")
    
    if diff > 1e-6:
        print("✅ PASS: DRAM errors successfully injected (outputs differ)")
    else:
        print("❌ FAIL: No difference detected - errors may not be working")
    
    print()

def test_individual_module_patching():
    """Test patching individual modules."""
    print("=" * 60)
    print("TEST 2: Individual Module Patching")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_simple_model(device)
    input_data = torch.randn(8, 128, device=device, dtype=torch.float16)
    
    # Patch only the first linear layer
    patch_linear_module(
        model[0],  # First Linear layer
        error_prob=5e-4,
        module_name="first_layer"
    )
    
    # Patch the last linear layer with different error rate
    patch_linear_module(
        model[4],  # Last Linear layer (index 4 in sequential)
        error_prob=1e-4,
        module_name="output_layer"
    )
    
    # Run forward pass
    with torch.no_grad():
        output = model(input_data)
    
    print(f"Output shape: {output.shape}")
    print("✅ PASS: Individual module patching completed successfully")
    print()

def test_vector_error_probability():
    """Test using tensor (vector) error probabilities."""
    print("=" * 60)
    print("TEST 3: Vector Error Probability")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_simple_model(device)
    input_data = torch.randn(4, 128, device=device, dtype=torch.float16)
    
    # Create a custom error probability vector
    error_vector = torch.ones(1024, device=device) * 1e-6  # Base low error rate
    error_vector[:100] = 1e-4  # First 100 bits have higher error rate
    error_vector[500:600] = 0  # Some bits have no errors
    error_vector[900:1000] = 5e-4  # Last 100 bits have very high error rate
    
    print(f"Error vector stats:")
    print(f"  Shape: {error_vector.shape}")
    print(f"  Min: {error_vector.min():.2e}")
    print(f"  Max: {error_vector.max():.2e}")
    print(f"  Mean: {error_vector.mean():.2e}")
    
    # Patch with vector error probabilities
    patch_model_linear_layers(
        model,
        error_prob=error_vector
    )
    
    # Run forward pass
    with torch.no_grad():
        output = model(input_data)
    
    print(f"Output shape: {output.shape}")
    print("✅ PASS: Vector error probability test completed successfully")
    print()

def test_error_effects_comparison():
    """Compare outputs with different error rates."""
    print("=" * 60)
    print("TEST 4: Error Effects Comparison")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_data = torch.randn(8, 128, device=device, dtype=torch.float16)
    error_rates = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
    outputs = []
    
    for error_rate in error_rates:
        model = create_simple_model(device)
        
        if error_rate > 0:
            patch_model_linear_layers(
                model,
                error_prob=error_rate
            )
        
        with torch.no_grad():
            output = model(input_data)
            outputs.append(output)
    
    # Compare outputs
    baseline = outputs[0]  # No error case
    print("Error Rate | Mean Abs Diff | Max Abs Diff | Relative Change")
    print("-" * 60)
    
    for i, error_rate in enumerate(error_rates):
        if i == 0:
            print(f"{error_rate:9} | {'0.000000':>13} | {'0.000000':>12} | {'0.00%':>13}")
        else:
            diff = torch.abs(outputs[i] - baseline)
            mean_diff = diff.mean().item()
            max_diff = diff.max().item()
            relative_change = (diff.mean() / torch.abs(baseline).mean() * 100).item()
            
            print(f"{error_rate:9.0e} | {mean_diff:13.6f} | {max_diff:12.6f} | {relative_change:12.2f}%")
    
    print("✅ PASS: Error effects comparison completed")
    print()

def test_model_with_bias():
    """Test with models that have bias terms."""
    print("=" * 60)
    print("TEST 5: Model with Bias Terms")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with bias
    model = nn.Sequential(
        nn.Linear(64, 64, bias=True),
        nn.ReLU(),
        nn.Linear(64, 64, bias=True),
        nn.ReLU(), 
        nn.Linear(64, 10, bias=False)  # No bias for this layer
    ).to(device).half()  # Move to GPU and convert to fp16
    
    input_data = torch.randn(4, 64, device=device, dtype=torch.float16)
    
    # Check which layers have bias
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            has_bias = layer.bias is not None
            print(f"Layer {i}: Linear({layer.in_features}, {layer.out_features}, bias={has_bias})")
    
    # Patch the model
    patch_model_linear_layers(
        model,
        error_prob=1e-4
    )
    
    # Run forward pass
    with torch.no_grad():
        output = model(input_data)
    
    print(f"Final output shape: {output.shape}")
    print("✅ PASS: Model with bias terms test completed")
    print()

def test_size_requirements():
    """Test tensor size requirements (divisible by 64)."""
    print("=" * 60)
    print("TEST 6: Tensor Size Requirements")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with sizes that should work (weight shape divisible by 64)
    # For nn.Linear(in_features, out_features), weight shape is (out_features, in_features)
    # So total elements = in_features * out_features
    good_configs = [
        (64, 64),    # 64*64 = 4096 (divisible by 64)
        (128, 64),   # 128*64 = 8192 (divisible by 64)  
        (64, 128),   # 64*128 = 8192 (divisible by 64)
        (192, 64),   # 192*64 = 12288 (divisible by 64)
    ]
    
    bad_configs = [
        (63, 64),    # 63*64 = 4032 (not divisible by 64)
        (64, 63),    # 64*63 = 4032 (not divisible by 64)
        (65, 64),    # 65*64 = 4160 (not divisible by 64)
        (100, 50),   # 100*50 = 5000 (not divisible by 64)
    ]
    
    print("Testing valid layer configurations (should work):")
    for in_features, out_features in good_configs:
        try:
            model = nn.Linear(in_features, out_features).to(device).half()
            patch_linear_module(model, error_prob=1e-6)
            
            input_data = torch.randn(1, in_features, device=device, dtype=torch.float16)
            with torch.no_grad():
                output = model(input_data)
            print(f"  ✅ Linear({in_features}, {out_features}): SUCCESS (weight elements: {in_features*out_features})")
        except Exception as e:
            print(f"  ❌ Linear({in_features}, {out_features}): FAILED - {str(e)[:60]}...")
    
    print("\nTesting invalid layer configurations (should fail at patch time):")
    for in_features, out_features in bad_configs:
        try:
            model = nn.Linear(in_features, out_features).to(device).half()
            patch_linear_module(model, error_prob=1e-6)
            
            # If we get here, patching unexpectedly succeeded
            input_data = torch.randn(1, in_features, device=device, dtype=torch.float16)
            with torch.no_grad():
                output = model(input_data)
            print(f"  ❌ Linear({in_features}, {out_features}): UNEXPECTED SUCCESS")
        except ValueError as e:
            # Expected failure due to shape validation
            print(f"  ✅ Linear({in_features}, {out_features}): EXPECTED FAILURE - Shape validation caught it")
        except Exception as e:
            print(f"  ⚠️  Linear({in_features}, {out_features}): DIFFERENT ERROR - {type(e).__name__}")
    
    print("\nTesting bias tensor validation:")
    try:
        # Create a layer where bias size is not divisible by 64
        # This is tricky because bias size = out_features, so we need out_features not divisible by 64
        model = nn.Linear(128, 63, bias=True).to(device).half()  # bias will have 63 elements
        patch_linear_module(model, error_prob=1e-6)
        print("  ❌ Bias validation: UNEXPECTED SUCCESS")
    except ValueError as e:
        print("  ✅ Bias validation: EXPECTED FAILURE - Bias size validation works")
    except Exception as e:
        print(f"  ⚠️  Bias validation: DIFFERENT ERROR - {type(e).__name__}")
    
    print()

def main():
    """Run all tests."""
    print("🧪 DRAM Error Simulation Monkey Patch Tests")
    print("=" * 60)
    
    # Check CUDA availability and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"Using dtype: torch.float16")
    print()
    
    # Run all tests
    test_basic_patching()
    test_individual_module_patching()
    test_vector_error_probability()
    test_error_effects_comparison()
    test_model_with_bias()
    test_size_requirements()
    
    print("🎉 All tests completed!")

if __name__ == "__main__":
    main()
