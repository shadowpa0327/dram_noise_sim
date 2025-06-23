import torch
import torch.nn as nn
import types
from typing import Optional, List, Union
from dram_error_simulation import dram_bitflip, _update_dram_stats

def _apply_dram_errors(
        tensor: torch.Tensor, 
        error_prob: Union[float, torch.Tensor], 
        protect_sign_and_exponent: bool = False, 
        context: str = "unknown",
        log_stats: bool = True
    ) -> torch.Tensor:
    """
    Apply DRAM bit-flip errors to a tensor, handling dtype conversion as needed.
    """
    # Handle zero probability cases
    if isinstance(error_prob, (int, float)) and error_prob == 0.0:
        return tensor
    elif isinstance(error_prob, torch.Tensor) and torch.all(error_prob == 0.0):
        return tensor
    
    # Convert to float16 if necessary
    original_dtype = tensor.dtype
    if original_dtype != torch.float16:
        tensor_f16 = tensor.to(torch.float16)
    else:
        tensor_f16 = tensor
    
    # Apply DRAM bit-flip errors (will raise error if numel() % 64 != 0)
    corrupted_tensor, dmg_mask, bits_flipped, bit_pos_flips = dram_bitflip(
        tensor_f16, 
        error_prob, 
        inplace=False,
        protect_sign_and_exponent=protect_sign_and_exponent
    )
    
    # Log statistics
    if log_stats:
        _update_dram_stats(context, bits_flipped, 1, tensor.numel(), bit_pos_flips)
    
    # Convert back to original dtype if necessary
    if original_dtype != torch.float16:
        corrupted_tensor = corrupted_tensor.to(original_dtype)
    
    return corrupted_tensor

def create_dram_error_forward(
    original_forward,
    error_prob: Union[float, torch.Tensor] = 1e-6,
    module_name: str = "Linear",
    protect_sign_and_exponent: bool = False,
    log_stats: bool = True
):
    """
    Create a new forward function that applies DRAM errors before calling the linear operation.
    """
    def forward_with_dram_errors(self, input: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        # Apply DRAM errors to input activations
        corrupted_input = _apply_dram_errors(input, error_prob, protect_sign_and_exponent, f"{module_name}_input", log_stats)
        
        # # Apply DRAM errors to weights
        corrupted_weight = None
        if hasattr(self, 'weight') and self.weight is not None:
            corrupted_weight = _apply_dram_errors(self.weight, error_prob, protect_sign_and_exponent, f"{module_name}_weight", log_stats)
        
        # # Apply DRAM errors to bias if present
        corrupted_bias = None
        if hasattr(self, 'bias') and self.bias is not None:
            corrupted_bias = _apply_dram_errors(self.bias, error_prob, protect_sign_and_exponent, f"{module_name}_bias", log_stats)
        
        # Call F.linear directly with corrupted tensors
        return F.linear(corrupted_input, corrupted_weight, corrupted_bias)
    
    return forward_with_dram_errors

def patch_linear_module(
    module: nn.Linear,
    error_prob: Union[float, torch.Tensor] = 1e-6,
    module_name: str = None,
    protect_sign_and_exponent: bool = False,
    log_stats: bool = True
) -> nn.Linear:
    """
    Patch a single nn.Linear module to inject DRAM errors.
    
    Parameters
    ----------
    module : nn.Linear
        The Linear module to patch
    error_prob : float or torch.Tensor, default 1e-6
        Per-bit flip probability(ies). Can be:
        - A scalar float: 0 <= p <= 1, same probability for all bits
        - A tensor of shape (1024,): individual probabilities for each bit position
          within the 1024-bit DRAM burst, with all elements in [0, 1]
    module_name : str, optional
        Name for debugging/logging purposes
    protect_sign_and_exponent : bool, default False
        Whether to protect the sign and exponent bits from DRAM errors
        
    Returns
    -------
    nn.Linear
        The patched module (same object, modified in-place)
    """
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected nn.Linear module, got {type(module)}")
    
    # Check tensor shapes - weights and bias must have numel() divisible by 64
    if hasattr(module, 'weight') and module.weight is not None:
        weight_numel = module.weight.numel()
        if weight_numel % 64 != 0:
            raise ValueError(
                f"Weight tensor has {weight_numel} elements, but DRAM error simulation requires "
                f"tensors with element count divisible by 64. Current weight shape: {module.weight.shape}"
            )
    
    if hasattr(module, 'bias') and module.bias is not None:
        bias_numel = module.bias.numel()
        if bias_numel % 64 != 0:
            raise ValueError(
                f"Bias tensor has {bias_numel} elements, but DRAM error simulation requires "
                f"tensors with element count divisible by 64. Current bias shape: {module.bias.shape}"
            )
    
    # Store original forward method and config
    module._original_forward = module.forward
    module._dram_error_config = {
        'error_prob': error_prob,
        'module_name': module_name or f"Linear_{id(module)}",
        'protect_sign_and_exponent': protect_sign_and_exponent
    }
    
    # Replace forward method using types.MethodType
    new_forward = create_dram_error_forward(
        module._original_forward,
        error_prob,
        module_name or "Linear",
        protect_sign_and_exponent,
        log_stats
    )
    module.forward = types.MethodType(new_forward, module)
    
    print(f"Patched {module._dram_error_config['module_name']} with DRAM error simulation")
    if isinstance(error_prob, torch.Tensor):
        print(f"  - Error prob: tensor of shape {error_prob.shape} (min: {error_prob.min():.2e}, max: {error_prob.max():.2e})")
    else:
        print(f"  - Error prob: {error_prob}")
    print(f"  - Protect sign/exponent: {protect_sign_and_exponent}")
    
    return module

def patch_model_linear_layers(
    model: nn.Module,
    error_prob: Union[float, torch.Tensor] = 1e-6,
    layer_filter: Optional[callable] = None,
    protect_sign_and_exponent: bool = False,
    log_stats: bool = True
) -> List[nn.Linear]:
    """
    Patch all Linear layers in a model.
    
    Parameters
    ----------
    model : nn.Module
        The model containing Linear layers to patch
    error_prob : float or torch.Tensor, default 1e-6
        Per-bit flip probability(ies). Can be:
        - A scalar float: 0 <= p <= 1, same probability for all bits
        - A tensor of shape (1024,): individual probabilities for each bit position
          within the 1024-bit DRAM burst, with all elements in [0, 1]
    layer_filter : callable, optional
        Function to filter which layers to patch. Should take (name, module) and return bool
    protect_sign_and_exponent : bool, default False
        Whether to protect the sign and exponent bits from DRAM errors
        
    Returns
    -------
    List[nn.Linear]
        List of patched Linear modules
    """
    patched_modules = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Apply filter if provided
            if layer_filter is not None and not layer_filter(name, module):
                continue
                
            patch_linear_module(
                module,
                error_prob,
                name,
                protect_sign_and_exponent,
                log_stats
            )
            patched_modules.append(module)
    
    print(f"Patched {len(patched_modules)} Linear layers in model")
    return patched_modules
