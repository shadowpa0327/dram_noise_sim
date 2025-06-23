import torch
import triton
import triton.language as tl
from typing import Optional


################################################################################
# Triton kernel: pack 16 × {0,1} → 1 × int16 and XOR in place
################################################################################
@triton.jit
def _pack_and_xor(
    x_ptr,              # int16* -- length = N_words
    rb_ptr,             # int8* -- length = N_words * 16 (rand_bits)
    n_words: tl.int32,  # total number of 16‑bit words
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)                      # one program per block
    offs = pid * BLOCK + tl.arange(0, BLOCK)          # [BLOCK] global word indices
    mask = offs < n_words

    # ------------------------------------------------------------------ #
    # 1. Load 16‑bit words to be damaged
    # ------------------------------------------------------------------ #
    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.int16)

    # ------------------------------------------------------------------ #
    # 2. Load 16 independent Bernoulli bits per word      shape → [BLOCK,16]
    # ------------------------------------------------------------------ #
    bit_offs = offs[:, None] * 16 + tl.arange(0, 16)[None, :]
    rb = tl.load(rb_ptr + bit_offs, mask=mask[:, None], other=0).to(tl.int16)

    # 2.1 Pack the 16 bits into one int16
    shifts = (1 << tl.arange(0, 16)).to(tl.int16)    # [16]
    dmg_mask = tl.sum(rb * shifts[None, :], 1).to(tl.int16)   # [BLOCK]

    # ------------------------------------------------------------------ #
    # 3. Flip the bits
    # ------------------------------------------------------------------ #
    x ^= dmg_mask

    # ------------------------------------------------------------------ #
    # 4. Store back
    # ------------------------------------------------------------------ #
    tl.store(x_ptr + offs, x, mask=mask)


################################################################################
# Python wrapper that plugs the Triton kernel into your original routine
################################################################################
def pack_and_xor_triton(x: torch.Tensor,
                        rand_bits: torch.Tensor  # (N_words,16) uint8 / bool
                       ):
    """
    Parameters
    ----------
    x : float16 tensor that has already been reshaped to (N_words,) **int16 view**
        *i.e.* `x_int = x.view(torch.int16).contiguous()`
    rand_bits : uint8 / bool tensor of shape (N_words, 16) holding the
                pre‑sampled Bernoulli(p) outcomes.
    """
    if x.dtype != torch.int16 or rand_bits.shape[1] != 16:
        raise ValueError("Inputs must be int16 view and (N_words,16) bits")

    N_WORDS = x.numel()
    BLOCK = 128                    # 1024 × int16 = 2 KiB per CTA
    GRID = (triton.cdiv(N_WORDS, BLOCK),)

    _pack_and_xor[GRID](
        x_ptr=x,                    # int16 view -> in‑place
        rb_ptr=rand_bits,
        n_words=N_WORDS,
        BLOCK=BLOCK,
    )

    return x


def dram_bitflip(
    x: torch.Tensor,
    p: torch.Tensor | float,
    *,
    generator: Optional[torch.Generator] = None,
    inplace: bool = False,
    protect_sign_and_exponent: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Inject independent bit‑flip noise into a (d_in, d_out) float16 tensor,
    assuming a DRAM burst size of 1024 bits (=16 × float16 words).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of dtype torch.float16 (any shape).
    p : float or torch.Tensor
        Per‑bit flip probability(ies). Can be:
        - A scalar float: 0 <= p <= 1, same probability for all bits
        - A tensor of shape (1024,): individual probabilities for each bit position
          within the 1024-bit DRAM burst, with all elements in [0, 1]
    generator : torch.Generator, optional
        For deterministic sampling (e.g. torch.Generator(device='cuda').manual_seed(0)).
    inplace : bool, default False
        If True, corrupt `x` in place; otherwise a new tensor is returned.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (corrupted_tensor, damage_mask) where corrupted_tensor has the same 
        shape & dtype as input but with randomly flipped bits, and damage_mask shows
        which bits were flipped.
    """
    if x.dtype != torch.float16:
        raise TypeError("Input must be float16")

    if x.numel() % 64 != 0:
        raise ValueError("x must have a number of elements divisible by 64")

    # Handle p validation and conversion
    if isinstance(p, (int, float)):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        p_tensor = None  # Use scalar p directly
    else:
        # p is a tensor
        if not isinstance(p, torch.Tensor):
            raise TypeError("p must be a float or torch.Tensor")
        if p.shape != (1024,):
            raise ValueError("p tensor must have shape (1024,)")
        if torch.any(p < 0.0) or torch.any(p > 1.0):
            raise ValueError("All elements of p must be in [0, 1]")
        p_tensor = p.to(x.device)  # Ensure p is on the same device as x

    # Work with a view that groups the data in 1024‑bit bursts (16 float16 words)
    x_view = x.view(-1, 64)                               # shape (N_bursts, 16)
    # Re‑interpret those 16‑bit words as signed int16 so bitwise ops are allowed
    x_int = x_view.view(torch.int16)                      # same storage, no copy

    if not inplace:
        x_int = x_int.clone()                             # preserve the original

    # ------------------------------------------------------------------
    # HIGH-LEVEL LOGIC OVERVIEW:
    # 1. Reshape data into 1024-bit DRAM bursts (64 float16 words per burst)
    # 2. For each burst, generate random bit-flip mask based on probability p
    # 3. Apply bit-flips using XOR operation
    # 4. Return corrupted data with same shape as input
    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    # 1. Generate random bit-flip decisions for each bit position
    # ------------------------------------------------------------------
    # Strategy: Generate uniform random [0,1] values, then compare with p
    # This gives us Boolean decisions for whether each bit should flip
    N, W = x_int.shape
    assert W == 64, "W must be 64"
    
    # Generate random values for all bit positions across all bursts
    # Shape: (N_bursts, 64_words * 16_bits_per_word) = (N, 1024)
    rand_uniform = torch.rand((N, 64*16), device=x.device, generator=generator)
    
    if p_tensor is None:
        # Uniform probability: same p for all bit positions
        rand_bits = rand_uniform < p
    else:
        # Per-bit probabilities: p_tensor[i] for bit position i within 1024-bit burst
        # Broadcast p_tensor (1024,) across N bursts -> (N, 1024)
        rand_bits = rand_uniform < p_tensor.unsqueeze(0)
    
    # ------------------------------------------------------------------
    # 2. Build bit-flip masks for each 16-bit word
    # ------------------------------------------------------------------
    # Reshape: (N, 1024) -> (N, 64_words, 16_bits_per_word)
    rand_bits = rand_bits.view(N, 64, 16) 
    
    # Optional protection: Don't flip sign and exponent bits (top 6 bits of float16)
    # NOTE: rand_bits is in reverse bit order, so -6: means bits [15,14,13,12,11,10]
    if protect_sign_and_exponent:
        rand_bits[:, :, -6:] = False
    
    # Convert Boolean decisions to actual bit masks:
    # For each word, pack 16 Boolean values into a single int16 mask
    # rand_bits[i,j,k] -> bit k of word j in burst i
    # bit_shifts = [2^0, 2^1, 2^2, ..., 2^15] to create positional bit values
    bit_shifts = torch.arange(16, device=x.device, dtype=torch.int16)\
                  .view(1, 1, 16)
    # Sum across bit positions: each True bit contributes 2^position to the mask
    dmg_mask = (rand_bits.to(torch.int16) << bit_shifts).sum(dim=2).to(torch.int16)

    # ------------------------------------------------------------------
    # 3. Apply bit-flips using XOR operation
    # ------------------------------------------------------------------
    # XOR each word with its corresponding damage mask:
    # - Bits set to 1 in dmg_mask will be flipped in x_int
    # - Bits set to 0 in dmg_mask remain unchanged in x_int
    # This simulates the effect of DRAM bit-flip errors
    x_int ^= dmg_mask                                     # in‑place within the view

    # ------------------------------------------------------------------
    # 4. Return results with original tensor shape and dtype
    # ------------------------------------------------------------------
    return x_int.view(torch.float16).view_as(x), dmg_mask


def dram_bitflip_triton(
    x: torch.Tensor,
    p: torch.Tensor | float,
    *,
    generator: Optional[torch.Generator] = None,
    inplace: bool = False,
    protect_sign_and_exponent: bool = False,
) -> torch.Tensor:
    """
    Triton-accelerated version of dram_bitflip that uses the pack_and_xor_triton kernel
    for faster bit packing and XOR operations.
    
    Parameters are the same as dram_bitflip(), but returns only the corrupted tensor.
    """
    if x.dtype != torch.float16:
        raise TypeError("Input must be float16")

    # Handle p validation and conversion (same as original)
    if isinstance(p, (int, float)):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        p_tensor = None  # Use scalar p directly
    else:
        # p is a tensor
        if not isinstance(p, torch.Tensor):
            raise TypeError("p must be a float or torch.Tensor")
        if p.shape != (1024,):
            raise ValueError("p tensor must have shape (1024,)")
        if torch.any(p < 0.0) or torch.any(p > 1.0):
            raise ValueError("All elements of p must be in [0, 1]")
        p_tensor = p.to(x.device)  # Ensure p is on the same device as x

    # Work with a view that groups the data in 1024‑bit bursts (16 float16 words)
    x_view = x.view(-1, 64)                               # shape (N_bursts, 16)
    # Re‑interpret those 16‑bit words as signed int16 so bitwise ops are allowed
    x_int = x_view.view(torch.int16)                      # same storage, no copy

    if not inplace:
        x_int = x_int.clone()                             # preserve the original

    # ------------------------------------------------------------------
    # 1. Generate random bits in the format expected by Triton kernel
    # ------------------------------------------------------------------
    N, W = x_int.shape
    assert W == 64, "W must be 64"
    
    # Generate uniform random numbers
    rand_uniform = torch.rand((N, 64*16), device=x.device, generator=generator)
    
    if p_tensor is None:
        # Scalar p case (original behavior)
        rand_bits = rand_uniform < p
    else:
        # Vector p case: broadcast p_tensor across N bursts
        # p_tensor has shape (1024,), rand_uniform has shape (N, 1024)
        rand_bits = rand_uniform < p_tensor.unsqueeze(0)  # p_tensor becomes (1, 1024)
    
    # Reshape for Triton kernel: (N, 64*16) -> (N*64, 16)
    # Each "word" (int16) gets 16 bits
    rand_bits = rand_bits.view(N, 64, 16)
    
    # Force first 6 bits to be error-free by setting them to False
    # NOTE(brian1009): The rand_bits is in reverse order. 
    if protect_sign_and_exponent:
        rand_bits[:, :, -6:] = False
    
    rand_bits_triton = rand_bits.view(-1, 16).to(torch.uint8)  # (N*64, 16) as uint8
    
    # Flatten x_int to (N*64,) for Triton kernel
    x_int_flat = x_int.view(-1).contiguous()  # (N*64,) int16 tensor
    
    # ------------------------------------------------------------------
    # 2. Call Triton kernel to pack bits and XOR in one go
    # ------------------------------------------------------------------
    pack_and_xor_triton(x_int_flat, rand_bits_triton)
    
    # The Triton kernel modifies x_int_flat in-place, and since it's a view
    # of x_int, the changes are reflected in the original tensor structure

    # Return with original shape & dtype
    return x_int.view(torch.float16).view_as(x) 