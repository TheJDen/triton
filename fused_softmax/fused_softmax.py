"""
- reduce memory read/writes by fusing
- how to get GPU specifications
- more GPU arch details
- how to define meta-parameters using heuristics and GPU-specific attributes
- more about masking and how to choose the value of extra masked-out entries
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU Properties
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
TOTAL_SRAM_PER_SM = properties["max_shared_mem"]
WARP_SIZE = properties["warp_size"]


def naive_softmax(x: torch.Tensor):
    # assume input size is (M, N)

    # reads M x N elements and writes M elements
    x_max = x.max(dim=1)[0] # shape (M,)

    # read M x N + M elements, subtraction is M x N flops, write M x N elements
    z = x - x_max[:, None] # shape (M, N) - shape (M, 1) = shape (M, N)

    # read M x N elements, write M x N elements
    numerator = torch.exp(z) # shape (M, N)

    # read M x N elements, sum is M x N flops, write M elements
    denominator = numerator.sum(dim=1) # shape (M, N) -> shape (M,)

    # read M x N + M elements, division is M x N flops, write M x N elements
    out = numerator / denominator[:, None] # shape (M, N) / shape (M, 1) = shape (M, N)

    # in total we did 8 x M x N + 4 x M memory operations 
    return out

def softmax(x: torch.Tensor):
    assert x.ndim == 2
    n_rows, n_cols = x.shape

    # assume rows fit in SRAM
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 4
    if BLOCK_SIZE > 2048:
        num_warps = 8
    if BLOCK_SIZE > 4096:
        num_warps = 16

    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2

    y = torch.empty_like(x)

    kernel = _softmax_kernel.warmup(
        x, y
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,)
    )
    kernel._init_handles()
    n_regs_per_program = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared

    reg_occupance = NUM_REGS // n_regs_per_program

def test_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3, device=DEVICE):
    assert type(size) == tuple and len(size) == 2
    torch.manual_seed(30)
    x = torch.randn(size[0], size[1], device=device)
    z_triton = softmax(x)
    z_torch = torch.softmax(x, dim=1)
    torch.testing.assert_close(z_triton, z_torch, atol=atol, rtol=rtol)
    print("PASSED")

