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

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
TOTAL_SRAM_PER_SM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

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
    assert x.is_contiguous()
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
        x, y,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,)
    )
    assert kernel is not None
    kernel._init_handles()
    n_regs_per_program = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared

    reg_occupancy = NUM_REGS // (n_regs_per_program * WARP_SIZE * num_warps)
    # EXAMPLE
    # NUM_REGS = 65536 (64 KB)
    # each program might use 
    #     n_regs_per_program = 32
    #     WARP_SIZE = 32
    #     num_warps = 8
    # so each program needs (n_regs_per_program * WARP_SIZE * num_warps) registers total
    # all programs in SM share registers
    # therefore we may fit reg_occupancy programs on each SM
    # 65536 // (32 * 32 * 8) = 8 programs per SM

    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program

    programs_per_sm = min(reg_occupancy, sram_occupancy)

    num_programs = min(NUM_SM * programs_per_sm, n_rows)

    grid = (num_programs, 1, 1)

    kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE,
        num_stages
    )

    # x.stride()
    # x is shape (M, N)
    # x.stride() would be (N, 1)
    # x.stride(0) would be N
    # x.stride(1) would be 1
    # z is shape (B, N, D)
    # z.stride() is (N x D, D, 1)
    # each entry is proportional to product of suffix

    return y

@triton.jit
def _softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr
):
    # shape (M, N)
    # BLOCK_SIZE = next power of 2 bigger than N
    pid = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(pid, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=float("-inf")) # shape (BLOCK_SIZE) roughly (n_cols)

        row_minus_max = row - tl.max(row, axis=0) # shape (BLOCK_SIZE) - (1) -> (BLOCK_SIZE)
        numerator = tl.exp(row_minus_max) # shape (BLOCK_SIZE)
        denominator = tl.sum(numerator, axis=0) # shape (1)
        softmax_output = numerator / denominator # shape (BLOCK_SIZE) / (1) -> (BLOCK_SIZE)

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


def test_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3, device=DEVICE):
    assert type(size) == tuple and len(size) == 2
    torch.manual_seed(30)
    x = torch.randn(size[0], size[1], device=device)
    z_triton = softmax(x)
    z_torch = torch.softmax(x, dim=1)
    torch.testing.assert_close(z_triton, z_torch, atol=atol, rtol=rtol)
    print("PASSED")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals = [128 * i for i in range(2, 100)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", '-'), ('green', '-')],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={'M': 4096}

    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    else:
        ms = triton.testing.do_bench(lambda: softmax(x)) 
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

if __name__ == "__main__":
    test_softmax_kernel(size=(1823, 781))

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        benchmark.run(save_path='.', print_data=False)