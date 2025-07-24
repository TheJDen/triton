"""
- basics
- tests
- benchmarks
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {DEVICE}")

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    n_elements,
    block_size: tl.constexpr, # static argument known at compile time
):
    pid = tl.program_id(axis=0) # one of 0, 1, ... , grid

    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements # make sure read does not overflow buffer

    # load data from HBM to SRAM
    x = tl.load(x_ptr + offsets, mask=mask) # shape (BLOCK_SIZE)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    # write back to HBM
    tl.store(z_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # pre-allocate output
    z = torch.empty_like(x)

    # check tensors on same device
    assert x.device == y.device == z.device

    # define launch grid
    n_elements = z.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["block_size"]),)

    add_kernel[grid](x, y, z, n_elements, block_size=1024)

    return z

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[1 << i for i in range(12, 28)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name="vector-addition-performance",
        args={},
    )
)
def benchmark(size, provider):
    # create input data
    x = torch.randn(size, device=DEVICE, dtype=torch.float32)
    y = torch.randn(size, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.05, 0.95]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3) # ops * n_elements_ n_size GB / s
    return gbps(ms), gbps(max_ms), gbps(min_ms)

def test_add_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE):
    torch.manual_seed(34)

    x = torch.randn(size, device=DEVICE)
    y = torch.randn(size, device=DEVICE)

    z_triton = add(x, y)
    z_torch = x + y

    torch.testing.assert_close(z_triton, z_torch, atol=atol, rtol=rtol)
    print(f"Test passed for size {size}")

if __name__ == "__main__":
    test_add_kernel(size=4096)
    test_add_kernel(size=4097)
    test_add_kernel(size=98432)

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=True)
