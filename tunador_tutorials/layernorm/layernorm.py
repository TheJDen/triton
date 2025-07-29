"""
- backwards passes
- connecting bwd to pytorch graph
- reuse intermediate values from fwd in bwd
- locks & atomics
- two sequential kernels can be faster than one single kernel
"""

import torch
import triton
import triton.language as tl
DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")




@triton.jit
def _layernorm_forward(
    x_ptr, y_ptr, weight_ptr, bias_ptr, mean_ptr, rstd_ptr,
    stride_M, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    x_ptr += row * stride_M
    y_ptr += row * stride_M

    sum_accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.)
        sum_accumulator += x
    mean = tl.sum(sum_accumulator) / N

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.)
        diff = tl.where(cols < N, x - mean, 0.)
        acc += diff * diff
    var = tl.sum(acc, axis=0) / N # shape (BLOCK_SIZE,) -> shape (1,)
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(weight_ptr + cols, mask=mask)
        b = tl.load(bias_ptr + cols, mask=mask)
        x = tl.load(x_ptr + cols, mask=mask)

        x_normed = (x - mean) * rstd
        y = x_normed * w + b
        tl.store(y_ptr + cols, y, mask=mask)

@triton.jit
def _layernorm_backward_dLdx(
    x_ptr, dLdx_ptr, dLdy_ptr, weight_ptr,
    dLdw_intermediate_ptr, dLdb_intermediate_ptr,
    mean_ptr, rstd_ptr,
    locks_ptr,
    stride, N,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    x_ptr += pid * stride
    dLdx_ptr += pid * stride
    dLdy_ptr += pid * stride

    x = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)
    dLdy = tl.load(dLdy_ptr + cols, mask=mask, other=0.).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask).to(tl.float32)
    mean = tl.load(mean_ptr + pid)
    rstd = tl.load(rstd_ptr + pid)

    x_normed = tl.where(mask, (x - mean) * rstd, 0.)
    dydx_normed = tl.where(mask, dLdy * w, 0.)
    c1 = tl.sum(x_normed * dydx_normed, axis=0) / N
    c2 = tl.sum(dydx_normed, axis=0) / N
    dLdx = (dydx_normed - (x_normed * c1 + c2)) * rstd
    tl.store(dLdx_ptr + cols, dLdx, mask=mask)

    dLdw_intermediate = (dLdy * x_normed).to(w.dtype)
    dLdb_intermediate = dLdy.to(w.dtype)

    lock_id = pid % GROUP_SIZE
    locks_ptr += lock_id
    count_ptr = locks_ptr + GROUP_SIZE

    dLdw_intermediate_ptrs = dLdw_intermediate_ptr + lock_id * N + cols
    dLdb_intermediate_ptrs = dLdb_intermediate_ptr + lock_id * N + cols

    while tl.atomic_cas(locks_ptr, 0, 1) == 1:
        pass
        # if it's 0 (unlocked), change it to 1 (lock it) and return 0 so we leave th loop
        # if it's 1 (locked), leave it at 1 and return 1 so we continue the loop

    count = tl.load(count_ptr)
    if count == 0:
        tl.atomic_xchg(count_ptr, 1)
    else:
        dLdw_intermediate += tl.load(dLdw_intermediate_ptrs, mask=mask)
        dLdb_intermediate += tl.load(dLdb_intermediate_ptrs, mask=mask)

    tl.store(dLdw_intermediate_ptrs, dLdw_intermediate, mask=mask)
    tl.store(dLdb_intermediate_ptrs, dLdb_intermediate, mask=mask)

    tl.atomic_xchg(locks_ptr, 0)

@triton.jit
def _layernorm_backward_dLdw_dLdb(
    dLdw_intermediate_ptr, dLdb_intermediate_ptr,
    dLdw_ptr, dLdb_ptr,
    GROUP_SIZE, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    col_offsets = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    dLdw_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    dLdb_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, GROUP_SIZE, BLOCK_SIZE_M):
        row_offsets = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (row_offsets[:, None] < GROUP_SIZE) & (col_offsets[None, :] < N)
        offsets = row_offsets[:, None] * N + col_offsets[None, :]

        dLdw_acc += tl.load(dLdw_intermediate_ptr + offsets, mask=mask, other=0.)
        dLdb_acc += tl.load(dLdb_intermediate_ptr + offsets, mask=mask, other=0.)

    dLdw_chunk = tl.sum(dLdw_acc, axis=0) # shape (BLOCK_SIZE_N,)
    dLdb_chunk = tl.sum(dLdb_acc, axis=0) # shape (BLOCK_SIZE_N,)
    
    tl.store(dLdw_ptr + col_offsets, dLdw_chunk, mask = col_offsets < N)
    tl.store(dLdb_ptr + col_offsets, dLdb_chunk, mask = col_offsets < N)

class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        normalized_shape,
        weight,
        bias,
        eps
    ):
        M, N = x.reshape(-1, x.shape[-1]).shape
        y = torch.empty_like(x)
        mean = torch.empty((M, ), dtype=torch.float32, device=DEVICE)
        rstd = torch.empty((M, ), dtype=torch.float32, device=DEVICE)

        MAX_FUSED_SIZE = 655536 // x.element_size() # 64kB
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        _layernorm_forward[(M,)](
            x, y, weight, bias, mean, rstd,
            x.stride(0), N, eps,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps

        return y

    @staticmethod
    def backward(ctx, dLdy):
        x, w, b, mean, rstd = ctx.saved_tensors
        M, N = x.reshape(-1, x.shape[-1]).shape

        dLdx = torch.empty_like(x) # (M, N)
        dLdw = torch.empty_like(w) # (N,)
        dLdb = torch.empty_like(b) # (N,)

        GROUP_SIZE = 64
        if N <= 8192: GROUP_SIZE = 96
        if N <= 4096: GROUP_SIZE = 128
        if N <= 1024: GROUP_SIZE = 256

        dLdw_intermediate = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=w.device)
        dLdb_intermediate = torch.zeros((GROUP_SIZE, N), dtype=x.dtype, device=b.device)

        locks = torch.zeros(2 * GROUP_SIZE, dtype=torch.int32, device=w.device)

        _layernorm_backward_dLdx[(M,)](
            x, dLdx, dLdy, w, dLdw_intermediate, dLdb_intermediate,
            mean, rstd, locks, x.stride(0), N,
            GROUP_SIZE=GROUP_SIZE, BLOCK_SIZE_N=ctx.BLOCK_SIZE, num_warps=ctx.num_warps
        )
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
        _layernorm_backward_dLdw_dLdb[grid](
            dLdw_intermediate, dLdb_intermediate, dLdw, dLdb,
            min(GROUP_SIZE, M), N,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=128
        )
        return dLdx, None, dLdw, dLdb, None

layernorm = LayerNorm.apply

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["Pytorch", "Triton"],
        styles=[('green', '-'), ('blue', '-')],
        ylabel="GB/s",
        plot_name="layernorm-backward-performance",
        args={"M": 4096, "dtype": torch.float16, "mode": "backward"}   
    )
)

def benchmark(M, N, dtype, provider, mode="backward", eps=1e-5, device=DEVICE):
    x = -2.3 + 0.5 * torch.randn((M, N), dtype=dtype, device=device)
    x.requires_grad_(True)
    weight = torch.rand((N,), dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand((N,), dtype=dtype, device=device, requires_grad=True)
    quantiles = [0.5, 0.05, 0.95]
    def y_fwd():
        if provider == "torch":
            return torch.nn.functional.layer_norm(x, (N,), weight, bias, eps)
        elif provider == "triton":
            return layernorm(x, (N,), weight, bias, eps)
    y = y_fwd()
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    dLdy = 0.1 * torch.randn_like(x)
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: y.backward(dLdy, retain_graph=True),
        quantiles=quantiles,
        grad_to_none=[x],
        rep=500
    )
    return gbps(ms), gbps(max_ms), gbps(min_ms)

def test_layernorm_kernel(M, N, dtype, eps=1e-5, device=DEVICE):
    x = -2.3 + 0.5 * torch.randn((M, N), dtype=dtype, device=device)
    x.requires_grad_(True)
    weight = torch.rand((N,), dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand((N,), dtype=dtype, device=device, requires_grad=True)
    y_triton = layernorm(x, (N,), weight, bias, eps)
    y_torch = torch.nn.functional.layer_norm(x, (N,), weight, bias, eps).to(dtype)
    torch.testing.assert_close(y_triton, y_torch, atol=1e-2, rtol=0)
    print("passed fwd")

    dLdy = 0.1 * torch.randn_like(x)
    y_triton.backward(dLdy, retain_graph=True)
    dLdx_triton, dLdw_triton, dLdb_triton = [tensor.grad.clone() for tensor in (x, weight, bias)]
    x.grad, weight.grad, bias.grad = None, None, None

    y_torch.backward(dLdy)
    dLdx_torch, dLdw_torch, dLdb_torch = [tensor.grad.clone() for tensor in (x, weight, bias)]

    torch.testing.assert_close(dLdx_triton, dLdx_torch, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdw_triton, dLdw_torch, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdb_triton, dLdb_torch, atol=1e-2, rtol=0)
    print("passed bwd")

if __name__ == "__main__":
    test_layernorm_kernel(1151, 8192, torch.float16)

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)