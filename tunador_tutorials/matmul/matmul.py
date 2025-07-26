"""
- automatic performance tuning
- PID reordering for improved SRAM sharing between PIDs
- multi-dimensional pointer arithmetic
- data types; high-precision accumulation
- triton interpreter for improved debugging

A @ B = C
(M, K) @ (K, N) -> (M, N)
for m in range(M):
    for n in range(N):
        a_vec = A[m, :]
        b_vec = B[:, n]
        C[m, n] = dot(a_vec, b_vec)

A @ B = C
(M, K) @ (K, N) -> (M, N)
for m in range(0, M, BLOCK_SIZE_M): # parallel, each iteration is own pid
    for n in range(0, N, BLOCK_SIZE_N): # parallel, each iteration is own pid
        acc = tl.zeros(shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_SIZE_K):
            a = A[m: m + BLOCK_SIZE_M, k: k + BLOCK_SIZE_K]
            b = B[k: k + BLOCK_SIZE_K, n: n + BLOCK_SIZE_N] 
            acc += tl.dot(a, b)
        C[m: m + BLOCK_SIZE_M, n: n + BLOCK_SIZE_N] = acc
"""
import torch
import triton
import triton.language as tl

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

# how to debug
# import os
# os.environ["TRITON_INTERPRET"] = "1"
# will run with numpy simulator instead of on GPU allowing things like print statements, pdb, etc

# BLOCK_SIZE_M, BLOCK_SIZE_N
autotune_configs = [ # heuristically chosen, can do exhaustively with loops
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2), 
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2)
]

@triton.autotune(configs=autotune_configs, key=["M", "N", "K"])
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_a_M, stride_a_K,
    stride_b_K, stride_b_N,
    stride_c_M, stride_c_N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    """
    M = N = K = 8
    BLOCK_SIZE_M/N/K = 2
    [0,   1,  2,  3]
    [4,   5,  6,  7]
    [8,   9, 10, 11]
    [12, 13, 14, 15] 
    each num represents a 2 x 2 chunk of C (a block)
        A        @     B        =     C 
    [x, x, x, x]   [x, _, _, _]   [0, _, _, _]
    [_, _, _, _]   [x, _, _, _]   [_, _, _, _]
    [_, _, _, _]   [x, _, _, _]   [_, _, _, _]
    [_, _, _, _]   [x, _, _, _]   [_, _, _, _]
        A        @     B       
    [--------->]   [|, _, _, _]
    [_, _, _, _]   [|, _, _, _]
    [_, _, _, _]   [|, _, _, _]
    [_, _, _, _]   [v, _, _, _]

    we know each SM has its own SRAM pool and Triton is smart enough to not duplicate reads
    when PIDs belonging to the same SM load the same data
    So, we want to maximize shared memory between successive PIDs to minimize duplicate reads
    and we can achieve this by being intentional about which PIDs are assigned to the same SM

    Notice PIDs 0, 1, 2, 3 from visual up above all would read the first row of A for accumulating dot products
    But they all read different columns, (columns 0, 1, 2, 3, etc), effectively reading the entirety of B (no SM reuse on cols)
    This is 1 row read + 4 col reads = 5 reads

    Notice PIDs 0, 1, 4, 5 from visual up above all would read the first and second row of A (0, 1 read first, 4, 5 read second)
    And they read the first and second columns, (0, 4, read first, 1, 5, read second)
    This is 2 row reads + 2 col reads = 4 reads

    By grouping PIDs instead of assigning in row-major order, we reduced the number of reads by 1
    
    Group indexing scheme
    [0,  2,  4,  6]
    [1,  3,  5,  7]
    [8, 10, 12, 14]
    [9, 11, 13, 15]


    """
    pid = tl.program_id(axis=0)
    num_pid_along_M = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_along_N = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE * num_pid_along_N
    group_id = pid // num_pid_in_group
    first_pid_in_group_along_M = group_id * GROUP_SIZE
    group_size_adj = min(num_pid_along_M - first_pid_in_group_along_M, GROUP_SIZE)
    n_index, m_index = (pid % num_pid_in_group) // group_size_adj, (pid % num_pid_in_group) % group_size_adj
    pid_m = first_pid_in_group_along_M + m_index
    pid_n = n_index

    offsets_M = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)
    a_offsets = offsets_M[:, None] * stride_a_M + offsets_K[None, :] * stride_a_K
    b_offsets = offsets_K[:, None] * stride_b_K + offsets_N[None, :] * stride_b_N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask = offsets_K < (K - k * BLOCK_SIZE_K )
        a = tl.load(a_ptr + a_offsets, mask=mask[None, :], other=0.0)
        b = tl.load(b_ptr + b_offsets, mask=mask[:, None], other=0.0)

        acc = tl.dot(a, b, acc=acc)

        a_offsets += BLOCK_SIZE_K * stride_a_K
        b_offsets += BLOCK_SIZE_K * stride_b_K

    c_offsets = offsets_M[:, None] * stride_c_M + offsets_N[None, :] * stride_c_N
    c_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N)
    tl.store(c_ptr + c_offsets, acc.to(tl.float16), mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.ndim == b.ndim == 2
    assert a.shape[1] == b.shape[0]

    (M, K), (_, N) = a.shape, b.shape
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
    """
    [0,   1,  2,  3]
    [4,   5,  6,  7]
    [8,   9, 10, 11]
    [12, 13, 14, 15]
    """
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    ) # (16,)
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),

    )
    return c
    

def test_matmul_kernel(size: tuple, atol=1e-2, rtol=1e-1, device=DEVICE):
    torch.manual_seed(6)
    assert type(size) == tuple and len(size) == 2
    a = torch.randn(size, device=DEVICE, dtype=torch.float16)
    b = torch.randn(size, device=DEVICE, dtype=torch.float16)

    c_triton = matmul(a, b)
    c_torch = torch.matmul(a, b)

    torch.testing.assert_close(c_triton, c_torch, atol=atol, rtol=rtol)
    print("PASSED")

configs = [
    triton.testing.Benchmark(
        x_names = ['M', 'N', 'K'],
        x_vals = [128 * i for i in range(2, 33)],
        line_arg = "provider",
        line_vals=["torch", "triton"],
        line_names=["Pytorch", "Triton"],
        styles=[('green', '-'), ('blue', '-')],
        ylabel = "GB/s",
        plot_name = "matmul-performance",
        args={}
    )
]

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.05, 0.95]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 3 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == "__main__":
    test_matmul_kernel(size=(512, 512))
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)
    

