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
    

def test_matmul_kernel(size: tuple, atol=1e-2, rtol=1e-1, device=DEVICE):
    torch.manual_seed(6)
    assert type(size) == tuple and len(size) == 2
    a = torch.randn(size, device=DEVICE, dtype=torch.float16)
    b = torch.randn(size, device=DEVICE, dtype=torch.float16)

    c_triton = matmul(a, b)
    c_torch = torch.matmul(a, b)

    torch.testing.assert_close(c_triton, c_torch, atol=atol, rtol=rtol)
    print("PASSED")


