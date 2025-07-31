"""
- call sub-kernels
- using tl.exp2()
- specific flash-attention
- tl.static_assert()
- multi-axis launch grid
- pre-compute in separate kernel
- using approximate constants

DOES NOT INCLUDE
- only fp32
- dropout
- non-causal
- i forgor 💀
"""

import torch
import triton
import triton.language as tl
import math
DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")

@triton.jit
def _attn_forward_inner(
    Q, O, L, M,
    K_ptr, V_ptr,
    K_T_offsets, V_offsets,
    block_index_QO,
    scale,
    stride_K_N, stride_V_N,
    N,
    BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
    DIAGONAL: tl.constexpr,
    offsets_QO_N: tl.constexpr, offsets_KV_N: tl.constexpr,
    Dh: tl.constexpr
):
    if DIAGONAL:
        lo = block_index_QO * BLOCK_SIZE_QO
        hi = (block_index_QO + 1) * BLOCK_SIZE_QO
    else:
        lo, hi = 0, block_index_QO * BLOCK_SIZE_QO

    K_T_offsets += lo * stride_K_N
    V_offsets += lo * stride_V_N
    offsets_KV_N += lo

    for start_KV in range(lo, hi, BLOCK_SIZE_KV):
        start_KV = tl.multiple_of(start_KV, BLOCK_SIZE_KV)
        mask_KV_N = offsets_KV_N < N
        K_T = tl.load(K_ptr + K_T_offsets, mask=mask_KV_N[None, :], other=0.0) # shape (Dh, BLOCK_SIZE_KV)
        S = tl.dot(Q, K_T) * scale # shape (BLOCK_SIZE_QO, BLOCK_SIZE_KV)

        if DIAGONAL:
            causal_mask = offsets_QO_N[:, None] >= offsets_KV_N[None, :]
            S = tl.where(causal_mask, S, -1e6) # shape (BLOCK_SIZE_QO, BLOCK_SIZE_KV)

        M_new = tl.maximum(M, tl.max(S, axis=1)) # shape (BLOCK_SIZE_QO,)
        S -= M_new[:, None]

        P = tl.exp2(S) # shape (BLOCK_SIZE_QO, BLOCK_SIZE_KV)
        L_new = tl.sum(P, axis=1) # shape (BLOCK_SIZE_QO,)
        alpha = tl.exp2(M - M_new)
        L = L * alpha + L_new

        V = tl.load(V_ptr + V_offsets, mask=mask_KV_N[:, None], other=0.0) 
        O = O * alpha[:, None] # shape (BLOCK_SIZE_QO, Dh)
        O = tl.dot(P, V, acc=O)

        M = M_new
        K_T_offsets += BLOCK_SIZE_KV * stride_K_N
        V_offsets += BLOCK_SIZE_KV * stride_V_N
        offsets_KV_N += BLOCK_SIZE_KV
    
    return O, L, M

@triton.autotune(
        [
            triton.Config(
                {"BLOCK_SIZE_QO": BLOCK_SIZE_QO, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
                num_stages=num_stages, num_warps=num_warps
            )
            for BLOCK_SIZE_QO in [16]#, 32, 64, 128]
            for BLOCK_SIZE_KV in [16]#, 32, 64, 128]
            for num_stages in [3]#, 5, 7]
            for num_warps in [4]#, 8, 16]
        ],
        key=["Dh"]
)

@triton.jit
def attn_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    LSE_ptr,
    scale,
    stride_Q_B, stride_Q_H, stride_Q_N, stride_Q_Dh,
    stride_K_B, stride_K_H, stride_K_N, stride_K_Dh,
    stride_V_B, stride_V_H, stride_V_N, stride_V_Dh,
    stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
    stride_LSE_B, stride_LSE_H, stride_LSE_N,
    B, H, N,
    Dh: tl.constexpr,
    BLOCK_SIZE_QO: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr
):
    rln2: tl.constexpr = 1.4426950408889634 # 1 / math.log(2)
    # e^x = 2^(x * rln2)
    scale *= rln2

    tl.static_assert(BLOCK_SIZE_KV <= Dh)

    index_BH = tl.program_id(axis=1)
    index_B = index_BH // H
    index_H = index_BH % H
    Q_ptr += index_B * stride_Q_B + index_H * stride_Q_H
    K_ptr += index_B * stride_K_B + index_H * stride_K_H
    V_ptr += index_B * stride_V_B + index_H * stride_V_H
    O_ptr += index_B * stride_O_B + index_H * stride_O_H
    
    block_index_QO = tl.program_id(axis=0)
    offsets_QO_N = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO)
    offsets_KV_N = tl.arange(0, BLOCK_SIZE_KV)
    offsets_Dh = tl.arange(0, Dh)

    Q_offsets = offsets_QO_N[:, None] * stride_Q_N + offsets_Dh[None, :] * stride_Q_Dh # shape (BLOCK_SIZE_QO, Dh)
    K_T_offsets = offsets_Dh[:, None] * stride_K_Dh + offsets_KV_N[None, :] * stride_K_N # shape (Dh, BLOCK_SIZE_KV)
    V_offsets = offsets_KV_N[:, None] * stride_V_N + offsets_Dh[None, :] * stride_V_Dh # shape (BLOCK_SIZE_KV, Dh)

    mask_QO_N = offsets_QO_N < N # if Dh was not multiple of warp size we would also mask
    Q = tl.load(Q_ptr + Q_offsets, mask=mask_QO_N[:, None], other=0.0) # shape (BLOCK_SIZE_QO, Dh)

    M = tl.full(shape=[BLOCK_SIZE_QO], value=-1e6, dtype=tl.float32)
    L = tl.full(shape=[BLOCK_SIZE_QO], value=1.0, dtype=tl.float32) # e^0 = 1
    O = tl.zeros((BLOCK_SIZE_QO, Dh), dtype=tl.float32)

    O, L, M = _attn_forward_inner(
        Q, O, L, M,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        scale,
        stride_K_N, stride_V_N,
        N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        False, # diagonal
        offsets_QO_N, offsets_KV_N,
        Dh
    )

    O, L, M = _attn_forward_inner(
        Q, O, L, M,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        scale,
        stride_K_N, stride_V_N,
        N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        True, # diagonal
        offsets_QO_N, offsets_KV_N,
        Dh
    )

    O = O / L[:, None]
    LSE = M + tl.math.log2(L) # shape (BLOCK_SIZE_QO,)
    """
    softmax(x_i) = exp(x_i - m_i) / l_i
                 = exp(x_i - m_i) / exp(log(l_i))
                 = exp(x_i - (m_i + log(l_i)))
    """
    LSE_offsets = index_BH * stride_LSE_H + offsets_QO_N
    LSE_mask = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO) < N
    tl.store(LSE_ptr + LSE_offsets, LSE, mask=LSE_mask)

    O_offsets = offsets_QO_N[:, None] * stride_O_N + offsets_Dh[None, :] * stride_O_Dh
    tl.store(O_ptr + O_offsets, O, mask=mask_QO_N[:, None])






class _flashattention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):
        assert q.shape == k.shape == v.shape, "q, k, v must have the same shape"
        assert q.shape[-1] <= 128, "kernel only supports head dimension up to 128" # I'll do math on 1660 Super SRAM later
        assert q.device == k.device == v.device, "q, k, v must be on the same device"
        assert q.dtype == k.dtype == v.dtype == torch.float32, "q, k, v must be float32"
        B, H, N, Dh = q.shape

        O = torch.empty_like(q)
        LSE = torch.empty((B, H, N), device=q.device, dtype=torch.float32)

        grid = lambda args: (
            triton.cdiv(N, args["BLOCK_SIZE_QO"]),
            B * H
        )

        attn_fwd[grid](
            q, k, v, O, LSE, scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), q.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            B, H, N, Dh,

        )

        ctx.save_for_backward(q, k, v, O, LSE)
        ctx.grid = grid
        ctx.B, ctx.H, ctx.N, ctx.Dh = B, H, N, Dh
        ctx.scale = scale
        return O

flash_attention = _flashattention.apply

def test_flash_attention_kernel(B, H, N, Dh, device=DEVICE, atol=5e-3):
    q = torch.randn((B, H, N, Dh), device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn((B, H, N, Dh), device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn((B, H, N, Dh), device=device, dtype=torch.float32, requires_grad=True)
    scale = 1 / math.sqrt(Dh)
    attention_triton = flash_attention(q, k, v, scale)
    attention_torch = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(attention_triton, attention_torch, atol=atol, rtol=0)
    print(f"Forward test passed for B={B}, H={H}, N={N}, Dh={Dh}")

configs = [
    triton.testing.Benchmark(
        x_names=["SEQ_LEN"],
        x_vals = [512 * i for i in range(1, 17)],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["torch.nn.functional.scaled_dot_product_attention", "our flash_attention"],
        styles=[("red", '-'), ("blue", '-')],
        ylabel="TFLOPS",
        plot_name=f"attention-performance-{mode}",
        args={"mode": mode}
    )
    for mode in ["fwd"]
]

@triton.testing.perf_report(configs)
def benchmark_flash_attention(SEQ_LEN, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float32
    BATCH, N_HEADS = 32, 4
    HEAD_DIM = 128
    q = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), device=device, dtype=dtype, requires_grad=True)
    k = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), device=device, dtype=dtype, requires_grad=True)
    v = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), device=device, dtype=dtype, requires_grad=True)
    scale = 1 / math.sqrt(HEAD_DIM)
    if provider == "torch":
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    elif provider == "triton":
        fn = lambda: flash_attention(q, k, v, scale)
    if mode == "bwd":
        O = fn()
        dLdO = torch.randn_like(O)
        fn = lambda: O.backward(dLdO, retain_graph=True)
    ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * N_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM
    total_flops = flops_per_matmul
    if mode == "bwd":
        total_flops *= 2.5
    return total_flops * 1e-12 / (ms * 1e-3)  # TFLOPS

if __name__ == "__main__":
    test_flash_attention_kernel(1, 1, 128, 32)
    test_flash_attention_kernel(1, 1, 128, 64)
    test_flash_attention_kernel(1, 1, 128, 128)
    test_flash_attention_kernel(32, 8, 69, 128)

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark_flash_attention.run(save_path='.', print_data=True)
