import torch
import triton
import triton.language as tl
import math

############################################################
########## Naive Implementation of Self-Attention ##########
############################################################


# kernel 1
@triton.jit
def matmal_qk_kernel(
    Q_ptr,
    K_ptr,
    S_ptr,
    M,
    N,
    K,
    stride_qm,
    stride_qk,
    stride_kn,
    stride_kk,
    stride_om,
    stride_on,
    scale,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Naive implementation of matmul_qk_kernel.

    Computes S = Q(M, K) * K(N, K)^T

    :param Q_ptr: Address of Tensor Q
    :param K_ptr: Address of Tensor K
    :param S_ptr: Address of Tensor S
    :param M: First dimension of Q
    :param N: First dimension of K
    :param K: Second dimension of Q and K
    :param stride_qm: Stride of first dimension (M) of Q
    :param stride_qk: Stride of second dimension (K) of Q
    :param stride_kn: Stride of first dimension (N) of K
    :param stride_kk: Stride of second dimension (K) of K
    :param stride_om: Stride of first dimension (M) of output
    :param stride_on: Stride of second dimension (N) of output
    :param scale: Scale factor for S. Computes P = softmax(Q*K^T/sqrt(d))
    :param BLOCK_SIZE_M: Block size in dimension M
    :param BLOCK_SIZE_N: Block size in dimension N
    :param BLOCK_SIZE_K: Block size in dimension K
    """
    # 1. block id
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 2. create offsets
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Q ptrs
    offs_qm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    q_ptrs = Q_ptr + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)

    # K ptrs
    offs_kn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    k_ptrs = K_ptr + (offs_kn[None, :] * stride_kn + offs_k[:, None] * stride_kk)

    # 3. accumulate
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # LOAD blocks
        q_val = tl.load(q_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        k_val = tl.load(k_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        acc += tl.dot(q_val, k_val)

        q_ptrs += BLOCK_SIZE_K * stride_qk
        k_ptrs += BLOCK_SIZE_K * stride_kk

    # 4. write out
    c = acc * scale
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = S_ptr + (offs_cm[:, None] * stride_om + offs_cn[None, :] * stride_on)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# kernel 2
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr, M, N, stride_m, stride_n, BLOCK_SIZE: tl.constexpr
):
    """
    Softmax Kernel

    :param input_ptr: input matrix
    :param output_ptr: output matrix
    :param M: first dim of matrix
    :param N: second dim of matrix
    :param stride_m: stride of first dim
    :param stride_n: stride of second dim
    :param BLOCK_SIZE: BlockSIZE for N
    :type BLOCK_SIZE: tl.constexpr
    """
    row_idx = tl.program_id(0)

    # 1. ptrs
    row_start_ptr = input_ptr + row_idx * stride_m
    out_row_start_ptr = output_ptr + row_idx * stride_m
    col_offs = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offs * stride_n

    # 2. LOAD
    row = tl.load(input_ptrs, mask=col_offs < N, other=-float("inf"))

    # compute softmax
    # 3. findmax
    row_minus_max = row - tl.max(row, axis=0)
    # 4. exp(x-m)
    numerator = tl.exp(row_minus_max)
    # 5. sum
    denominator = tl.sum(numerator, axis=0)
    # 6. div
    sm_output = numerator / denominator

    # 7. STORE write out
    tl.store(out_row_start_ptr + col_offs * stride_n, sm_output, mask=col_offs < N)


# kernel 3
@triton.jit
def matmal_pv_kernel(
    P_ptr,
    V_ptr,
    Out_ptr,
    M,
    N,
    K,
    stride_pm,
    stride_pk,
    stride_vn,
    stride_vk,
    stride_om,
    stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Naive implementation of matmul_pv_kernel.

    Computes Out = P(M, N) * V(N, d), where M=M, K=N, N=d


    :param P_ptr: Address of Tensor P
    :param V_ptr: Address of Tensor V
    :param Out_ptr: Address of Output Tensor
    :param M: First dimension of P
    :param N: Second dimension of V
    :param K: Shared dimension of P and V
    :param stride_pm: Stride of first dimension (M) of P
    :param stride_pk: Stride of second dimension (K) of P
    :param stride_vn: Stride of second dimension (N) of V
    :param stride_vk: Stride of first dimension (K) of V
    :param stride_om: Stride of first dimension (M) of output
    :param stride_on: Stride of second dimension (N) of output
    :param BLOCK_SIZE_M: Block size in dimension M
    :param BLOCK_SIZE_N: Block size in dimension N
    :param BLOCK_SIZE_K: Block size in dimension K
    """
    # 1. block id
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 2. create offsets
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # P ptrs
    offs_pm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    p_ptrs = P_ptr + (offs_pm[:, None] * stride_pm + offs_k[None, :] * stride_pk)

    # V ptrs
    offs_vn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    v_ptrs = V_ptr + (offs_k[:, None] * stride_vk + offs_vn[None, :] * stride_vn)

    # 3. accumulate
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # LOAD blocks
        p_val = tl.load(p_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        v_val = tl.load(v_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        acc += tl.dot(p_val, v_val)

        p_ptrs += BLOCK_SIZE_K * stride_pk
        v_ptrs += BLOCK_SIZE_K * stride_vk

    # 4. write out
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = Out_ptr + (offs_cm[:, None] * stride_om + offs_cn[None, :] * stride_on)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def softmax_attention(Q, K, V):
    """
    Docstring for softmax_attention

    :param Q: Description
    :param K: Description
    :param V: Description
    """
    M, d = Q.shape
    N, _ = K.shape

    # 1. alloc mem
    S = torch.empty((M, N), device=Q.device, dtype=torch.float32)
    P = torch.empty((M, N), device=Q.device, dtype=torch.float32)
    Output = torch.empty((M, d), device=Q.device, dtype=torch.float32)

    # 2. run S=Q*K^T/sqrt(dk)
    BLOCK_M = tl.constexpr(16)
    BLOCK_N = tl.constexpr(16)
    BLOCK_K = tl.constexpr(32)  # d
    grid_1 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmal_qk_kernel[grid_1](
        Q,
        K,
        S,
        M,
        N,
        d,
        Q.stride(0),
        Q.stride(1),
        K.stride(0),
        K.stride(1),
        S.stride(0),
        S.stride(1),
        1.0 / math.sqrt(d),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
    )

    # 3. P=softmax(S)
    BLOCK_SIZE_SM = triton.next_power_of_2(N)
    softmax_kernel[(M,)](S, P, M, N, S.stride(0), S.stride(1), BLOCK_SIZE=BLOCK_SIZE_SM)

    # 4. Out(M,d)=P(M,N)*V(N,d)
    grid_2 = (triton.cdiv(M, BLOCK_M), triton.cdiv(d, BLOCK_N))
    matmal_pv_kernel[grid_2](
        P,
        V,
        Output,
        M,
        d,
        N,
        P.stride(0),
        P.stride(1),
        V.stride(1),
        V.stride(0),
        Output.stride(0),
        Output.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
    )

    return Output


if __name__ == "__main__":
    torch.manual_seed(10913)

    M, N, d = 1024, 1024, 256
    Q = torch.randn(M, d, device="cuda", dtype=torch.float32)
    K = torch.randn(N, d, device="cuda", dtype=torch.float32)
    V = torch.randn(N, d, device="cuda", dtype=torch.float32)

    # triton
    output = softmax_attention(Q, K, V)

    # torch golden
    t_S = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(d)
    t_P = torch.softmax(t_S, dim=-1)
    t_O = torch.matmul(t_P, V)

    # check
    if torch.allclose(output, t_O, atol=1e-3):
        print("Test PASS!")
    else:
        print("Test FAILED!")
        print("Max diff:", torch.max(torch.abs(output - t_O)))
