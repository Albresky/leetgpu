# Scaled Dot-Product Attention

## Problem

**Title:** Scaled Dot-Product Attention (Naive)

**Difficulty:** Medium

**Description:**
Implement the Scaled Dot-Product Attention function as described in the paper "*Attention Is All You Need*".

Given a query matrix $Q$ of size $M \times d_k$, a key matrix $K$ of size $N \times d_k$, and a value matrix $V$ of size $N \times d_v$, compute the output matrix $O$ of size $M \times d_v$.

The mathematical formula is:
$$
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


**Specifics:**

1. **Scaling:** Before softmax, scale the dot product by $\frac{1}{\sqrt{d_k}}$.
2. **Softmax:** Applied row-wise (along the dimension of size $N$).
3. **Output:** Store the result in `output`.

**Constraints:**

* Matrix dimensions:
  - $Q$: $M \times d_k$
  - $K$: $N \times d_k$
  - $V$: $N \times d_v$ (for simplicity, assume $d_k = d_v = d$)
  - $O$: $M \times d$
* $1 \leq M, N \leq 32768$ (Practical size for naive without blocking)
* $1 \leq d \leq 1024$ (Dimension size)
* External libraries (cuBLAS, cuDNN) are **NOT** permitted.
* Assume row-major storage for all matrices.

**Function Signature:**

```cpp
void solve(float* Q, float* K, float* V, float* output, int M, int N, int d);
```

## 思路

本质就是执行 $M$ 次独立的 1D Softmax($S_i$) 操作，求得 $S_{M \times N}= Q \times K^T$ 每一行的概率分布；再乘以 $V$， 对 $d_v$ 维度的每个向量进行加权求和。
