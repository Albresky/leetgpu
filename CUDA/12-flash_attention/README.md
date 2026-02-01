# FlashAttention

参考资料：

- [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)

- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/pdf/2307.08691)


## Problem

**Title:** FlashAttention (IO-Aware Exact Attention)

**Difficulty:** Hard

**Description:**
Implement the **FlashAttention V1** algorithm as described in the paper "*[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)*".

Given a query matrix , key matrix , and value matrix , compute the attention output :

Unlike standard attention which requires  memory to store the intermediate attention score matrix () and probability matrix (), FlashAttention computes the exact result in  memory using **Tiling** and **Recomputation**, significantly reducing High Bandwidth Memory (HBM) accesses.

**Specifics:**

1. **Tiling:** Load inputs into SRAM in blocks to compute partial results.
2. **Online Softmax:** Update Softmax statistics (max  and sum ) on-the-fly without accessing the full row.
3. **Kernel Fusion:** Fuse all operations (, Softmax, ) into a single CUDA kernel.

**Constraints:**

* Matrix dimensions:  (Sequence Length) up to 32k+,  (Head Dim) up to 128.
* **No intermediate  matrix writes to global memory.**
* Use standard `float` (FP32) for accumulation.
* Assume row-major storage.

**Function Signature:**

```cpp
void solve(float* Q, float* K, float* V, float* output, int M, int N, int d);
```

## 简介

FlashAttention 的核心创新不在于改变了 Attention 的 max 和 sum 的 **计算顺序** 和 **内存访问模式**。

传统的 Attention 算法是 **Layer-by-Layer** 的（先算完所有的 $S=QK^T$ ，存入 HBM(GMEM)；再读出来算 Softmax，存入 HBM...）。
FlashAttention 是 **Block-by-Block** 的，利用 [Online Softmax](../07-softmax/README.md) 技巧，在一次循环中流式更新 Output，避免了 $O(n^2)$ 的 HBM 读写。

---

### 数学原理：Online Softmax 的递推推导

为了在不存储完整 $N$ 列数据的情况下计算 Softmax，我们需维护运行时的统计量。

#### 1. 符号定义

假设我们正在计算某一行 $Q_i$ 的 Attention。
我们将 $K,V$ 分割为多个块 $B_1, B_2, \ldots, B_t$ 。当前处理到第 $B_j$ 个块。

- $S_{ij}=Q_iK_j^T$ : 当前块的 Score
- $m_{j-1}$: 处理完前 $(j-1)$ 个块后的局部最大值
- $d_{j-1}$: 处理完前 $(j-1)$ 个块后的局部指数和（分母）
- $O_{j-1}$ : 处理完前 $(j-1)$ 个块后的**非归一化** Output
- $m_{j}^*$ : 当前第 $j$ 个块内部的最大值

#### 2. Softmax 的数值稳定性 (Safe-Softmax)

这里不再展开介绍，直接给公式：

$$
softmax(x_k) = \frac{e^{x_k - m}}{\sum e^{x_i - m}}\text{, where } m = \max(x_i)
$$



#### 3. 递推公式

当我们处理新的块 $B_j$ 时，我们需要将“旧的统计量”和“当前块的统计量”合并。由于 $m_{j-1}$ 和  $m_j$ 可能不同，我们需要通过 **重缩放 (Rescaling)** 来统一指数的基底。

**Step 1: 更新全局最大值**

$$
m_{j}= \max(m_{j-1}, m_j)
$$

**Step 2: 计算重缩放因子**

**旧数据的衰减系数（如果 $m_j > m_{j-1}$ ，旧数据变小）：**

$$
\alpha = e^{{m_{j-1}} - {m_{j}}}
$$

新数据的缩放系数（将当前块平移到全局最大值基准）：

$$
\beta = e^{m_{j}^* - m_{j}}
$$

**Step 3: 更新分母 (Denominator)**

$$
d_j = \alpha \cdot d_{j-1} + \sum_{k \in B_j} e^{S_{ik} - m_{j}} = \alpha \cdot d_{j-1} + \beta \cdot \sum_{k \in B_j} e^{S_{ik} - m_{j}^*} 
$$

**Step 4: 更新分子 (Numerator / Output)**

$$
O_j = \alpha \cdot O_{j-1} + \sum_{k \in B_j} e^{S_{ik} - m_{j}} V_k = \alpha \cdot O_{j-1} + \beta \cdot \sum_{k \in B_j} e^{S_{ik} - m_{j}^*} V_k
$$

#### 4. 最终结果

遍历完所有块后，最终的 Output 为：

$$
O_{final} = \frac{O_{t}}{d_{t}}
$$

## CUDA 实现思路

### 1. 宏观映射：Grid 与 Block

FlashAttention 采用 **Query-Stationary**（Query 驻留）策略。

- **Grid (网格)**:
我们把 Output 矩阵（以及 Q 矩阵）按行切分为大小为 $B_r$ 的块
    - `blockIdx.x` 对应 $Q$ 的行块索引
    - **物理含义**：每个 Thread Block 负责计算 $Q$ 的 $B_r$ 行对应的完整 Attention 结果


- **Thread Block (线程块)**
    - `blockDim.y` = $B_r$：每个线程 Warp（或一组线程）负责 $Q$ 的一行（或多行）
    - `blockDim.x` = 32 (Warp Size) 或其他：用于在 $d$ 特征维度上进行并行计算（如加载、P*V 乘法）
    - **寄存器分配**：每个线程私有维护 `mi` 最大值, `di` 分母, `acc_o` 累积输出向量（的部分元素）



### 2. 内存层级设计

- **SRAM (Shared Memory)**:
    - `s_Q [Br][d]`: **驻留**。Kernel 开始时加载一次，直到结束
    - `s_K [Bc][d]`, `s_V [Bc][d]`: **流动**。在循环中不断被新的块覆写


- **Registers**:
    - `acc_o [d]`: 存储当前线程负责的那一行在对应列上的计算结果。这是为了避免反复读写 SRAM，直接在最快的寄存器中完成累加。



### 3. 核心循环与加载逻辑 (Coalesced Access)

我们使用 **协作加载 (Collaborative Loading)** 以最大化 GMEM 带宽。

如：

```cpp
// 目标：将 GMEM 中的 K 块 (TileK x d) 加载到 SRAM s_K
// 线程配置：blockDim.x (对应列 d), blockDim.y (对应行 Br)
// 数据形状：TileK (行) x d (列)

for (int r = tidy; r < TileK; r += blockDim.y) { // 行方向循环
  int gr = t * TileK + r; // 计算 Global Row Index
  for (int c = tidx; c < d; c += blockDim.x) {   // 列方向循环
     // ... 加载逻辑 ...
     s_K[r][c] = K[gr * d + c];
  }
}

```

**Mapping 详解：**

1. **线程不足问题**：
通常 $B_c \times d$ (例如 $16 \times 256$) 的数据量远大于线程数 ($16 \times 32$)。
因此，每个线程需要搬运多个数据，每LOAD一次，需向后跳跃 `blockDim` 的步长，继续搬下一个。
2. **`tidy` 与 `r` 的区别**：
    - `tidy`: 线程在 Block 内的 Y 轴 ID
    - `r`: 数据在 Tile 内的行坐标
    - **映射关系**：`r` 从 `tidy` 开始，每次增加 `blockDim.y`。这确保了所有线程像扫过整个数据块


3. **合并访存 (Coalescing)**：
    - 内层循环使用 `tidx` 索引 `d` 维度（列）
    - 由于矩阵是行主序存储，相邻的 `tidx` 访问的是相邻的内存地址 `K[... + c]` 和 `K[... + c + 1]`
    - 这完美符合 GPU 内存控制器 **128-byte transaction** 的合并访问要求



### 4. Device 内核

在 `s_Q` 驻留，`s_K, s_V` 加载完毕后，执行以下步骤：

1. **Score 计算**: $S_{ij}=s_Q \times s_K^T$。
    - 这里实际上是在 SRAM 上做小规模 GEMM


2. **Online Softmax 更新**:
    - 计算局部 max 和 sum
    - 利用 $\alpha, \beta   $ 系数更新寄存器中的 `mi, di`


3. **Output 累加**:
    - $O_j = \alpha \cdot O_{j-1} + \beta \cdot \sum_{k \in B_j} e^{S_{ik}} V_k$
    - 这里 $\sum_{k \in B_j} e^{S_{ik}} V_k$ 是当前 Tile 的贡献


4. **最终归一化**:
* 循环结束后，执行 `acc_o / di` 写入 GMEM。


### 性能分析

## 性能数据对比

- RTX 4090 (sm89, Ada Lovelace)

| Kernel Version | Avg Latency (ms) | Memory Bandwidth (GB/s) | Throughput (GFLOPS) |
| :--- | :--- | :--- | :--- |
| **__K0** | 0.7420 | 2.8264 | 723.5504 |


## Naive

```bash
void fa1_kernel<16, 16, 128>(const float *, const float *, const float *, float *, int, int, float) (1, 64, 1)x(32, 16, 1), Context 1, Stream 7, Device 0, CC 8.9
  Section: Command line profiler metrics
  -------------------------------------------------------- ----------- ------------
  Metric Name                                              Metric Unit Metric Value
  -------------------------------------------------------- ----------- ------------
  dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector    2,113,536
  l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector       16,384
  smsp__inst_executed.sum                                         inst  203,816,960
  smsp__sass_inst_executed_op_shared_ld.sum                       inst   39,845,888
  smsp__sass_inst_executed_op_shared_st.sum                       inst      528,384
  -------------------------------------------------------- ----------- ------------
```