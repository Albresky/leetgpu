# Softmax Attention (Scaled Dot-Product Attention)

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



## 性能数据对比

- RTX 4090 (sm89, Ada Lovelace)

| Kernel Version | Avg Latency (ms) | Memory Bandwidth (GB/s) | Throughput (GFLOPS) |
| :--- | :--- | :--- | :--- |
| **__K0** | 1.9342 | 17.3604 | 2223.2113 |


## Naive

```bash
  void mm_kernel<16>(const float *, const float *, float *, int, int, int, float) (64, 64, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector   16,777,216
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector      131,072
    smsp__inst_executed.sum                                         inst  129,400,832
    smsp__sass_inst_executed_op_shared_ld.sum                       inst   41,943,040
    smsp__sass_inst_executed_op_shared_st.sum                       inst    4,194,304
    -------------------------------------------------------- ----------- ------------
    
  void mm_transposed_kernel<16>(const float *, const float *, float *, int, int, int, float) (64, 64, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector   16,777,216
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector      131,072
    smsp__inst_executed.sum                                         inst  121,110,528
    smsp__sass_inst_executed_op_shared_ld.sum                       inst   41,943,040
    smsp__sass_inst_executed_op_shared_st.sum                       inst    4,194,304
    -------------------------------------------------------- ----------- ------------

  bmax_kernel(const float *, float *, int) (2, 1024, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector      131,072
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector        2,048
    smsp__inst_executed.sum                                         inst    3,682,304
    smsp__sass_inst_executed_op_shared_ld.sum                       inst       83,968
    smsp__sass_inst_executed_op_shared_st.sum                       inst       73,728
    -------------------------------------------------------- ----------- ------------

  max_kernel(const float *, float *, int) (1024, 1, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector        1,024
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector        1,024
    smsp__inst_executed.sum                                         inst    1,802,240
    smsp__sass_inst_executed_op_shared_ld.sum                       inst       41,984
    smsp__sass_inst_executed_op_shared_st.sum                       inst       36,864
    -------------------------------------------------------- ----------- ------------

  bsum_kernel(const float *, float *, const float *, int) (2, 1024, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector      163,840
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector        2,048
    smsp__inst_executed.sum                                         inst    4,108,288
    smsp__sass_inst_executed_op_shared_ld.sum                       inst       83,968
    smsp__sass_inst_executed_op_shared_st.sum                       inst       73,728
    -------------------------------------------------------- ----------- ------------

  sum_kernel(const float *, float *, int) (1024, 1, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector        1,024
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector        1,024
    smsp__inst_executed.sum                                         inst    1,802,240
    smsp__sass_inst_executed_op_shared_ld.sum                       inst       41,984
    smsp__sass_inst_executed_op_shared_st.sum                       inst       36,864
    -------------------------------------------------------- ----------- ------------

  norm_kernel(const float *, float *, const float *, const float *, int) (2, 1024, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector      196,608
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector      131,072
    smsp__inst_executed.sum                                         inst    1,343,488
    smsp__sass_inst_executed_op_shared_ld.sum                       inst            0
    smsp__sass_inst_executed_op_shared_st.sum                       inst            0
    -------------------------------------------------------- ----------- ------------
```
