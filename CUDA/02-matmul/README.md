# Matrix Multiplication

### 编译

`-O0` 禁用编译器优化。

Matrix sizes
 - M = 1024
 - N = 1024
 - K = 1024

### 性能数据对比

- RTX 4090 (Ada Lovelace)

| Kernel Version | Avg Latency (ms) | Memory Bandwidth (GB/s) | Compute (GFLOPS) |
| :--- | :--- | :--- | :--- |
| **(__K0) Naive** | 0.4397 | 28.4940 |  4862.9793 |
| **(__K1) Smem** | 0.3431 | 36.6695 | 6258.2574 |
| **(__K2) GemmPerThread + Smem** | 0.0956 | 131.5632 | 22453.4480 |

### 分析

- 1. 关于线程块的二维形状
目前是 blockDim.x == blockDim.y 的方形设计。由于我们希望能够对原始矩阵和转置矩阵的列访存都实现最大化 memory coalescing，那么一旦他们不相等，将会顾此失彼。

