# Matrix Multiplication

### 编译

`-O0` 禁用编译器优化。

Matrix sizes
 - M = 1024
 - N = 1024
 - K = 1024

### 性能数据对比

| Kernel Version | Avg Latency (ms) | Memory Bandwidth (GB/s) | Compute (GFLOPS) |
| :--- | :--- | :--- | :--- |
| **Naive (__K0)** | 0.4397 | 28.4940 |  4862.9793 |
| **Smem (__K1)** | 0.3431 | 36.6695 | 6258.2574 |

