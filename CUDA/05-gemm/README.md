# General Matrix-Matrix Multiplication (GEMM)

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
| **(__K0) Naive** | 0.4395 | 14.3154 | 4886.3127 |


