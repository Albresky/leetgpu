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
| **(__K0) Naive** | 0.4487 | 14.0210 | 4785.8499 |
| **(__K1) Using SMEM** | 0.3421 | 18.3898 | 6277.0476 |
| **(__K2) Using GemmPerThread + SMEM** | 0.0906 | 69.4225 | 23696.218 |


