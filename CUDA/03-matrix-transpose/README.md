# Matrix Transpose

### 编译

`-O0` 禁用编译器优化。

Matrix sizes
 - M = 16384
 - N = 16384
 - K = 16384

### 性能数据对比

- RTX 4090 (sm89, Ada Lovelace)

| Kernel Version | Avg Latency (ms) | Memory Bandwidth (GB/s) | Compute (GFLOPS) |
| :--- | :--- | :--- | :--- |
| **Naive (__K0)** | 3.4611 | 620.4692 | 0 |
