# Reduction

### 编译

`-O0` 禁用编译器优化。

Matrix sizes
 - M = 2048
 - N = 2048
 - K = 2048

### 性能数据对比

- RTX 4090 (sm89, Ada Lovelace)

| Kernel Version | Avg Latency (ms) | Memory Bandwidth (GB/s) | Throughput (GFLOPS) |
| :--- | :--- | :--- | :--- |
| **(__K0) Naive** | 0.0398 | 0.4114 | 0.0514 |
| **(__K1) Using Smem + Parallel reduce** | 0.0154(-61.31%) | 1.0631(+158.41%) | 0.1329(+158.56%) |
| **(__K2) Using Smem + shuffle** | 0.0124(-68.84%) | 1.3170(+220.13%) | 0.1646(+220.23%) |

### 分析

LeepGPU OJ 的 testcases 精度累计会丢失，[Kahan Algorithm](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) 也没用，提升精度至 double 可以，但是性能下降显著。
