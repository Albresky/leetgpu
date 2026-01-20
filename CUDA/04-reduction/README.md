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
| **(__K3) Using Smem + Parallel reduce + Once atomicAdd** | 0.0225(-43.47%) | 0.7274(+76.81%) | 0.0909(+76.85%) |


### 分析

LeepGPU OJ 的 testcases 精度累计会丢失，[Kahan Algorithm](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) 也没用，提升精度至 double 可以，但是性能下降显著。

### 回答几个问题

**1. 为什么 `__shfl_down_sync` 更快？**
 
> 使用 SMEM 实现规约需要 a) 读写 SMEM；b)潜在 bank conflicts; c) 需要控制线程间同步。而 **shuffle** 直接利用寄存器跨线程传递数据，指令更短更快。

**2. shuffle 中， 掩码 Mask `0xfffffff` 和 `__activemask()` 区别**

> 0xfffffff 假设 warp 内的 32 个线程全部参与；而 **__activemask()** 适合控制实际参与计算的线程，包括：a) 含**分支发散**（warp内有线程不执行suffle）；b) 最后一个warp不满，比如 gid>N 时被裁剪的线程用 full mask 会读到垃圾值。

**3. Grid 内的线程偏移**

```C++
int tid = threadIdx.x;
int gid = blockIdx.x * blockDim.x + tid;
int stride = blockDim.x * gridDim.x;

float sum = 0.0f;
for (int i = gid; i < N; i += stride) {
  sum += input[i];
}
```

**stride** 记录grid内线程总数，i从当前全局threadid开始索引，偏移线程总数，一直覆盖到全部的N，可以巧妙实现所有线程一直从不同起点开始。举个例子，假如 *N = 60*, *threadsPerBlock=8*, *blocksPerGrid=2*，那么每个线程是如下索引的：

- **tid-0**: 0, 16, 24, 32, ........., 56
- **tid-1**: 1, 17, 25, 33, ........., 57
- **tid-2**: 2, 18, 26, 34, ........., 58
- ...
- **tid-4**: 4, 20, 28, 26, ........., 60
- ...
- **tid-7**: 7, 23, 31, 39, ..., 55