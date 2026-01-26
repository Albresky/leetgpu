
# A. 先确认“程序路径跑到哪一步”
**目标：** 确认程序真的进入 `init()` / `run()` / `verify()`。

1. 在 `init()` 开头打印一句：  
   - “init start”  
2. 在 `run()` 开头打印一句：  
   - “run start”  
3. 在 `verify()` 开头打印一句：  
   - “verify start”  
4. 运行 `make 11` 后，观察输出：  
   - 如果只看到 “init start”，说明 `run()` 没被执行  
   - 如果看不到 “verify start”，说明 `verify()` 没执行  
**下一步：**  
- 如果没有跑到 `run()`，就去看 problem 工厂、main 调用链是否正确。  
- 如果跑到 `run()` 但没 `verify()`，就看 `run()` 是否异常退出或提前返回。

---

# B. 必须加“CUDA 错误检查 + 强制同步”
**目标：** 捕获异步错误，让问题立即暴露。

1. 在每个 kernel launch 后立即插入：  
   - `cudaGetLastError()`  
   - `cudaDeviceSynchronize()`  
2. 在每个 `cudaMalloc/cudaMemcpy` 后检查返回值。  
3. 重新运行，观察是否报错。  
**下一步：**  
- 如果报错，记录是哪个 kernel 或哪次分配失败。  
- 如果不报错，继续下一步。

---

# C. 使用 compute-sanitizer 检查越界和非法访问
**目标：** 排查 silent crash 和无输出的常见原因：非法访问。

```bash
(base) ➜  11-attention git:(main) ✗ compute-sanitizer --tool memcheck ./11_runner
========= COMPUTE-SANITIZER
Running global main for /root/wkspace/leetgpu/CUDA/11-attention/./11_runner...
Init. Attention params: Q(8192, 2048), K(16384, 2048), V(16384, 2048)
Warming up...
Running benchmark (1 iterations)...
run.
========= Invalid __shared__ write of size 4 bytes
=========     at max_kernel(const float *, float *, int)+0x100
=========     by thread (272,0,0) in block (0,0,0)
=========     Access to 0x440 is out of bounds
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: solve [0xad93] in 11_runner
=========         Host Frame: AttentionProblem::run() [0xcaf5] in 11_runner
=========         Host Frame: main [0x9f9a] in 11_runner
========= 
========= Invalid __shared__ write of size 4 bytes
=========     at max_kernel(const float *, float *, int)+0x100
=========     by thread (273,0,0) in block (0,0,0)
=========     Access to 0x444 is out of bounds
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: solve [0xad93] in 11_runner
=========         Host Frame: AttentionProblem::run() [0xcaf5] in 11_runner
=========         Host Frame: main [0x9f9a] in 11_runner

...
...

========= Invalid __shared__ write of size 4 bytes
=========     at max_kernel(const float *, float *, int)+0x100
=========     by thread (371,0,0) in block (0,0,0)
=========     Access to 0x5cc is out of bounds
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: solve [0xad93] in 11_runner
=========         Host Frame: AttentionProblem::run() [0xcaf5] in 11_runner
=========         Host Frame: main [0x9f9a] in 11_runner
========= 
Error: main.cu:39, code:719, reason: unspecified launch failure
========= Target application returned an error
========= ERROR SUMMARY: 74476 errors
========= ERROR SUMMARY: 74376 errors were not printed. Use --print-limit option to adjust the number of printed errors
```

1. 用 memcheck 运行：
   - 观察是否出现 “Invalid global read/write”  
2. 如果出现，记下错误地址和 kernel 名称。  
3. 再用 racecheck 看是否有竞态。  
**下一步：**  
- 有非法访问 → 查对应 kernel 的索引/步长/边界判断  
- 没有非法访问 → 继续

---

# D. 用 Nsight Systems 确认 kernel 是否真的被 launch
**目标：** 区分“没 launch”和“launch 失败”的问题。

1. 使用 Nsight Systems 运行程序  
2. 打开 timeline  
3. 查看是否有 kernel 名称和执行时间  
**下一步：**  
- 如果没有 kernel 记录 → launch 没发生 / 早退出  
- 如果有但时间=0 或异常 → 说明 launch 失败或被中断

---

# E. 缩小规模做快速验证
**目标：** 快速复现并看清楚问题。

1. 把 `M,N,d` 改小到 8/16/32  
2. 只跑一个小 batch  
3. 重新运行  
**下一步：**  
- 小规模能跑 → 问题可能是资源限制或溢出  
- 小规模也不行 → 问题是逻辑或索引错误

---

# F. 分阶段验证 GPU 中间结果
**目标：** 精确定位是哪一步算错。

1. 只跑第一步：`S = Q * K^T`  
   - 拷回 S 和 CPU 结果对比  
2. 只跑 softmax：  
   - 对比 `bmax`, `sum`, `S_norm`  
3. 最后跑 `S_norm * V`  
**下一步：**  
- 哪一步开始不一致，就定位那一步的 kernel

---

# G. 核对索引和步长（重点检查）
**目标：** 排除最常见的 GPU bug：偏移/stride 错。

对以下情况逐条检查：  
- `S + m` 是否应该是 `S + m * N`  
- `S_norm + M` 是否应该是 `S_norm + m * N`  
- reduction kernel 中是否正确使用 `blockDim.x`  
**下一步：**  
- 如果发现 index 可疑，先记录并标记为“高优先级错误点”。

---

# H. 资源与配置检查
**目标：** 排查资源限制导致 launch 失败。

检查：  
- block size 是否超过设备限制  
- shared memory 是否超限  
- grid size 是否合理  
**下一步：**  
- 如果资源超限 → 调小 block 或 shared memory

---

# I. 对比 GPU/CPU 单元素调试
**目标：** 细粒度定位错误来源。

1. 选择一个 `(m,n)`  
2. 在 GPU 端 printf 输出该元素  
3. 在 CPU 端打印同一元素  
**下一步：**  
- 如果某一步就不一致 → kernel 内逻辑错误  
- 如果第一步一致但后续错 → softmax 或后续计算错误

---

# J. 如果仍无输出：排查死锁/挂起
**目标：** 排查 GPU hang 或死循环。

1. 每个 kernel 后强制 `cudaDeviceSynchronize()`  
2. 逐个 kernel 注释运行，看哪一个挂住  
3. 如果某个 kernel 一执行就无输出 → 该 kernel 内部可能死锁或非法访问  
**下一步：**  
- 聚焦该 kernel 的内存访问和同步

---

# 总结执行顺序（最重要）
1. **日志确认路径**  
2. **CUDA 错误检查 + 同步**  
3. **compute-sanitizer**  
4. **Nsight Systems**  
5. **缩小规模**  
6. **中间结果对比**  
7. **索引/步长核对**  
8. **资源限制检查**  
9. **单元素打印**  
10. **逐 kernel 排查 hang**

