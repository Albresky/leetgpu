# Matrix Transpose

### 编译

`-O0` 禁用编译器优化。

Matrix sizes
 - M = 16384
 - N = 16384
 - K = 16384

### 性能数据对比

- RTX 4090 (sm89, Ada Lovelace)

| Kernel Version | Avg Latency (ms) | Memory Bandwidth (GB/s) | Compute (GFLOPS) | 总 Block 数量 | BlockDim | 总指令数量 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **(__K0) Naive** | 3.4611 | 620.4692 | 0 | $\frac{Rows \times Cols}{32 \times 32} $ | ${32 \times 32}$ | 243,269,632 |
| **(__K1) Using Smem for output memory coalescing** | 2.6166 | 820.7062 | 0 | $\frac{Rows \times Cols}{32 \times 32} $ | ${32 \times 32}$ | 243,269,632 |
| **(__K2) Using float4 + Smem for output memory coalescing** | 2.7007 | 795.1448 | 0 | $\frac{Rows \times Cols}{32 \times 32} $ | ${8 \times 32}$ | 81,788,928 |

### 分析

- K1 相较 K0 的带宽显著提升，原因在于通过 SMEM 解决了内存合并问题。K1 中每个 block 的线程读取 smem 的列，并写到结果的行，实现了 warp 内的访存合并（前提：输入输出均 row-major）。

- K2 相较 K1 的带宽略微下降，二者的线程块内线程数量一致。我在 K2 引入 float4 试图实现矢量化的 GMEM 读写，且保持 G2S 和 S2G 的访存合并特性，这亮点都没问题。性能差异的本质在于 S2G，此时每个线程从 SMEM 标量化地读取列（这 4 次 SMEM load操作并不能 reduce 为一条 vector-load 指令，因为它们地址是间隔的）。**值得注意**，本例通过 TILESZ + 1， 即 leading dimension  + 1 的方式，避免了 SMEM load 操作的 bank conflict。

### 总结

1. **K1 已通过 smem 实现 GMEM 合并访存，每次 GMEM 事务读/写 128B（warp 32 线程 × 4B）。**

```bash
  matrix_transpose_kernel(const float *, float *, int, int) (512, 512, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                1,518,357
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector   33,554,432
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector   33,554,432
    smsp__inst_executed.sum                                         inst  243,269,632
    smsp__sass_inst_executed_op_shared_ld.sum                       inst    8,388,608
    smsp__sass_inst_executed_op_shared_st.sum                       inst    8,388,608
    -------------------------------------------------------- ----------- ------------
```

2. **K2 用 float4 进一步把每个线程的 GMEM 事务从 4B 降至 16B。对于每个线程：**
    - 好处
      - GMEM 事务数减少 4 倍
    - 代价
      - SMEM 指令增加 4 倍
        - G2S: 1 次 smem store， 4 条 store
        - S2G: 1 次 smem load， 4 条 load

```bash
  matrix_transpose_kernel(const float *, float *, int, int) (512, 512, 1)x(8, 32, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector   33,554,432
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector   33,554,432
    smsp__inst_executed.sum                                         inst   81,788,928
    smsp__sass_inst_executed_op_shared_ld.sum                       inst    8,388,608
    smsp__sass_inst_executed_op_shared_st.sum                       inst    8,388,608
    -------------------------------------------------------- ----------- ------------
```

对于访存密集型 kernel（本例），瓶颈在 GMEM 带宽。但 K1 的 GMEM 访问已经完美合并，因此向量化的边际收益有限。而 smem 指令数量翻 4 倍带来的开销（指令发射、寄存器压力）反而抵消了 float4 的好处。