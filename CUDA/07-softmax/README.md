# Softmax

## 思路

目前 Ada 架构不支持 DSMEM (distributed shared memory)，不能跨block进行数据reduce，因此这里我们将在 SMEM 上面分两步进行规约：**第1步**， 每个block处理一部分gmem数据；**第2步**，单个 block 对第一步规约的结果再进行规约，得到最终规约结果。实现findmax和reduce到sum值，步骤完全一致。

简单而言，以 findmax 为例。将输入的 N 个数据分布到 $BlockPerGrid = \lceil \frac{N}{BlockDim} \rceil$ 个线程块，每个线程块求本地的max值，得到 $BlockPerGrid$ 个局部 max 值；然后使用 1 个线程块，对这$BlockPerGrid$ 个值再找 max，此时该线程块内每个元素需串行找 $\frac{BlockPerGrid}{BlockDim}$ 次。

*未来计划实现 Hopper（Blackwell）下利用 DSMEM 的跨线程块算法。*

## 性能数据对比

- RTX 4090 (sm89, Ada Lovelace)

| Kernel Version | Avg Latency (ms) | Memory Bandwidth (GB/s) | Throughput (GFLOPS) |
| :--- | :--- | :--- | :--- |
| **(__K0) Naive (<=sm89)** | 0.0297 | 4.4294 | 1.9301 |


## Naive

```bash
block_max_kernel(const float *, float *, int) (32, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector        1,024
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector           32
    smsp__inst_executed.sum                                         inst       25,760
    smsp__sass_inst_executed_op_shared_ld.sum                       inst          800
    smsp__sass_inst_executed_op_shared_st.sum                       inst          640
    -------------------------------------------------------- ----------- ------------
reduce_bmax_kernel(const float *, float *, int) (1, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector            4
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector            1
    smsp__inst_executed.sum                                         inst          798
    smsp__sass_inst_executed_op_shared_ld.sum                       inst           25
    smsp__sass_inst_executed_op_shared_st.sum                       inst           20
    -------------------------------------------------------- ----------- ------------

block_sum_kernel(const float *, const float *, float *, int) (32, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector        1,280
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector           32
    smsp__inst_executed.sum                                         inst       29,344
    smsp__sass_inst_executed_op_shared_ld.sum                       inst          800
    smsp__sass_inst_executed_op_shared_st.sum                       inst          640
    -------------------------------------------------------- ----------- ------------

reduce_bsum_kernel(const float *, float *, int) (1, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector            4
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector            1
    smsp__inst_executed.sum                                         inst          798
    smsp__sass_inst_executed_op_shared_ld.sum                       inst           25
    smsp__sass_inst_executed_op_shared_st.sum                       inst           20
    -------------------------------------------------------- ----------- ------------

softmax_kernel(const float *, float *, int, const float *, const float *) (32, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector        1,536
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector        1,024
    smsp__inst_executed.sum                                         inst       11,264
    smsp__sass_inst_executed_op_shared_ld.sum                       inst          256
    smsp__sass_inst_executed_op_shared_st.sum                       inst          256
    -------------------------------------------------------- ----------- ------------
```
