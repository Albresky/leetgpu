# RMS Normalization

均方根（Root Mean Square）正则化。

RMS 数学原理、与 LayerNorm 区别 参考：
- [均方根归一化 RMSNorm 详解：原理、实现与应用](https://blog.csdn.net/shizheng_Li/article/details/145830637)
- [均方根归一化（RMSNorm）简介与推导](https://inuyashayang.github.io/AIDIY/LLM_Pages/Normalization/RMSNorm/)

## 性能数据对比

- RTX 4090 (sm89, Ada Lovelace)

| Kernel Version | Avg Latency (ms) | Memory Bandwidth (GB/s) | Throughput (GFLOPS) |
| :--- | :--- | :--- | :--- |
| **(__K0) Naive (<=sm89)** | 0.0235 | 1.0486 | 0.4362 |


## Naive

```bash
bsum_kernel(const float *, float *, int) (8, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector          256
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector            8
    smsp__inst_executed.sum                                         inst        6,696
    smsp__sass_inst_executed_op_shared_ld.sum                       inst          264
    smsp__sass_inst_executed_op_shared_st.sum                       inst          224
    -------------------------------------------------------- ----------- ------------

  rms_kernel(float *, float *, int, int, float) (1, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector            1
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector            1
    smsp__inst_executed.sum                                         inst          817
    smsp__sass_inst_executed_op_shared_ld.sum                       inst           25
    smsp__sass_inst_executed_op_shared_st.sum                       inst           20
    -------------------------------------------------------- ----------- ------------

  rms_norm_kernel(const float *, const float *, float, float, float *, int) (8, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector          320
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector          256
    smsp__inst_executed.sum                                         inst        1,856
    smsp__sass_inst_executed_op_shared_ld.sum                       inst            0
    smsp__sass_inst_executed_op_shared_st.sum                       inst            0
    -------------------------------------------------------- ----------- ------------
```
