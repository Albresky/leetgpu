# Batch Normalization

## 思路

[Welfolr 算法](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm) 的 Naive 实现。

## 性能数据对比

- RTX 4090 (sm89, Ada Lovelace)

| Kernel Version | Avg Latency (ms) | Memory Bandwidth (GB/s) | Throughput (GFLOPS) |
| :--- | :--- | :--- | :--- |
| **(__K0) Welford/Naive (<=sm89)** | 0.0147 | 6.7882 | 4.1667 |


## Naive

```bash
MBM_kernel(const float *, float *, int, int) (1, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector          768
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector            8
    smsp__inst_executed.sum                                         inst          772
    smsp__sass_inst_executed_op_shared_ld.sum                       inst            0
    smsp__sass_inst_executed_op_shared_st.sum                       inst            0
    -------------------------------------------------------- ----------- ------------

  MBV_kernel(const float *, const float *, float *, int, int, float) (1, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector          776
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector            8
    smsp__inst_executed.sum                                         inst          980
    smsp__sass_inst_executed_op_shared_ld.sum                       inst            0
    smsp__sass_inst_executed_op_shared_st.sum                       inst            0
    -------------------------------------------------------- ----------- ------------

  norm_scale_shift_kernel(const float *, float *, float *, const float *, const float *, float *, int, int) (2, 3, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector        3,840
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector          768
    smsp__inst_executed.sum                                         inst        8,640
    smsp__sass_inst_executed_op_shared_ld.sum                       inst            0
    smsp__sass_inst_executed_op_shared_st.sum                       inst            0
    -------------------------------------------------------- ----------- ------------

  MBM_kernel(const float *, float *, int, int) (1, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector          768
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector            8
    smsp__inst_executed.sum                                         inst          772
    smsp__sass_inst_executed_op_shared_ld.sum                       inst            0
    smsp__sass_inst_executed_op_shared_st.sum                       inst            0
    -------------------------------------------------------- ----------- ------------

  MBV_kernel(const float *, const float *, float *, int, int, float) (1, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector          776
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector            8
    smsp__inst_executed.sum                                         inst          980
    smsp__sass_inst_executed_op_shared_ld.sum                       inst            0
    smsp__sass_inst_executed_op_shared_st.sum                       inst            0
    -------------------------------------------------------- ----------- ------------

  norm_scale_shift_kernel(const float *, float *, float *, const float *, const float *, float *, int, int) (2, 3, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector        3,840
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector          768
    smsp__inst_executed.sum                                         inst        8,640
    smsp__sass_inst_executed_op_shared_ld.sum                       inst            0
    smsp__sass_inst_executed_op_shared_st.sum                       inst            0
    -------------------------------------------------------- ----------- ------------
```
