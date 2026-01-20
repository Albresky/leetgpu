# Softmax

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
