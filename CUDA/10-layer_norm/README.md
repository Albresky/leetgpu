# LayerNorm


## 性能数据对比

- RTX 4090 (sm89, Ada Lovelace)

| Kernel Version | Avg Latency (ms) | Memory Bandwidth (GB/s) | Throughput (GFLOPS) |
| :--- | :--- | :--- | :--- |
| **(__K0) Naive (<=sm89)** | 0.0285 | 1.1521 | 0.5574 |


## Naive

```bash
 bsum_kernel(const float *, float *, float) (4, 1, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector          256
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector            4
    smsp__inst_executed.sum                                         inst        7,124
    smsp__sass_inst_executed_op_shared_ld.sum                       inst          164
    smsp__sass_inst_executed_op_shared_st.sum                       inst          144
    -------------------------------------------------------- ----------- ------------

  sum_kernel(const float *, float *, float, float) (1, 1, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector            1
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector            1
    smsp__inst_executed.sum                                         inst        1,769
    smsp__sass_inst_executed_op_shared_ld.sum                       inst           41
    smsp__sass_inst_executed_op_shared_st.sum                       inst           36
    -------------------------------------------------------- ----------- ------------

  bvar_kernel(const float *, float *, const float *, int) (4, 1, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector          320
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector            4
    smsp__inst_executed.sum                                         inst        7,380
    smsp__sass_inst_executed_op_shared_ld.sum                       inst          164
    smsp__sass_inst_executed_op_shared_st.sum                       inst          144
    -------------------------------------------------------- ----------- ------------

  var_kernel(float *, float *, int, float, int) (1, 1, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector            1
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector            1
    smsp__inst_executed.sum                                         inst        1,761
    smsp__sass_inst_executed_op_shared_ld.sum                       inst           41
    smsp__sass_inst_executed_op_shared_st.sum                       inst           36
    -------------------------------------------------------- ----------- ------------

  norm_kernel(const float *, float *, const float *, const float *, float, float, int) (4, 1, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector          384
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector          256
    smsp__inst_executed.sum                                         inst        2,112
    smsp__sass_inst_executed_op_shared_ld.sum                       inst            0
    smsp__sass_inst_executed_op_shared_st.sum                       inst            0
    -------------------------------------------------------- ----------- ------------
```