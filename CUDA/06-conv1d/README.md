# Conv1D

## 性能数据对比

- RTX 4090 (sm89, Ada Lovelace)

| Kernel Version | Avg Latency (ms) | Memory Bandwidth (GB/s) | Throughput (GFLOPS) |
| :--- | :--- | :--- | :--- |
| **(__K0) Naive** | 0.0637 | 1.0282 | 234.2811 |
| **(__K1) Using Smem** | 0.0429(-32.65%) | 1.5276(+48.57%) | 342.2085(+46.07%) |


## Naive 实现

```bash
-------------------------------------------------------- ----------- ------------
Metric Name                                              Metric Unit Metric Value
-------------------------------------------------------- ----------- ------------
dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector    1,349,632
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector          897
smsp__inst_executed.sum                                         inst      816,098
smsp__sass_inst_executed_op_shared_ld.sum                       inst            0
smsp__sass_inst_executed_op_shared_st.sum                       inst            0
-------------------------------------------------------- ----------- ------------
```

## Using SMEM

```bash
-------------------------------------------------------- ----------- ------------
Metric Name                                              Metric Unit Metric Value
-------------------------------------------------------- ----------- ------------
dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector        2,944
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector          897
smsp__inst_executed.sum                                         inst      631,717
smsp__sass_inst_executed_op_shared_ld.sum                       inst      288,000
smsp__sass_inst_executed_op_shared_st.sum                       inst          768
-------------------------------------------------------- ----------- ------------
```


**目前的瓶颈：**

- 通过 SMEM 解决：
    - [x] 输入和卷积核重复从 GMEM 读取，没有利用 SMEM 进行数据复用
    - [x] 滑动窗口重叠极大