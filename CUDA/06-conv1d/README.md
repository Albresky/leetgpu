# Conv1D


## Naive 实现

```bash
-------------------------------------------------------- ----------- ------------
Metric Name                                              Metric Unit Metric Value
-------------------------------------------------------- ----------- ------------
dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector            6
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector            1
smsp__inst_executed.sum                                         inst          120
smsp__sass_inst_executed_op_shared_ld.sum                       inst            0
smsp__sass_inst_executed_op_shared_st.sum                       inst            0
-------------------------------------------------------- ----------- ------------
```

**目前的瓶颈：**

- 输入和卷积核重复从 GMEM 读取，没有利用 SMEM 进行数据复用
- 滑动窗口重叠极大