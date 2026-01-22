# NVIDIA Nsight Compute (NCU) Profiling 指南

## 目录

- [基础用法](#基础用法)
- [常用指标分类](#常用指标分类)
- [典型场景命令](#典型场景命令)
- [指标查询方法](#指标查询方法)
- [实战案例](#实战案例)

---

## 基础用法

### 基本命令格式

```bash
ncu [options] <application> [application-arguments]
```

### 常用选项

| 选项 | 说明 |
|:--|:--|
| `--metrics <metrics>` | 指定要收集的指标（逗号分隔） |
| `--kernel-name <name>` | 只 profile 指定名称的 kernel |
| `--kernel-id <id>` | 只 profile 第 N 次 kernel 调用 |
| `--launch-skip <n>` | 跳过前 N 次 kernel 调用 |
| `--launch-count <n>` | 只 profile N 次 kernel 调用 |
| `-o <file>` | 输出到 .ncu-rep 文件（可用 GUI 打开） |
| `--set <set>` | 使用预定义指标集 |
| `--section <section>` | 收集特定 section 的指标 |
| `-f` | 覆盖已存在的输出文件 |

### 预定义指标集 (--set)

```bash
ncu --list-sets  # 查看所有可用的 set
```

| Set | 说明 |
|:--|:--|
| `default` | 默认指标集 |
| `full` | 所有指标（最慢，最全） |
| `basic` | 基础性能指标 |
| `roofline` | Roofline 分析所需指标 |

### 预定义 Section (--section)

```bash
ncu --list-sections  # 查看所有可用的 section
```

| Section | 说明 |
|:--|:--|
| `ComputeWorkloadAnalysis` | 计算负载分析 |
| `MemoryWorkloadAnalysis` | 内存负载分析 |
| `LaunchStats` | Kernel 启动统计 |
| `Occupancy` | 占用率分析 |
| `SchedulerStats` | 调度器统计 |
| `SpeedOfLight` | 性能上限分析 |
| `WarpStateStats` | Warp 状态统计 |

---

## 常用指标分类

### 1. Global Memory (GMEM) 指标

| 指标 | 说明 |
|:--|:--|
| `dram__bytes.sum` | DRAM 总传输字节数 |
| `dram__bytes_read.sum` | DRAM 读取字节数 |
| `dram__bytes_write.sum` | DRAM 写入字节数 |
| `dram__throughput.avg_pct_of_peak_sustained_elapsed` | DRAM 带宽利用率 (%) |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | GMEM load 事务数（sectors） |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` | GMEM store 事务数（sectors） |
| `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` | GMEM load 字节数 |
| `l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum` | GMEM store 字节数 |

#### GMEM 合并效率

```bash
# 理想情况：每个请求对应最少的 sectors
# sectors / requests 越接近 1 越好（完美合并时为 1）
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
```

### 2. Shared Memory (SMEM) 指标

| 指标 | 说明 |
|:--|:--|
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` | SMEM load wavefront 数 |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum` | SMEM store wavefront 数 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | SMEM load bank conflict 数 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` | SMEM store bank conflict 数 |
| `smsp__sass_inst_executed_op_shared_ld.sum` | SMEM load 指令执行数 |
| `smsp__sass_inst_executed_op_shared_st.sum` | SMEM store 指令执行数 |

### 3. L1/L2 Cache 指标

| 指标 | 说明 |
|:--|:--|
| `l1tex__t_sector_hit_rate.pct` | L1 cache 命中率 |
| `lts__t_sector_hit_rate.pct` | L2 cache 命中率 |
| `lts__t_sectors_srcunit_tex_op_read.sum` | L2 读取 sector 数 |
| `lts__t_sectors_srcunit_tex_op_write.sum` | L2 写入 sector 数 |

### 4. 指令执行指标

| 指标 | 说明 |
|:--|:--|
| `smsp__inst_executed.sum` | 总执行指令数 |
| `smsp__inst_executed_op_fp32.sum` | FP32 指令数 |
| `smsp__inst_executed_op_fp16.sum` | FP16 指令数 |
| `smsp__inst_executed_op_integer.sum` | 整数指令数 |
| `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum` | FP add 指令数 |
| `smsp__sass_thread_inst_executed_op_fmul_pred_on.sum` | FP mul 指令数 |
| `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` | FMA 指令数 |

### 5. Occupancy 指标

| 指标 | 说明 |
|:--|:--|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 实际 warp 占用率 |
| `launch__occupancy_limit_registers` | 寄存器限制的占用率 |
| `launch__occupancy_limit_shared_mem` | SMEM 限制的占用率 |
| `launch__occupancy_limit_blocks` | Block 数限制的占用率 |
| `launch__occupancy_limit_warps` | Warp 数限制的占用率 |
| `launch__registers_per_thread` | 每线程寄存器数 |
| `launch__shared_mem_per_block_allocated` | 每 block 分配的 SMEM |

### 6. Warp 调度指标

| 指标 | 说明 |
|:--|:--|
| `smsp__warps_issue_stalled_wait_any` | Stall 的 warp 数 |
| `smsp__warps_issue_stalled_long_scoreboard` | 等待长延迟操作的 warp |
| `smsp__warps_issue_stalled_short_scoreboard` | 等待短延迟操作的 warp |
| `smsp__warps_issue_stalled_mio_throttle` | MIO 限流导致的 stall |
| `smsp__warps_issue_stalled_lg_throttle` | LG 限流导致的 stall |
| `smsp__warps_issue_stalled_tex_throttle` | TEX 限流导致的 stall |

### 7. 计算吞吐指标

| 指标 | 说明 |
|:--|:--|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 吞吐利用率 |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 计算内存吞吐 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed` | FMA 管线利用率 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | Tensor Core 利用率 |

---

## 典型场景命令

### 场景 1: 内存带宽分析

```bash
ncu --metrics \
dram__bytes.sum,\
dram__throughput.avg_pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct \
./your_program
```

### 场景 2: SMEM 分析（Bank Conflict）

```bash
ncu --metrics \
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
smsp__sass_inst_executed_op_shared_ld.sum,\
smsp__sass_inst_executed_op_shared_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum \
./your_program
```

### 场景 3: Occupancy 分析

```bash
ncu --metrics \
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,\
launch__registers_per_thread,\
launch__shared_mem_per_block_allocated \
./your_program
```

### 场景 4: 计算 vs 访存分析（Roofline）

```bash
ncu --set roofline -o roofline_report ./your_program
# 然后用 ncu-ui 打开 roofline_report.ncu-rep
```

### 场景 5: Warp Stall 分析

```bash
ncu --metrics \
smsp__warps_issue_stalled_long_scoreboard,\
smsp__warps_issue_stalled_short_scoreboard,\
smsp__warps_issue_stalled_mio_throttle,\
smsp__warps_issue_stalled_lg_throttle,\
smsp__warps_issue_stalled_wait \
./your_program
```

### 场景 6: 完整 Profile（生成报告）

```bash
ncu --set full -o full_report ./your_program
# 然后用 ncu-ui 打开查看
```

### 场景 7: 只 Profile 特定 Kernel

```bash
# 按 kernel 名称
ncu --kernel-name "matrix_transpose_kernel" --metrics ... ./your_program

# 按调用次序（跳过前 5 次，收集接下来 3 次）
ncu --launch-skip 5 --launch-count 3 --metrics ... ./your_program
```

---

## 指标查询方法

### 查看所有可用指标

```bash
ncu --query-metrics
```

### 按关键字搜索指标

```bash
ncu --query-metrics | grep -i "shared"    # 搜索 SMEM 相关
ncu --query-metrics | grep -i "dram"      # 搜索 DRAM 相关
ncu --query-metrics | grep -i "bank"      # 搜索 bank conflict 相关
ncu --query-metrics | grep -i "sector"    # 搜索 sector 相关
ncu --query-metrics | grep -i "occupancy" # 搜索占用率相关
```

### 查看指标详细说明

```bash
ncu --query-metrics-mode all | grep -A5 "l1tex__data_bank_conflicts"
```

---

## 实战案例

### 案例 1: 矩阵转置优化验证

**目标**: 验证 GMEM 合并访问和 SMEM bank conflict

```bash
# 编译两个版本
nvcc -D__K1 -o transpose_k1 matrixtrans.cu
nvcc -D__K2 -o transpose_k2 matrixtrans.cu

# Profile K1
ncu --metrics \
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
smsp__sass_inst_executed_op_shared_ld.sum,\
smsp__sass_inst_executed_op_shared_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
./transpose_k1

# Profile K2
ncu --metrics \
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
smsp__sass_inst_executed_op_shared_ld.sum,\
smsp__sass_inst_executed_op_shared_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
./transpose_k2
```

**预期结果**:
| 指标 | K1 | K2 | 说明 |
|:--|:--|:--|:--|
| GMEM load sectors | 高 | 低 (1/4) | float4 减少事务 |
| GMEM store sectors | 高 | 低 (1/4) | float4 减少事务 |
| SMEM load 指令数 | 低 | 高 (4x) | 标量读取 |
| SMEM store 指令数 | 低 | 高 (4x) | 标量写入 |
| SMEM bank conflict | ~0 | ~0 | +1 padding 有效 |

### 案例 2: GEMM 性能分析

```bash
ncu --metrics \
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
dram__throughput.avg_pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
./gemm_kernel
```

---

## 常见问题

### Q1: 为什么有些指标显示 N/A？

某些指标只在特定架构或特定条件下可用。使用 `--query-metrics` 查看当前 GPU 支持的指标。

### Q2: Profile 太慢怎么办？

1. 减少指标数量，只收集关心的
2. 使用 `--launch-count 1` 只 profile 一次调用
3. 使用 `--set basic` 而不是 `--set full`

### Q3: 如何对比两个 kernel？

```bash
# 生成两个报告
ncu -o report_v1 ./kernel_v1
ncu -o report_v2 ./kernel_v2

# 用 ncu-ui 打开并对比
ncu-ui report_v1.ncu-rep report_v2.ncu-rep
```

### Q4: 如何确认 GMEM 合并是否完美？

```bash
ncu --metrics \
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
./your_program

# 完美合并: sectors/requests ≈ 4 (128B / 32B per sector)
# 无合并: sectors/requests ≈ 32
```

---

## 参考资料

- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
