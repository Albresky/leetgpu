# Softmax Attention (Scaled Dot-Product Attention)

## Problem

**Title:** Softmax Attention (Scaled Dot-Product Attention)

**Difficulty:** Medium

**Description:**
Implement the Scaled Dot-Product Attention function as described in the paper "*[Attention Is All You Need - 2017](https://arxiv.org/abs/1706.03762)*".

Given a query matrix $Q$ of size $M \times d_k$, a key matrix $K$ of size $N \times d_k$, and a value matrix $V$ of size $N \times d_v$, compute the output matrix $O$ of size $M \times d_v$.

The mathematical formula is:

$$
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


**Specifics:**

1. **Scaling:** Before softmax, scale the dot product by $\frac{1}{\sqrt{d_k}}$.
2. **Softmax:** Applied row-wise (along the dimension of size $N$).
3. **Output:** Store the result in `output`.

**Constraints:**

* Matrix dimensions:
  - $Q$: $M \times d_k$
  - $K$: $N \times d_k$
  - $V$: $N \times d_v$ (for simplicity, assume $d_k = d_v = d$)
  - $O$: $M \times d$
* $1 \leq M, N \leq 32768$ (Practical size for naive without blocking)
* $1 \leq d \leq 1024$ (Dimension size)
* External libraries (cuBLAS, cuDNN) are **NOT** permitted.
* Assume row-major storage for all matrices.

**Function Signature:**

```cpp
void solve(float* Q, float* K, float* V, float* output, int M, int N, int d);
```

## 简介

> 网上关于 self-attention 看到很多名称：Softmax Attention，Self-Attention，Scaled Dot-Product Attention，但其数学本质一样。这里纪要这些命名区别。

**这三者分别描述了同一个机制的不同维度：**

* **Self-Attention：** 描述**应用场景**（数据的来源）。指 Q、K、V 三个矩阵均来自**同一个输入源**
* **Scaled Dot-Product Attention：** 描述**具体数学实现**。指通过点积计算相似度，并除以缩放因子  的方法。
* **Softmax Attention：** 描述**归一化方式**。指使用 *Softmax* 函数将得分为概率分布。在 Transformer 的语境下，它通常等同于 Scaled Dot-Product Attention。

简言之：**我们通常使用“Scaled Dot-Product Attention”这个数学公式，来实现“Self-Attention”这一功能模块。**

---

### 原理

为清晰区分，我们需要从 **数据流向（Flow）** 和 **计算内核（Kernel）** 两个角度拆解：

#### 1. Self-Attention 描述“数据从哪来”

从架构层面定义，特征在于 **Q, K, V 同源**。

* **定义：** 输入序列  经过三个不同的线性变换矩阵 ($W_Q, W_K, W_V$) 分别得到 $Q, K, V$。即 $Q = XW_Q, K = XW_K, V = XW_V$ 。
* **意义：** 让序列中的每一个元素都能看到序列中的其他所有元素，从而捕捉序列内部的依赖关系。
* **对比：** 如果 $Q$ 来自解码器， $K, V$ 来自编码器，那便是 **Cross-Attention（交叉注意力）**，而非 Self-Attention。

#### 2. Scaled Dot-Product Attention 描述“内部怎么算”

从数学层面定义：这是一个函数 $f(Q, K, V)$:

$$
f(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**其计算步骤如下**：

  - 1. **Dot-Product** 计算相似度矩阵: $S = QK^T$ （点积）

  - 2. **Scaled** 缩放： $S' = \frac{S}{\sqrt{d_k}}$ （除以维度的平方根）

  - 3. **Softmax** 归一化： $P = softmax(S')$ （Softmax）, 将每一行转为概率分布

  - 4. **Weighted Sum** 加权求和： $O = PV$ （乘以 V 得到输出）


#### 3. Softmax Attention 描述归一化方式

从归一化方式定义：任何使用 Softmax 作为归一化算子的注意力机制。

 因为 Scaled Dot-Product Attention 的核心步骤是 Softmax，所以它属于 Softmax Attention 的一种。与之相对的是 Linear Attention 或 Sparse Attention（使用 ReLU 或其他稀疏化算子代替 Softmax）。

---

### 结构化对比

总结如下表；

| 术语 | 关注点 | 特征 |
| --- |  --- | --- |
| **Self-Attention** | **Who** | 输入  均源自同一序列 $X$ |
| **Scaled Dot-Product** | **How** | 使用  $QK^T$点积 +  $\frac{1}{\sqrt{d_k}}$缩放 + Softmax |
| **Softmax Attention** | **Normalization** | 使用 Softmax 函数将 Score 转为概率分布 |


## 思路

本质就是执行 $M$ 次独立的 1D **3-Pass Softmax**($S_i$) 操作，求得 $S_{M \times N}= Q \times K^T$ 每一行的概率分布；再乘以 $V$， 对 $d_v$ 维度的每个向量进行加权求和。其中，**3-Pass** Softmax 指需要遍历 3 次 Q 矩阵的特征向量，每次**分别计算**：max、sumexp、final output，这会产生 3 次 S 矩阵 的 GMEM 重复访存。

### 关于 $C_{M \times N} = A_{M \times K} \times B_{N \times K}^T$ 的线程索引

在 `__global__ void mm_transposed_kernel()` 设备函数中，我们对B矩阵的加载，在 Grid 和 ThreadBlock 层面是 X、Y轴 **交错** 的，即：进行了**物理位置的重排（或者叫隐式转置）**，以便后续计算可以按行读取。。

```cpp
// B
// load B^T
// bidx -->N, tidy < TileN, tidx --> dk
if (blockIdx.x * Tile + tidy < N && k * Tile + tidx < K)
  s_B[tidx][tidy] = B[(blockIdx.x * Tile + tidy) * K + k * Tile + tidx];
else
  s_B[tidx][tidy] = .0f;
```

该**交错**在全局（Grid）层面利用 X 维度，而在局部搬运（Block）层面利用 Y 维度进行索引，这是为了实现 **GMEM 的 Coalesced Access**。

**访存合并 (Coalesced Access) 分析：**

1.  **全局内存 GMEM 读取**：
    矩阵 $K$ 是行主序，连续的地址在 $d$ (特征) 维度。
    因此，warp 中的连续线程 (`threadIdx.x`) 应当读取连续的 $d$ 维度索引。
    
    代码如下：
    ```cpp
    // tidx 映射到 K 维度 (d), tidy 映射到 N 维度
    idx = (... + tidy) * K_dim + (... + tidx); 
    ```
    这样保证了从 GMEM 读取 $B$ 是 Coalesced。

2.  **共享内存 SMEM 写入与布局**：
    `s_B[tidx][tidy] = ...`
    这里我们将读取到的数据写入 SMEM，这里通常需要考虑 Bank Conflict。
    如果是为了后续计算方便（通常计算在这个Tile上做外积或者点积），数据在 SMEM 中的排布通常会根据计算的核心循环 (Inner Loop) 方向来定。

**具体而言，我们有如下观察：**

$$
{GMEM_{rowIdx(Dim_N)}} = \underbrace{\text{blockIdx.x} \times \text{NTile}}_{\text{Grid X}} + \underbrace{\text{threadIdx.y}}_{\text{Block Y}}
$$

1. **数据的物理布局**：
    矩阵 $B$ 是 **行主序（Row-Major）** 的，维度是 $(N, K)$。
    这意味着 **$K$ 维度（列方向）在内存地址上是连续的**，而 $N$ 维度（行方向）是大跨度的。

2.  **GPU 方寸原则**：
    GPU 的一个 Warp（32个线程，即连续的 `threadIdx.x`）如果去读取 **连续的内存地址**，就能实现合并访问，效率最高。
    因此，如果要高效率读取 $B$，必须让 **`threadIdx.x` 负责 $K$ 维度**。

3.  **行主序转置矩阵的访存策略**：
    *   **计算时的逻辑映射**：为计算 $C = A \times B^T$，我们通常让 Grid 的 X 轴对应输出矩阵 $C$ 的列（也就是 $N$ 维度）。所以 `blockIdx.x` 锁定了 $N$ 维度的一大块。
    *   **加载时的物理映射**：
        *   我们需要加载一个 Tile 的 $B$ 矩阵
        *   为了让内存读取连续，我们指定 `threadIdx.x` 去索引 $K$ 维度
        *   那么剩下的那个维度（$N$ 维度），就只能交给 `threadIdx.y` 索引

4.  **结果**：
    *   **Grid 层**：`blockIdx.x` 决定处理哪几行（ $N$ 维度 ）
    *   **Block 内**：`threadIdx.y` 决定当前线程负责这几行里的哪一行（ $N$ 维度偏移 ）
    *   **Block 内**：`threadIdx.x` 决定当前线程读取该行的哪一列（ $K$ 维度 ）


## 性能数据对比

- RTX 4090 (sm89, Ada Lovelace)

| Kernel Version | Avg Latency (ms) | Memory Bandwidth (GB/s) | Throughput (GFLOPS) |
| :--- | :--- | :--- | :--- |
| **__K0** | 1.9342 | 17.3604 | 2223.2113 |


## Naive

```bash
  void mm_kernel<16>(const float *, const float *, float *, int, int, int, float) (64, 64, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector   16,777,216
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector      131,072
    smsp__inst_executed.sum                                         inst  129,400,832
    smsp__sass_inst_executed_op_shared_ld.sum                       inst   41,943,040
    smsp__sass_inst_executed_op_shared_st.sum                       inst    4,194,304
    -------------------------------------------------------- ----------- ------------
    
  void mm_transposed_kernel<16>(const float *, const float *, float *, int, int, int, float) (64, 64, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector   16,777,216
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector      131,072
    smsp__inst_executed.sum                                         inst  121,110,528
    smsp__sass_inst_executed_op_shared_ld.sum                       inst   41,943,040
    smsp__sass_inst_executed_op_shared_st.sum                       inst    4,194,304
    -------------------------------------------------------- ----------- ------------

  bmax_kernel(const float *, float *, int) (2, 1024, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector      131,072
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector        2,048
    smsp__inst_executed.sum                                         inst    3,682,304
    smsp__sass_inst_executed_op_shared_ld.sum                       inst       83,968
    smsp__sass_inst_executed_op_shared_st.sum                       inst       73,728
    -------------------------------------------------------- ----------- ------------

  max_kernel(const float *, float *, int) (1024, 1, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector        1,024
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector        1,024
    smsp__inst_executed.sum                                         inst    1,802,240
    smsp__sass_inst_executed_op_shared_ld.sum                       inst       41,984
    smsp__sass_inst_executed_op_shared_st.sum                       inst       36,864
    -------------------------------------------------------- ----------- ------------

  bsum_kernel(const float *, float *, const float *, int) (2, 1024, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector      163,840
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector        2,048
    smsp__inst_executed.sum                                         inst    4,108,288
    smsp__sass_inst_executed_op_shared_ld.sum                       inst       83,968
    smsp__sass_inst_executed_op_shared_st.sum                       inst       73,728
    -------------------------------------------------------- ----------- ------------

  sum_kernel(const float *, float *, int) (1024, 1, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector        1,024
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector        1,024
    smsp__inst_executed.sum                                         inst    1,802,240
    smsp__sass_inst_executed_op_shared_ld.sum                       inst       41,984
    smsp__sass_inst_executed_op_shared_st.sum                       inst       36,864
    -------------------------------------------------------- ----------- ------------

  norm_kernel(const float *, float *, const float *, const float *, int) (2, 1024, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    dram__throughput.avg_pct_of_peak_sustained_elapsed                        (!) n/a
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                sector      196,608
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                sector      131,072
    smsp__inst_executed.sum                                         inst    1,343,488
    smsp__sass_inst_executed_op_shared_ld.sum                       inst            0
    smsp__sass_inst_executed_op_shared_st.sum                       inst            0
    -------------------------------------------------------- ----------- ------------
```

## 将手写 CUDA 算子注册到 PyTorch

以 Softmax Attention 算子为例，介绍如何通过 **PyTorch C++ Extension** 机制，将手写 CUDA Kernel 封装为 PyTorch 的 Python API。

### 1. 编写 C++ 绑定接口 (`attention_bind.cpp`)

我们需要一个 C++ 文件作为桥梁，负责接收 Python 传入的 `torch::Tensor`，进行必要的检查和转换、提取数据指针，然后调用手写的 CUDA 函数。

关键组件：
*   **`#include <torch/extension.h>`**：引入 PyTorch C++ API
*   **输入检查**：PyTorch Tensor 在内存中可能是非连续的（View 等操作导致），而大多数自定义 CUDA Kernel 假设内存连续，因此必须使用 `.contiguous()` 确保数据紧凑
*   **数据指针提取**：使用 `.data_ptr<float>()` 获取 GMEM 首地址传给 CUDA Kernel
*   **PYBIND11 宏**：定义 Python 模块名称和函数名称

```cpp
#include <torch/extension.h>

// 1. 声明原始 CUDA C 函数
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d);

// 2. C++ Wrapper
torch::Tensor attention_cuda_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // 确保 Tensor 在内存中是连续的
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();

    // Shape 检查，参数提取
    // ...

    // 分配输出 Tensor
    auto output = torch::empty({M, d}, torch::dtype(torch::kFloat32).device(Q.device()));

    // 调用原始 CUDA 函数，传入裸指针
    solve(Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), output.data_ptr<float>(), M, N, d);

    return output;
}

// 3. 注册为 Python 模块
// TORCH_EXTENSION_NAME 会在 setup.py 编译时确定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_cuda_forward, "Attention forward (CUDA)");
}
```

### 2. 编写构建脚本 (`setup.py`)

使用 `setuptools` 和 `torch.utils.cpp_extension` 来编译 `.cpp` 和 `.cu` 文件。

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='softmax_attention', # 编译后的包名
    ext_modules=[
        CUDAExtension(
            'softmax_attention', 
            [
                'attention_bind.cpp', # C++ 绑定文件
                'attention.cu'        # CUDA 核心实现
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
```

### 3. 编译与使用

在目录下运行安装命令：

```bash
python setup.py install
```

然后在 Python 中即可直接调用：

```python
import torch
import softmax_attention # 导入自定义模块

# ... 准备数据 ...
# 调用我们在 PYBIND11_MODULE 中定义的 .def("forward", ...)
output = softmax_attention.forward(Q, K, V)
```
