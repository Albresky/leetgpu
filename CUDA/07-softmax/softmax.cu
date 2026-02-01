#include <cuda_runtime.h>
#include <math.h>
#include <cfloat>

#define __ONLINE_SOFTMAX__

//////////////////////////////////////////////////////////////////
////////////////////// 1. Before sm90 (Hopper) //////////////////////
//////////////////////////////////////////////////////////////////
__global__ void block_max_kernel(const float* input, float* bmax, int N)
{
  __shared__ float s_input[256];

  int gidx      = blockIdx.x * blockDim.x + threadIdx.x;
  int tidx      = threadIdx.x;
  s_input[tidx] = (gidx < N) ? input[gidx] : -INFINITY;
  __syncthreads();

  // tree findmax
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tidx < s) s_input[tidx] = fmaxf(s_input[tidx + s], s_input[tidx]);
    __syncthreads();
  }
  if (tidx == 0) bmax[blockIdx.x] = s_input[0];
}

__global__ void reduce_bmax_kernel(const float* bmax, float* out_max, int n)
{
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int idx = tid;

  float vmax = -INFINITY;
  while (idx < n) {
    vmax = fmaxf(vmax, bmax[idx]);
    idx += blockDim.x;
  }
  sdata[tid] = vmax;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }

  if (tid == 0) out_max[0] = sdata[0];
}

__global__ void block_sum_kernel(const float* input, const float* max_val, float* bsum, int N)
{
  __shared__ float s_input[256];

  int gidx      = blockIdx.x * blockDim.x + threadIdx.x;
  int tidx      = threadIdx.x;
  float m       = max_val[0];
  s_input[tidx] = (gidx < N) ? expf(input[gidx] - m) : 0.0f;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tidx < s) s_input[tidx] += s_input[tidx + s];
    __syncthreads();
  }
  if (tidx == 0) bsum[blockIdx.x] = s_input[0];
}

__global__ void reduce_bsum_kernel(const float* bsum, float* out_sum, int n)
{
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int idx = tid;

  float vsum = 0.0f;
  while (idx < n) {
    vsum += bsum[idx];
    idx += blockDim.x;
  }
  sdata[tid] = vsum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0) out_sum[0] = sdata[0];
}

__global__ void softmax_kernel(
  const float* input, float* output, int N, const float* max_val, const float* sum_val)
{
  __shared__ float s_input[256];

  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidx = threadIdx.x;
  float m  = max_val[0];
  float s  = sum_val[0];
  if (gidx < N) s_input[tidx] = expf(input[gidx] - m);
  __syncthreads();

  if (gidx < N) output[gidx] = s_input[tidx] / s;
}

//////////////////////////////////////////////////////////////////
/////////////////////// 2. Online Softmax ////////////////////////
//////////////////////////////////////////////////////////////////

// m: current max
// d: current denominator (sum of exp)
struct MD {
  float m;
  float d;
};

__device__ __forceinline__ MD init_md() { return {-FLT_MAX, 0.0f}; }

__device__ __forceinline__ MD combine(MD a, MD b)
{
  MD out;
  out.m = fmaxf(a.m, b.m);

  // d_new = d_a * exp(m_a - m_new) + d_b * exp(m_b - m_new)
  out.d = a.d * expf(a.m - out.m) + b.d * expf(b.m - out.m);

  return out;
}

__device__ __forceinline__ MD warp_reduce_md(MD val)
{
  for (int offset = 16; offset > 0; offset /= 2) {
    float other_m = __shfl_down_sync(0xffffffff, val.m, offset);
    float other_d = __shfl_down_sync(0xffffffff, val.d, offset);

    MD other = {other_m, other_d};
    val      = combine(val, other);
  }
  return val;
}

// Pass 1: compute m, d in block-level
__global__ void online_reduce_kernel(const float* input, float* bmax, float* bsum, int N)
{
  // 1. Thread Local Reduction (Grid-Stride Loop)
  MD local_val = init_md();

  int tid    = threadIdx.x;
  int idx    = blockIdx.x * blockDim.x + tid;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < N; i += stride) {
    MD elem   = {input[i], 1.0f};
    local_val = combine(local_val, elem);
  }

  // 2. Warp Reduction
  local_val = warp_reduce_md(local_val);

  // 3. Block Reduction
  // BlockSize = 256, 8 Warps
  static __shared__ float s_m[32];
  static __shared__ float s_d[32];

  int lane    = tid % 32;
  int warp_id = tid / 32;

  if (lane == 0) {
    s_m[warp_id] = local_val.m;
    s_d[warp_id] = local_val.d;
  }
  __syncthreads();

  MD block_val = init_md();
  if (tid < (blockDim.x / 32)) {
    block_val.m = s_m[tid];
    block_val.d = s_d[tid];
  }

  if (warp_id == 0) {
    block_val = warp_reduce_md(block_val);

    if (tid == 0) {
      bmax[blockIdx.x] = block_val.m;
      bsum[blockIdx.x] = block_val.d;
    }
  }
}

// Pass 2: get global m, d
// input:  bmax[], bsum[]
// output: d_max[0], d_sum[0]
__global__ void global_reduce_kernel(
  const float* bmax, const float* bsum, float* out_max, float* out_sum, int n_blocks)
{
  MD local_val = init_md();

  int tid = threadIdx.x;
  int idx = tid;

  for (; idx < n_blocks; idx += blockDim.x) {
    MD elem   = {bmax[idx], bsum[idx]};
    local_val = combine(local_val, elem);
  }

  // Warp Reduce
  local_val = warp_reduce_md(local_val);

  // Block Reduce
  static __shared__ float s_m[32];
  static __shared__ float s_d[32];
  int lane    = tid % 32;
  int warp_id = tid / 32;

  if (lane == 0) {
    s_m[warp_id] = local_val.m;
    s_d[warp_id] = local_val.d;
  }
  __syncthreads();

  if (tid < (blockDim.x / 32)) {
    MD block_val = {s_m[tid], s_d[tid]};
    block_val    = warp_reduce_md(block_val);
    if (tid == 0) {
      out_max[0] = block_val.m;
      out_sum[0] = block_val.d;
    }
  }
}

// Pass 3: softmax
__global__ void softmax_apply_kernel(
  const float* input, float* output, int N, const float* global_max, const float* global_sum)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float m = global_max[0];
  float s = global_sum[0];

  // Grid-Stride Loop
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < N; i += stride) {
    output[i] = expf(input[i] - m) / s;
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N)
{
#ifndef __ONLINE_SOFTMAX__
  int threadsPerBlock = 256;
  int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

  float* d_bmax = nullptr;
  float* d_max  = nullptr;
  float* d_bsum = nullptr;
  float* d_sum  = nullptr;
  cudaMalloc(&d_bmax, blocksPerGrid * sizeof(float));
  cudaMalloc(&d_max, sizeof(float));
  cudaMalloc(&d_bsum, blocksPerGrid * sizeof(float));
  cudaMalloc(&d_sum, sizeof(float));

  // 1. find the max value in each block: 1-thread --> 1--elem
  block_max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_bmax, N);
  // 2. find the max value from the max values of all blocks: 1-thread --> 1+ elem
  reduce_bmax_kernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
    d_bmax, d_max, blocksPerGrid);

  // Same logics as findmax
  // 3. compute sum for all values in each block: 1-thread --> 1--elem
  block_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_max, d_bsum, N);
  // 4. compute sum for all the sums from all blocks: 1-thread --> 1+ elem
  reduce_bsum_kernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
    d_bsum, d_sum, blocksPerGrid);

  softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, d_max, d_sum);
  cudaFree(d_bmax);
  cudaFree(d_max);
  cudaFree(d_bsum);
  cudaFree(d_sum);
  cudaDeviceSynchronize();
#else
  int threadsPerBlock = 256;
  int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
  if (blocksPerGrid > 2048) blocksPerGrid = 2048;

  float* d_bmax = nullptr;
  float* d_bsum = nullptr;
  float* d_max  = nullptr;
  float* d_sum  = nullptr;

  cudaMalloc(&d_bmax, blocksPerGrid * sizeof(float));
  cudaMalloc(&d_bsum, blocksPerGrid * sizeof(float));
  cudaMalloc(&d_max, sizeof(float));
  cudaMalloc(&d_sum, sizeof(float));

  // 1. Pass 1: Online Reduce
  online_reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_bmax, d_bsum, N);

  // 2. Pass 2: Reduce Block Results to Global Result
  global_reduce_kernel<<<1, 256>>>(d_bmax, d_bsum, d_max, d_sum, blocksPerGrid);

  // 3. Pass 3: Apply Softmax
  softmax_apply_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, d_max, d_sum);

  cudaFree(d_bmax);
  cudaFree(d_bsum);
  cudaFree(d_max);
  cudaFree(d_sum);
  cudaDeviceSynchronize();
#endif
}
