#include <cuda_runtime.h>
#include <math.h>

//////////////////////////////////////////////////////////////////
////////////////////// 1. Before sm90 (Hopper) //////////////////////
//////////////////////////////////////////////////////////////////
__global__ void block_max_kernel(const float* input, float* bmax, int N)
{
  __shared__ float s_input[2048];

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
  __shared__ float s_input[2048];

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
  __shared__ float s_input[2048];

  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidx = threadIdx.x;
  float m  = max_val[0];
  float s  = sum_val[0];
  if (gidx < N) s_input[tidx] = expf(input[gidx] - m);
  __syncthreads();

  if (gidx < N) output[gidx] = s_input[tidx] / s;
}
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N)
{
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

  block_max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_bmax, N);
  reduce_bmax_kernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
    d_bmax, d_max, blocksPerGrid);
  block_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_max, d_bsum, N);
  reduce_bsum_kernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
    d_bsum, d_sum, blocksPerGrid);

  softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, d_max, d_sum);
  cudaFree(d_bmax);
  cudaFree(d_max);
  cudaFree(d_bsum);
  cudaFree(d_sum);
  cudaDeviceSynchronize();
}
