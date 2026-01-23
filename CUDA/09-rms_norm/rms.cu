#include <cuda_runtime.h>

__global__ void bsum_kernel(const float* input, float* bsum, const int N)
{
  extern __shared__ float s_sum[];
  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidx = threadIdx.x;
  if (gidx < N)
    s_sum[tidx] = input[gidx];
  else
    s_sum[tidx] = 0.0f;

  __syncthreads();

  // compute square
  s_sum[tidx] *= s_sum[tidx];
  __syncthreads();

  // compute sum
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tidx < s) { s_sum[tidx] += s_sum[tidx + s]; }
    __syncthreads();
  }
  if (tidx == 0) { bsum[blockIdx.x] = s_sum[0]; }
}

__global__ void rms_kernel(float* bsum, float* rms, const int nsum, const int N, const float eps)
{
  extern __shared__ float s_sum[];
  int tidx = threadIdx.x;

  float sum = 0.0f;
  for (int i = tidx; i < nsum; i += blockDim.x) {
    sum += bsum[i];
  }
  s_sum[tidx] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tidx < s) s_sum[tidx] += s_sum[tidx + s];
    __syncthreads();
  }
  if (tidx == 0) *rms = sqrtf(s_sum[0] / N + eps);
}

__global__ void rms_norm_kernel(
  const float* input, const float* rms, float gamma, float beta, float* output, int N)
{
  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (gidx < N) {
    float x          = input[gidx];
    float x_norm     = x / rms[0];
    float y          = gamma * x_norm + beta;
    *(output + gidx) = y;
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N, float eps)
{
  dim3 threadBlock(256);
  dim3 blocksPerGrid((N + threadBlock.x - 1) / threadBlock.x);

  float* bsum = nullptr;
  float* rms  = nullptr;
  cudaMalloc(&bsum, blocksPerGrid.x * sizeof(float));
  cudaMalloc(&rms, 1 * sizeof(float));

  bsum_kernel<<<blocksPerGrid, threadBlock, threadBlock.x * sizeof(float)>>>(input, bsum, N);
  rms_kernel<<<1, threadBlock, threadBlock.x * sizeof(float)>>>(bsum, rms, blocksPerGrid.x, N, eps);
  rms_norm_kernel<<<blocksPerGrid, threadBlock>>>(input, rms, gamma, beta, output, N);

  cudaFree(bsum);
  cudaFree(rms);
}
