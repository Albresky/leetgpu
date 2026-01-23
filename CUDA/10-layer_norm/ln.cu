#include <cuda_runtime.h>
#include <cmath>

__global__ void bsum_kernel(const float* input, float* bsum, float N)
{
  extern __shared__ float s_sum[];
  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidx = threadIdx.x;

  if (gidx < N) {
    s_sum[tidx] = input[gidx];
  } else {
    s_sum[tidx] = .0f;
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tidx < s) s_sum[tidx] += s_sum[tidx + s];
    __syncthreads();
  }

  if (tidx == 0) bsum[blockIdx.x] = s_sum[0];
}

__global__ void sum_kernel(const float* bsum, float* mean, float nsum, float N)
{
  extern __shared__ float s_sum[];

  int tidx  = threadIdx.x;
  float sum = .0f;
  for (int i = tidx; i < nsum; i += blockDim.x) {
    sum += bsum[i];
  }
  s_sum[tidx] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tidx < s) s_sum[tidx] += s_sum[tidx + s];
    __syncthreads();
  }
  if (tidx == 0) *mean = s_sum[0] / N;
}

__global__ void bvar_kernel(const float* input, float* bvar, const float* mean, const int N)
{
  extern __shared__ float s_var[];
  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidx = threadIdx.x;

  if (gidx < N) {
    float val   = input[gidx];
    s_var[tidx] = (val - *mean) * (val - *mean);
  } else
    s_var[tidx] = 0;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tidx < s) s_var[tidx] += s_var[tidx + s];
    __syncthreads();
  }

  if (tidx == 0) bvar[blockIdx.x] = s_var[0];
}

__global__ void var_kernel(
  float* bvar, float* std_var, const int nsum, const float eps, const int N)
{
  extern __shared__ float s_var[];

  int tidx  = threadIdx.x;
  float sum = .0f;
  for (int i = tidx; i < nsum; i += blockDim.x) {
    sum += bvar[i];
  }
  s_var[tidx] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tidx < s) s_var[tidx] += s_var[tidx + s];
    __syncthreads();
  }

  if (tidx == 0) *std_var = sqrtf(s_var[0] / N + eps);
}

__global__ void norm_kernel(const float* input,
                            float* output,
                            const float* mean,
                            const float* std_var,
                            const float gamma,
                            const float beta,
                            const int N)
{
  extern __shared__ float s_input[];

  int gidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (gidx < N) {
    float x_norm     = (input[gidx] - *mean) / (*std_var);
    float y          = gamma * x_norm + beta;
    *(output + gidx) = y;
  }
}

extern "C" void solve(const float* input,
                      float* output,
                      const float gamma,
                      const float beta,
                      const float eps,
                      const float N)
{
  dim3 threadBlock(512);
  dim3 blocksPerGrid((N + threadBlock.x - 1) / threadBlock.x);

  float* bsum    = nullptr;
  float* mean    = nullptr;
  float* bvar    = nullptr;
  float* std_var = nullptr;
  cudaMalloc(&bsum, blocksPerGrid.x * sizeof(float));
  cudaMalloc(&mean, 1 * sizeof(float));
  cudaMalloc(&bvar, blocksPerGrid.x * sizeof(float));
  cudaMalloc(&std_var, 1 * sizeof(float));

  bsum_kernel<<<blocksPerGrid, threadBlock, threadBlock.x * sizeof(float)>>>(input, bsum, N);
  sum_kernel<<<1, threadBlock, threadBlock.x * sizeof(float)>>>(bsum, mean, blocksPerGrid.x, N);
  bvar_kernel<<<blocksPerGrid, threadBlock, threadBlock.x * sizeof(float)>>>(input, bvar, mean, N);
  var_kernel<<<1, threadBlock, threadBlock.x * sizeof(float)>>>(
    bvar, std_var, blocksPerGrid.x, eps, N);
  norm_kernel<<<blocksPerGrid, threadBlock, threadBlock.x * sizeof(float)>>>(
    input, output, mean, std_var, gamma, beta, N);

  cudaFree(bsum);
  cudaFree(mean);
  cudaFree(bvar);
  cudaFree(std_var);
}