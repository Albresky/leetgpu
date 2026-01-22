#include <cuda_runtime.h>

// 1. Mini-batch Mean
__global__ void MBM_kernel(const float* input, float* mean, const int N, const int C)
{
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tidx < C) {
    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
      sum += *(input + n * C + tidx);
    }
    *(mean + tidx) = sum / N;
  }
}

// 2. Mini-batch Variance, standard
__global__ void MBV_kernel(
  const float* input, const float* mean, float* variance, const int N, const int C, const float eps)
{
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tidx < C) {
    float var = 0.0f;
    float m   = *(mean + tidx);
    for (int n = 0; n < N; ++n) {
      float minus = *(input + n * C + tidx) - m;
      var += minus * minus;
    }
    var /= N;
    *(variance + tidx) = sqrtf(var + eps);
  }
}

// 3. Normalization, sacle, and shift
__global__ void norm_scale_shift_kernel(const float* input,
                                        float* mean,
                                        float* variance,
                                        const float* gamma,
                                        const float* beta,
                                        float* output,
                                        const int N,
                                        const int C)
{
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  if (tidx < C && tidy < N) {
    float x                     = *(input + tidy * C + tidx);
    float m                     = *(mean + tidx);
    float v                     = *(variance + tidx);
    float g                     = *(gamma + tidx);
    float b                     = *(beta + tidx);
    float x_new                 = (x - m) / v;
    float y                     = g * x_new + b;
    *(output + tidy * C + tidx) = y;
  }
}

// input, gamma, beta, output are device pointers
extern "C" void solve(
  const float* input, const float* gamma, const float* beta, float* output, int N, int C, float eps)
{
  dim3 threadsPerBlock1D(256);
  dim3 blocksPerGrid1D((C + threadsPerBlock1D.x - 1) / threadsPerBlock1D.x);
  dim3 threadsPerBlock2D(32, 32);
  dim3 blocksPerGrid2D((C + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
                       (N + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y);
  float* mean     = nullptr;
  float* variance = nullptr;
  cudaMalloc(&mean, C * sizeof(float));
  cudaMalloc(&variance, C * sizeof(float));
  MBM_kernel<<<blocksPerGrid1D, threadsPerBlock1D>>>(input, mean, N, C);
  MBV_kernel<<<blocksPerGrid1D, threadsPerBlock1D>>>(input, mean, variance, N, C, eps);
  norm_scale_shift_kernel<<<blocksPerGrid2D, threadsPerBlock2D>>>(
    input, mean, variance, gamma, beta, output, N, C);
}
