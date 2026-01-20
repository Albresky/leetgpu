#include <cuda_runtime.h>

#define __K1

/* 1. naive impl. */
#ifdef __K0
__global__ void convolution_1d_kernel(
  const float* input, const float* kernel, float* output, int input_size, int kernel_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= input_size - kernel_size + 1) return;

  float sum = 0;
  for (int i = 0; i < kernel_size; ++i) {
    sum += input[idx + i] * kernel[i];
  }
  output[idx] = sum;
}
#endif

/* 2. using smem for kernel*/
#ifdef __K1
__global__ void convolution_1d_kernel(
  const float* input, const float* kernel, float* output, int input_size, int kernel_size)
{
  // goal: s_i >> s_k
  const int blk_dim = 1024;
  const int max_k   = 2048;

  __shared__ float s_i[blk_dim + max_k];
  // min(kernel_size, blockDim), max(kernel_size) == 2048 (for testbench)
  // SMEM per SM is enough for holding full kernel
  __shared__ float s_k[2048];
  // __shared__ float s_o[512];

  int idx         = blockIdx.x * blockDim.x + threadIdx.x;
  int tidx        = threadIdx.x;
  int output_size = input_size - kernel_size + 1;

  // G2S: input window, full kernel
  int bidx     = blockIdx.x * blockDim.x;
  int tile_len = blockDim.x + kernel_size - 1;
  for (int i = tidx; i < tile_len; i += blockDim.x) {
    int gidx = bidx + i;
    s_i[i]   = (gidx < input_size) ? input[gidx] : 0.0f;
  }
  const int k_per_thd = (kernel_size + blk_dim - 1) / blk_dim;
  for (int j = 0; j < k_per_thd && tidx * k_per_thd + j < kernel_size; ++j) {
    s_k[tidx * k_per_thd + j] = kernel[tidx * k_per_thd + j];
  }
  __syncthreads();

  if (idx < output_size) {
    float sum = 0;
    for (int k = 0; k < kernel_size; ++k) {
      sum += s_i[tidx + k] * s_k[k];
    }
    output[idx] = sum;
  }
}
#endif

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(
  const float* input, const float* kernel, float* output, int input_size, int kernel_size)
{
  int output_size     = input_size - kernel_size + 1;
  int threadsPerBlock = 1024;
  int blocksPerGrid   = (output_size + threadsPerBlock - 1) / threadsPerBlock;

  convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    input, kernel, output, input_size, kernel_size);
  cudaDeviceSynchronize();
}
