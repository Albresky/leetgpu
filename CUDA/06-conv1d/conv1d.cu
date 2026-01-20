#include <cuda_runtime.h>

/* 1. naive impl. */
__global__ void convolution_1d_kernel(
  const float* input, const float* kernel, float* output, int input_size, int kernel_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= input_size - kernel_size + 1) return;

  float sum = 0;
  for (int i = 0; i < kernel_size; ++i) {
    sum += input[idx+i] * kernel[i];
  }
  __syncthreads();
  output[idx] = sum;
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(
  const float* input, const float* kernel, float* output, int input_size, int kernel_size)
{
  int output_size     = input_size - kernel_size + 1;
  int threadsPerBlock = 256;
  int blocksPerGrid   = (output_size + threadsPerBlock - 1) / threadsPerBlock;

  convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    input, kernel, output, input_size, kernel_size);
  cudaDeviceSynchronize();
}
