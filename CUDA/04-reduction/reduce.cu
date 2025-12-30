#include <cuda_runtime.h>

#define __K2

/* 0. naive impl */
/**
 * Lose precision and fails LeetGPU OJ, while using double to keep precision
 * slows down the program. And, Using Kahan algo won't help.
*/
#ifdef __K0
__global__ void reduce_kernel(const float *input, float *output, int N) {
  float sum = 0;
  float c = 0;
  for (int i = 0; i < N; ++i) {
    // sum += input[i];
    float y = input[i] - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  *output = sum;
}
#endif

/* 1. parallel reduction with smem */
#ifdef __K1
__global__ void reduce_kernel(const float *input, float *output, int N) {
  __shared__ float smem[1024];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // G2S
  if (gid < N) {
    smem[tid] = input[gid];
  } else {
    smem[tid] = 0.0f;
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] += smem[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(output, smem[0]);
  }
}
#endif

/* 2. parallel reduction using shuffle */
#ifdef __K2
#define FULL_MASK 0xffffffff
__global__ void reduce_kernel(const float *input, float *output, int N) {
  __shared__ float smem[32];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // G2S
  if (gid < N) {
    smem[tid] = input[gid];
  } else {
    smem[tid] = 0.0f;
  }
  __syncthreads();

//   auto mask= __ballot_sync(FULL_MASK,tid < 32);
  float sum = smem[tid];;
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    sum += __shfl_down_sync(FULL_MASK,sum,s);
  }

  if (tid == 0) {
    atomicAdd(output, sum);
  }
}
#endif

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N) {
#ifdef __K0
  dim3 threadsPerBlock(1);
  dim3 blocksPerGrid(1);
#elif defined(__K1)
  dim3 threadsPerBlock(1024);
  dim3 blocksPerGrid((N + 1024 - 1) / 1024);
#elif defined(__K2)
  dim3 threadsPerBlock(32);
  dim3 blocksPerGrid((N + 32 - 1) / 32);
#endif
  cudaMemset(output, 0, sizeof(float));
  reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
  cudaDeviceSynchronize();
}
