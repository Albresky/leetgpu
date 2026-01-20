#include <cuda_runtime.h>
#include<iostream>

#define __K3

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

/* 3. two-stage reduction with single atomicAdd */
#ifdef __K3
__global__ void reduce_stage1(const float *input, float *partials, int N) {
  extern __shared__ float smem[];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  int stride = blockDim.x * gridDim.x;

  float sum = 0.0f;
  for (int i = gid; i < N; i += stride) {
    sum += input[i];
  }
  smem[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] += smem[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partials[blockIdx.x] = smem[0];
  }
}

__global__ void reduce_stage2(const float *partials, float *output, int numPartials) {
  extern __shared__ float smem[];

  int tid = threadIdx.x;
  float sum = 0.0f;

  for (int i = tid; i < numPartials; i += blockDim.x) {
    sum += partials[i];
  }
  smem[tid] = sum;
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
#elif defined(__K3)
  dim3 threadsPerBlock(256);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
#endif
  cudaMemset(output, 0, sizeof(float));
#if defined(__K0) || defined(__K1) || defined(__K2)
  reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
  cudaDeviceSynchronize();
#elif defined(__K3)
  float *partials = nullptr;
  int numPartials = blocksPerGrid.x;
  cudaMalloc(&partials, numPartials * sizeof(float));
  reduce_stage1<<<blocksPerGrid, threadsPerBlock, threadsPerBlock.x * sizeof(float)>>>(
      input, partials, N);
  reduce_stage2<<<1, threadsPerBlock, threadsPerBlock.x * sizeof(float)>>>(
      partials, output, numPartials);
  cudaDeviceSynchronize();
  cudaFree(partials);
#endif
}
