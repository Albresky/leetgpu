#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define __K0

/* 0. naive impl */
#ifdef __K0
__global__ void gemm_kernel(
  const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta)
{
  int bidx = blockIdx.x * blockDim.x;
  int bidy = blockIdx.y * blockDim.y;
  int tidx = bidx + threadIdx.x;
  int tidy = bidy + threadIdx.y;

  if (tidy < M && tidx < N) {
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
      sum += (float)A[tidy * K + i] * (float)B[i * N + tidx];
    }
    float c_val = (float)C[tidy * N + tidx];
    C[tidy * N + tidx] = (half)(alpha * sum + beta * c_val);
  }
}
#endif

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(
  const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta)
{
  constexpr int threadsPB = 16;
  dim3 threadsPerBlock(threadsPB, threadsPB);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
  gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
  cudaDeviceSynchronize();
}
