#include <cuda_runtime.h>

#define __K0

/* 0. naive impl */
#ifdef __K0
__global__ void matrix_multiplication_kernel(const float *A, const float *B,
                                             float *C, int M, int N, int K) {
  int bidx = blockIdx.x * blockDim.x;
  int bidy = blockIdx.y * blockDim.y;
  int tidx = bidx + threadIdx.x;
  int tidy = bidy + threadIdx.y;

  if (tidy < M && tidx < K) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
      sum += A[tidy * N + i] * B[i * K + tidx]; // For matrix B, memory access are coalesced.
    }
    C[tidy * K + tidx] = sum;
  }
}
#endif

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int M, int N,
                      int K) {
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M,
                                                                   N, K);
  cudaDeviceSynchronize();
}
