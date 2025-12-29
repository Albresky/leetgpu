#include <cuda_runtime.h>

#define __K1

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
      sum += A[tidy * N + i] *
             B[i * K + tidx]; // For matrix B, memory access are coalesced.
    }
    C[tidy * K + tidx] = sum;
  }
}
#endif

/* 1. using smem for tiling */
#ifdef __K1
__global__ void matrix_multiplication_kernel(const float *A, const float *B,
                                             float *C, int M, int N, int K) {
  constexpr int MTile = 16;
  constexpr int NTile = 16;
  constexpr int KTile = 16;
  __shared__ float s_A[MTile][NTile];
  __shared__ float s_B[NTile][KTile];

  int bidx = blockIdx.x * blockDim.x;
  int bidy = blockIdx.y * blockDim.y;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int idx = bidx + tidx;
  int idy = bidy + tidy;

  float sum = 0.0f;
  for (int ni = 0; ni < (N + NTile - 1) / NTile; ++ni) {
    // G2S
    if (idy < M && (ni * NTile + tidx) < N) {
      s_A[tidy][tidx] = A[idy * N + ni * NTile + tidx];
    } else {
      s_A[tidy][tidx] = 0.0f;
    }
    if (idx < K && (ni * NTile + tidy) < N) {
      s_B[tidy][tidx] = B[(ni * NTile + tidy) * K + idx];
    } else {
      s_B[tidy][tidx] = 0.0f;
    }
    __syncthreads();

    // gemm
    for (int i = 0; i < NTile; ++i) {
      sum += s_A[tidy][i] * s_B[i][tidx];
    }
    __syncthreads();
  }

  // S2G
  if (idy < M && idx < K)
    C[idy * K + idx] = sum;
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
