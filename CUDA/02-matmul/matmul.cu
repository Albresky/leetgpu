#include <cuda_runtime.h>

#define __K2

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
  // each thread compute 1 elem
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

/* 2. using 4x4 smem for tiling */
#ifdef __K2
__global__ void matrix_multiplication_kernel(const float *A, const float *B,
                                             float *C, int M, int N, int K) {
  // each thread compute a 4x4 tile
  constexpr int BLK = 4;
  constexpr int MTile = 16 * BLK;
  constexpr int NTile = 16 * BLK;
  constexpr int KTile = 16 * BLK;
  __shared__ float s_A[MTile][NTile];
  __shared__ float s_B[NTile][KTile];
  float s_C[BLK][BLK];

  int bidx = blockIdx.x * blockDim.x;
  int bidy = blockIdx.y * blockDim.y;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int idx = bidx + tidx;
  int idy = bidy + tidy;

  // init smemC
  for (int bm = 0; bm < BLK; ++bm)
    for (int bk = 0; bk < BLK; ++bk)
      s_C[bm][bk] = 0;

  for (int ni = 0; ni < (N + NTile - 1) / NTile; ++ni) {
    // G2S
    for (int bm = 0; bm < BLK; ++bm) {
      for (int bn = 0; bn < BLK; ++bn) {
        if (idy * BLK + bm < M && (ni * NTile + tidx * BLK + bn) < N) {
          s_A[tidy * BLK + bm][tidx * BLK + bn] =
              A[(idy * BLK + bm) * N + ni * NTile + tidx * BLK + bn];
        } else {
          s_A[tidy * BLK + bm][tidx * BLK + bn] = 0.0f;
        }
      }
    }
    for (int bn = 0; bn < BLK; ++bn) {
      for (int bk = 0; bk < BLK; ++bk) {
        if (idx * BLK + bk < K && (ni * NTile + tidy * BLK + bn) < N) {
          s_B[tidy * BLK + bn][tidx * BLK + bk] =
              B[(ni * NTile + tidy * BLK + bn) * K + idx * BLK + bk];
        } else {
          s_B[tidy * BLK + bn][tidx * BLK + bk] = 0.0f;
        }
      }
    }
    __syncthreads();

    // gemm
    for (int i = 0; i < NTile / BLK; ++i) {
      // per thread perform a gemm
      for (int bm = 0; bm < BLK; ++bm)
        for (int bk = 0; bk < BLK; ++bk)
          for (int bn = 0; bn < BLK; ++bn)
            s_C[bm][bk] += s_A[tidy * BLK + bm][i * BLK + bn] *
                           s_B[i * BLK + bn][tidx * BLK + bk];
    }
    __syncthreads();
  }

  // S2G
  for (int bm = 0; bm < BLK; ++bm) {
    for (int bk = 0; bk < BLK; ++bk) {
      if ((idy * BLK + bm) < M && (idx * BLK + bk) < K) {
        C[(idy * BLK + bm) * K + idx * BLK + bk] = s_C[bm][bk];
      }
    }
  }
}
#endif

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int M, int N,
                      int K) {
  constexpr int threadsPB = 16;
  dim3 threadsPerBlock(threadsPB, threadsPB);
#if defined(__K2)
  constexpr int BLK = 4;
  dim3 blocksPerGrid(
      (K + threadsPerBlock.x * BLK - 1) / (threadsPerBlock.x * BLK),
      (M + threadsPerBlock.y * BLK - 1) / (threadsPerBlock.y * BLK));
#else
  dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
#endif
  matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M,
                                                                   N, K);
  cudaDeviceSynchronize();
}
