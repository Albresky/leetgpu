#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define __K2

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
    float c_val        = (float)C[tidy * N + tidx];
    C[tidy * N + tidx] = (half)(alpha * sum + beta * c_val);
  }
}
#endif

/* 1. using smem for tiling */
#ifdef __K1
__global__ void gemm_kernel(
  const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta)
{
  // each thread compute 1 elem
  constexpr int MTile = 16;
  constexpr int NTile = 16;
  constexpr int KTile = 16;
  __shared__ float s_A[MTile][KTile];
  __shared__ float s_B[KTile][NTile];

  int bidx = blockIdx.x * blockDim.x;
  int bidy = blockIdx.y * blockDim.y;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int idx  = bidx + tidx;
  int idy  = bidy + tidy;

  float sum = 0.0f;
  for (int ki = 0; ki < (K + KTile - 1) / KTile; ++ki) {
    // G2S
    if (idy < M && (ki * KTile + tidx) < K) {
      s_A[tidy][tidx] = A[idy * K + ki * KTile + tidx];
    } else {
      s_A[tidy][tidx] = 0.0f;
    }
    if (idx < N && (ki * KTile + tidy) < K) {
      s_B[tidy][tidx] = B[(ki * KTile + tidy) * N + idx];
    } else {
      s_B[tidy][tidx] = 0.0f;
    }
    __syncthreads();

    // gemm
    for (int i = 0; i < KTile; ++i) {
      sum += s_A[tidy][i] * s_B[i][tidx];
    }
    __syncthreads();
  }

  // S2G
  if (idy < M && idx < N) {
    float c          = C[idy * N + idx];
    c                = alpha * sum + beta * c;
    C[idy * N + idx] = c;
  }
}
#endif

/* 2. using 4x4 smem for tiling */
#ifdef __K2
__global__ void gemm_kernel(
  const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta)
{
  // each thread compute a 4x4 tile
  constexpr int BLK   = 4;
  constexpr int MTile = 16 * BLK;
  constexpr int NTile = 16 * BLK;
  constexpr int KTile = 16 * BLK;
  __shared__ float s_A[MTile][KTile];
  __shared__ float s_B[KTile][NTile];
  float s_C[BLK][BLK];

  int bidx = blockIdx.x * blockDim.x;
  int bidy = blockIdx.y * blockDim.y;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int idx  = bidx + tidx;
  int idy  = bidy + tidy;

  // init smemC
  for (int bm = 0; bm < BLK; ++bm)
    for (int bn = 0; bn < BLK; ++bn)
      s_C[bm][bn] = 0;

  for (int ki = 0; ki < (K + KTile - 1) / KTile; ++ki) {
    // G2S
    for (int bm = 0; bm < BLK; ++bm) {
      for (int bk = 0; bk < BLK; ++bk) {
        if (idy * BLK + bm < M && (ki * KTile + tidx * BLK + bk) < K) {
          s_A[tidy * BLK + bm][tidx * BLK + bk] =
            A[(idy * BLK + bm) * K + ki * KTile + tidx * BLK + bk];
        } else {
          s_A[tidy * BLK + bm][tidx * BLK + bk] = 0.0f;
        }
      }
    }
    for (int bk = 0; bk < BLK; ++bk) {
      for (int bn = 0; bn < BLK; ++bn) {
        if (idx * BLK + bn < N && (ki * KTile + tidy * BLK + bk) < K) {
          s_B[tidy * BLK + bk][tidx * BLK + bn] =
            B[(ki * KTile + tidy * BLK + bk) * N + idx * BLK + bn];
        } else {
          s_B[tidy * BLK + bk][tidx * BLK + bn] = 0.0f;
        }
      }
    }
    __syncthreads();

    // gemm
    for (int i = 0; i < KTile / BLK; ++i) {
      // per thread perform a gemm
      for (int bm = 0; bm < BLK; ++bm)
        for (int bn = 0; bn < BLK; ++bn)
          for (int bk = 0; bk < BLK; ++bk)
            s_C[bm][bn] += s_A[tidy * BLK + bm][i * BLK + bk] * s_B[i * BLK + bk][tidx * BLK + bn];
    }
    __syncthreads();
  }

  // S2G
  for (int bm = 0; bm < BLK; ++bm) {
    for (int bn = 0; bn < BLK; ++bn) {
      if ((idy * BLK + bm) < M && (idx * BLK + bn) < N) {
        float c                                  = C[(idy * BLK + bm) * N + idx * BLK + bn];
        c                                        = alpha * s_C[bm][bn] + beta * c;
        C[(idy * BLK + bm) * N + idx * BLK + bn] = c;
      }
    }
  }
}
#endif

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(
  const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta)
{
  constexpr int threadsPB = 16;
  dim3 threadsPerBlock(threadsPB, threadsPB);
#if defined(__K2)
  constexpr int BLK = 4;
  dim3 blocksPerGrid((N + threadsPerBlock.x * BLK - 1) / (threadsPerBlock.x * BLK),
                     (M + threadsPerBlock.y * BLK - 1) / (threadsPerBlock.y * BLK));
#else
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
#endif
  gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
  cudaDeviceSynchronize();
}
