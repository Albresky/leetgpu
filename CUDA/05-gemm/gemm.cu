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

  if (tidy < M && tidx < K) {
    half sum = 0.0;
    for (int i = 0; i < N; ++i) {
      sum += A[tidy * N + i] * B[i * K + tidx];  // For matrix B, memory access are coalesced.
    }
    __syncthreads();
    sum *= alpha;
    C[tidy * K + tidx] *= beta;
    C[tidy * K + tidx] += sum;
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
  __shared__ float s_A[MTile][NTile];
  __shared__ float s_B[NTile][KTile];

  int bidx = blockIdx.x * blockDim.x;
  int bidy = blockIdx.y * blockDim.y;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int idx  = bidx + tidx;
  int idy  = bidy + tidy;

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
  if (idy < M && idx < K) C[idy * K + idx] = sum;
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
  __shared__ float s_A[MTile][NTile];
  __shared__ float s_B[NTile][KTile];
  float s_C[BLK][BLK];

  int bidx = blockIdx.x * blockDim.x;
  int bidy = blockIdx.y * blockDim.y;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int idx  = bidx + tidx;
  int idy  = bidy + tidy;

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
            s_C[bm][bk] += s_A[tidy * BLK + bm][i * BLK + bn] * s_B[i * BLK + bn][tidx * BLK + bk];
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

/* 3. using 4x4 WMMA and smem for tiling */
/**
 * WARNING: This impl WON'T PASS in LeetGPU OJ due to the low precision of TF32.
 * In PTX programming model, we usually use `K` as the reducing dim, however,
 * in questions of LeetGPU I find most reducing dimension is dim `N`.
 */
#ifdef __K3
#include <mma.h>
using namespace nvcuda;

__global__ void gemm_kernel(
  const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta)
{
  const int BM = 32;
  const int BN = 16;
  const int BK = 64;

  const int WMMA_M   = 16;
  const int WMMA_N   = 16;
  const int WMMA_K   = 8;
  const int WARP_DIM = 32;

  // 2x4 warps
  const int WARPS_N = 4;

  // 1D thread id
  int tid    = threadIdx.y * blockDim.x + threadIdx.x;
  int warpId = tid / WARP_DIM;

  int warpRow = warpId / WARPS_N;
  int warpCol = warpId % WARPS_N;

  int blockRow = blockIdx.y * BM;
  int blockCol = blockIdx.x * BK;

  __shared__ float s_A[BM][BN];
  __shared__ float s_B[BN][BK];
  __shared__ float s_C[BM][BK];

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major>
    a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major>
    b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int n_idx = 0; n_idx < N; n_idx += BN) {
// G2S
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      int idx = tid + i * 256;
      if (idx < BM * BN) {
        int r = idx / BN;
        int c = idx % BN;
        if (blockRow + r < M && n_idx + c < N) {
          s_A[r][c] = A[(blockRow + r) * N + (n_idx + c)];
        } else {
          s_A[r][c] = 0.0f;
        }
      }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int idx = tid + i * 256;
      if (idx < BN * BK) {
        int r = idx / BK;
        int c = idx % BK;
        if (n_idx + r < N && blockCol + c < K) {
          s_B[r][c] = B[(n_idx + r) * K + (blockCol + c)];
        } else {
          s_B[r][c] = 0.0f;
        }
      }
    }
    __syncthreads();

    // wmma
    for (int k_step = 0; k_step < BN; k_step += WMMA_K) {
      wmma::load_matrix_sync(a_frag, &s_A[warpRow * 16][k_step], BN);
      wmma::load_matrix_sync(b_frag, &s_B[k_step][warpCol * 16], BK);

      // MMA
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    __syncthreads();
  }
  wmma::store_matrix_sync(&s_C[warpRow * 16][warpCol * 16], c_frag, BK, wmma::mem_row_major);

  __syncthreads();

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int idx = tid + i * 256;
    if (idx < BM * BK) {
      int r = idx / BK;
      int c = idx % BK;
      if (blockRow + r < M && blockCol + c < K) {
        C[(blockRow + r) * K + (blockCol + c)] = s_C[r][c];
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
  dim3 blocksPerGrid((K + threadsPerBlock.x * BLK - 1) / (threadsPerBlock.x * BLK),
                     (M + threadsPerBlock.y * BLK - 1) / (threadsPerBlock.y * BLK));
#elif defined(__K3)
  dim3 blocksPerGrid((K + 64 - 1) / 64, (M + 32 - 1) / 32);
#else
  dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
#endif
  gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
  cudaDeviceSynchronize();
}
