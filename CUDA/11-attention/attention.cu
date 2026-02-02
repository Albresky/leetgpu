#include <cuda_runtime.h>
#include <cmath>

template <int TILE_SIZE>
__global__ void mm_kernel(const float* A,
                          const float* B,
                          float* C,
                          const int M,
                          const int N,
                          const int K,
                          const float dk = 1.0f)
{
  // Tiled Matmul
  // Note: C(M,N) = A(M, K) * B(K, N)
  // Differing from dims in 02-matmul,
  // the Tile is square, this is the key to simplify thread usage.
  // Each thread computes 1 C-elem.
  constexpr int Tile = TILE_SIZE;
  __shared__ float s_A[Tile][Tile];
  __shared__ float s_B[Tile][Tile];

  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int gidy = blockIdx.y * blockDim.y + threadIdx.y;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;

  float sum = .0f;
  for (int k = 0; k < (K + Tile - 1) / Tile; ++k) {
    // G2S
    // A
    if (gidy < M && (k * Tile + tidx) < K)
      s_A[tidy][tidx] = A[gidy * K + k * Tile + tidx];
    else
      s_A[tidy][tidx] = .0f;
    // B
    if (gidx < N && (k * Tile + tidy) < K)
      s_B[tidy][tidx] = B[(k * Tile + tidy) * N + gidx];
    else
      s_B[tidy][tidx] = .0f;
    __syncthreads();

    // MM
    for (int kk = 0; kk < Tile; ++kk) {
      sum += s_A[tidy][kk] * s_B[kk][tidx];
    }
    __syncthreads();
  }

  if (gidy < M && gidx < N) C[gidy * N + gidx] = sum / dk;
}

template <int TILE_SIZE>
__global__ void mm_transposed_kernel(const float* A,
                                     const float* B,
                                     float* C,
                                     const int M,
                                     const int N,
                                     const int K,
                                     const float dk = 1.0f)
{
  // Tiled Matmul with transposed B
  // A: (M, K)
  // B: (N, K)
  // C: (M, N)
  // Note: C(M,N) = A(M,K) * B^T(K,N)
  constexpr int Tile = TILE_SIZE;
  __shared__ float s_A[Tile][Tile];  // MTile x KTile
  __shared__ float s_B[Tile][Tile];  // NTile x NTile

  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int gidy = blockIdx.y * blockDim.y + threadIdx.y;
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;

  float sum = .0f;
  for (int k = 0; k < (K + Tile - 1) / Tile; ++k) {
    // G2S
    // A
    if (gidy < M && (k * Tile + tidx) < K)
      s_A[tidy][tidx] = A[gidy * K + k * Tile + tidx];
    else
      s_A[tidy][tidx] = .0f;

    // B
    // load B^T
    // bidx -->N, tidy < TileN, tidx --> dk
    if (blockIdx.x * Tile + tidy < N && k * Tile + tidx < K)
      s_B[tidx][tidy] = B[(blockIdx.x * Tile + tidy) * K + k * Tile + tidx];
    else
      s_B[tidx][tidy] = .0f;

    __syncthreads();

    // MM
    // s_A[tidy][kk] -> A[m, k]
    // s_B[kk][tidx] -> B^T[k, n]
    for (int kk = 0; kk < Tile; ++kk) {
      sum += s_A[tidy][kk] * s_B[kk][tidx];
    }
    __syncthreads();
  }

  if (gidy < M && gidx < N) C[gidy * N + gidx] = sum / dk;
}

__global__ void bmax_kernel(const float* input, float* bmax, const int d)
{
  extern __shared__ float s_bmax[];

  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int bidy = blockIdx.y;
  int tidx = threadIdx.x;

  s_bmax[tidx] = gidx < d ? input[bidy * d + gidx] : -INFINITY;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tidx < s) s_bmax[tidx] = max(s_bmax[tidx], s_bmax[tidx + s]);
    __syncthreads();
  }

  if (tidx == 0) bmax[bidy * gridDim.x + blockIdx.x] = s_bmax[0];
}

__global__ void max_kernel(const float* bmax, float* maxv, const int nbmax)
{
  extern __shared__ float s_max[];

  int bidx = blockIdx.x;
  int tidx = threadIdx.x;

  float vmax = -INFINITY;
  for (int i = tidx; i < nbmax; i += blockDim.x)
    vmax = max(vmax, bmax[bidx * nbmax + i]);
  s_max[tidx] = vmax;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tidx < s) s_max[tidx] = max(s_max[tidx], s_max[tidx + s]);
    __syncthreads();
  }

  if (tidx == 0) maxv[bidx] = s_max[0];
}

__global__ void bsum_kernel(const float* input, float* bsum, const float* maxv, const int d)
{
  extern __shared__ float s_bsum[];

  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int bidy = blockIdx.y;
  int tidx = threadIdx.x;

  s_bsum[tidx] = gidx < d ? exp(input[bidy * d + gidx] - maxv[bidy]) : .0f;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tidx < s) s_bsum[tidx] += s_bsum[tidx + s];
    __syncthreads();
  }

  if (tidx == 0) bsum[bidy * gridDim.x + blockIdx.x] = s_bsum[0];
}

__global__ void sum_kernel(const float* bsum, float* sum, const int nbsum)
{
  extern __shared__ float s_sum[];

  int bidx   = blockIdx.x;
  int tidx   = threadIdx.x;
  float sumv = .0f;
  for (int i = tidx; i < nbsum; i += blockDim.x) {
    sumv += bsum[bidx * nbsum + i];
  }
  s_sum[tidx] = sumv;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tidx < s) s_sum[tidx] += s_sum[tidx + s];
    __syncthreads();
  }

  if (tidx == 0) sum[bidx] = s_sum[0];
}

__global__ void norm_kernel(
  const float* input, float* output, const float* maxv, const float* sumv, const int N)
{
  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int bidy = blockIdx.y;
  if (gidx < N) output[bidy * N + gidx] = exp(input[bidy * N + gidx] - maxv[bidy]) / sumv[bidy];
}

extern "C" void solve(/* M*dk */ const float* Q,
                      /* N*dk */ const float* K,
                      /* N*dv */ const float* V,
                      /* M*dv */ float* output,
                      const int M,
                      const int N,
                      const int d)
{
  constexpr int TILE_SIZE = 16;
  dim3 threadBlock(TILE_SIZE, TILE_SIZE);
  dim3 blockPerGrid((N + threadBlock.x - 1) / threadBlock.x,
                    (M + threadBlock.y - 1) / threadBlock.y);
  float* S      = nullptr;
  float* S_norm = nullptr;
  cudaMalloc(&S, M * N * sizeof(float));
  cudaMalloc(&S_norm, M * N * sizeof(float));

  // compute S = Q * K^T, scaled with sqrt(dk)
  mm_transposed_kernel<TILE_SIZE><<<blockPerGrid, threadBlock>>>(Q, K, S, M, N, d, sqrtf(d));

  // compute softmax(S)
  dim3 norm_threadBlock(256);
  dim3 norm_blockPerGrid((N + norm_threadBlock.x - 1) / norm_threadBlock.x, M);
  float* bmax = nullptr;
  float* max  = nullptr;
  float* bsum = nullptr;
  float* sum  = nullptr;
  cudaMalloc(&bmax, norm_blockPerGrid.x * norm_blockPerGrid.y * sizeof(float));
  cudaMalloc(&max, norm_blockPerGrid.y * sizeof(float));
  cudaMalloc(&bsum, norm_blockPerGrid.x * norm_blockPerGrid.y * sizeof(float));
  cudaMalloc(&sum, norm_blockPerGrid.y * sizeof(float));

  bmax_kernel<<<norm_blockPerGrid, norm_threadBlock, norm_threadBlock.x * sizeof(float)>>>(
    S, bmax, N);
  max_kernel<<<M, norm_threadBlock, norm_threadBlock.x * sizeof(float)>>>(
    bmax, max, norm_blockPerGrid.x);
  bsum_kernel<<<norm_blockPerGrid, norm_threadBlock, norm_threadBlock.x * sizeof(float)>>>(
    S, bsum, max, N);
  sum_kernel<<<M, norm_threadBlock, norm_threadBlock.x * sizeof(float)>>>(
    bsum, sum, norm_blockPerGrid.x);
  norm_kernel<<<norm_blockPerGrid, norm_threadBlock>>>(S, S_norm, max, sum, N);

  dim3 blockPerGrid_out((d + threadBlock.x - 1) / threadBlock.x,
                        (M + threadBlock.y - 1) / threadBlock.y);
  mm_kernel<TILE_SIZE><<<blockPerGrid_out, threadBlock>>>(S_norm, V, output, M, d, N);

  cudaFree(bmax);
  cudaFree(max);
  cudaFree(bsum);
  cudaFree(sum);
  cudaFree(S);
  cudaFree(S_norm);
}