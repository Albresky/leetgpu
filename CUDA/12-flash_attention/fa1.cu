#include <cuda_runtime.h>
#include <cmath>

/**
 * Naive implementation of FlashAttention V1. Reference: http://arxiv.org/abs/2205.14135
 * Note: 1-Pass Attention
 */

template <int Br, int Bc, int d>
__global__ void fa1_kernel(/* Mxd */ const float* Q,
                           /* Nxd */ const float* K,
                           /* Nxd */ const float* V,
                           /* Mxd */ float* O,
                           const int M,
                           const int N,
                           const float scale)
{
  constexpr int TileQ = Br;
  constexpr int TileK = Bc;

  __shared__ float s_Q[TileQ][d];
  __shared__ float s_K[TileK][d];
  __shared__ float s_V[TileK][d];

  int tidx   = threadIdx.x;
  int tidy   = threadIdx.y;
  int rowIdx = blockIdx.y * TileQ + tidy;

  // local registers for online softmax, output
  float acc_o[d] = {.0f};
  float mi       = -INFINITY;
  float di       = .0f;

  for (int i = 0; i < d; ++i)
    acc_o[i] = 0;

  // 1. G2S: load Q
  for (int c = tidx; c < d; c += blockDim.x) {
    if (rowIdx < M) {
      s_Q[tidy][c] = Q[rowIdx * d + c];
    } else {
      s_Q[tidy][c] = .0f;
    }
  }
  __syncthreads();

  // 2. outer&inner loop tiles of K, V
  for (int t = 0; t < (N + TileK - 1) / TileK; ++t) {
    // a) G2S: load K, V
    for (int r = tidy; r < TileK; r += blockDim.y) {
      int gr = t * TileK + r;
      for (int c = tidx; c < d; c += blockDim.x) {
        if (gr < N) {
          s_K[r][c] = K[gr * d + c];
          s_V[r][c] = V[gr * d + c];
        } else {
          s_K[r][c] = .0f;
          s_V[r][c] = .0f;
        }
      }
    }
    __syncthreads();

    // b) MM
    if (rowIdx < M) {
      float local_scores[TileK];
      float local_max = -INFINITY;

      for (int k = 0; k < TileK; ++k) {
        float score = .0f;
        for (int i = 0; i < d; ++i)
          score += s_Q[tidy][i] * s_K[k][i];
        score *= scale;

        if (t * TileK + k >= N) score = -INFINITY;
        local_scores[k] = score;
        local_max       = fmaxf(local_max, score);
      }

      // online softmax
      float m_cur = fmax(mi, local_max);
      float alpha = expf(mi - m_cur);
      float beta  = expf(local_max - m_cur);

      float local_sum = .0f;
      for (int k = 0; k < TileK; ++k) {
        local_scores[k] = expf(local_scores[k] - local_max);
        local_sum += local_scores[k];
      }

      di = alpha * di + beta * local_sum;
      mi = m_cur;

      // MM: update O
      for (int x = tidx; x < d; x += blockDim.x) {
        float pv = .0f;
        for (int k = 0; k < TileK; ++k) {
          pv += local_scores[k] * s_V[k][x];
        }
        acc_o[x] = alpha * acc_o[x] + beta * pv;
      }
    }
    __syncthreads();
  }

  // 3. norm, S2G
  if (rowIdx < M) {
    for (int x = tidx; x < d; x += blockDim.x) {
      O[rowIdx * d + x] = acc_o[x] / di;
    }
  }
}

extern "C" void solve(/* Mxd */ const float* Q,
                      /* Nxd */ const float* K,
                      /* Nxd */ const float* V,
                      /* Mxd */ float* O,
                      const int M,
                      const int N,
                      const int d)
{
  constexpr int Br = 16;
  constexpr int Bc = 16;
  constexpr int D  = 128;

  dim3 threadBlock(32, Br);
  dim3 blockPerGrid(1, (M + Br - 1) / Br);

  fa1_kernel<Br, Bc, D><<<blockPerGrid, threadBlock>>>(Q, K, V, O, M, N, 1.0f / sqrtf((float)D));
}