#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "../common.h"

extern "C" void solve(
  const float* Q, const float* K, const float* V, float* output, int M, int N, int d);

class AttentionProblem : public Problem {
 private:
  int M, N, dk, dv, d;
  int len_Q, len_K, len_V, len_o;
  size_t size_Q, size_K, size_V, size_output;
  float *h_Q, *h_K, *h_V, *h_o, *h_golden;
  float *d_Q, *d_K, *d_V, *d_o;

 public:
  AttentionProblem() : M(1024), N(1024), dk(1024)
  {
    d = dv      = dk;  // supposing dv=dk=d
    len_Q       = M * dk;
    len_K       = N * dk;
    len_V       = N * dv;
    len_o       = M * dv;
    size_Q      = len_Q * sizeof(float);
    size_K      = len_K * sizeof(float);
    size_V      = len_V * sizeof(float);
    size_output = len_o * sizeof(float);
  }

  ~AttentionProblem()
  {
    if (d_Q) cudaFree(d_Q);
    if (d_K) cudaFree(d_K);
    if (d_V) cudaFree(d_V);
    if (d_o) cudaFree(d_o);
    if (h_Q) free(h_Q);
    if (h_K) free(h_K);
    if (h_V) free(h_V);
    if (h_o) free(h_o);
    if (h_golden) free(h_golden);
  }

  void init() override
  {
    printf("Init. Attention params: Q(%d, %d), K(%d, %d), V(%d, %d)\n", M, dk, N, dk, N, dv);

    // Allocate host memory
    h_Q      = (float*)malloc(size_Q);
    h_K      = (float*)malloc(size_K);
    h_V      = (float*)malloc(size_V);
    h_o      = (float*)malloc(size_output);
    h_golden = (float*)malloc(size_output);

    // Initialize data with random values
    for (int i = 0; i < len_Q; i++)
      h_Q[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    for (int i = 0; i < len_K; i++)
      h_K[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    for (int i = 0; i < len_V; i++)
      h_V[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    // Allocate device memory
    CHECK(cudaMalloc((void**)&d_Q, size_Q));
    CHECK(cudaMalloc((void**)&d_K, size_K));
    CHECK(cudaMalloc((void**)&d_V, size_V));
    CHECK(cudaMalloc((void**)&d_o, size_output));

    // Copy data from host to device
    CHECK(cudaMemcpy(d_Q, h_Q, size_Q, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_K, h_K, size_K, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_V, h_V, size_V, cudaMemcpyHostToDevice));
  }

  void run() override
  {
    printf("run.\n");
    solve(d_Q, d_K, d_V, d_o, M, N, d);
  }

  void verify() override
  {
    // Copy result back to host
    CHECK(cudaMemcpy(h_o, d_o, size_output, cudaMemcpyDeviceToHost));

    printf("verify. Verifying result on CPU...\n");

    // Compute reference on CPU
    auto gemm =
      [](float* pa, float* pb, float* pc, int dim_M, int dim_N, int dim_K, float scale = 1.0) {
        for (int m = 0; m < dim_M; ++m) {
          for (int n = 0; n < dim_N; ++n) {
            float sum = .0f;
            for (int k = 0; k < dim_K; ++k) {
              sum += pa[m * dim_K + k] * pb[k * dim_N + n];
            }
            pc[m * dim_N + n] = sum / std::sqrt(scale);
          }
        }
      };

    // 1. compute S = Q * K^T
    float* S = (float*)malloc(size_output);
    // scale S by factor sqrt(d_k)
    gemm(h_Q, h_K, S, M, dk, N, dk);

    // 2. compute softmax(S), by row-wise
    float* S_norm = (float*)malloc(size_output);  // normalized S
    for (int m = 0; m < M; ++m) {
      float* pS      = S + m * N;
      float* pS_norm = S_norm + m * N;
      // a) find max
      float _max = -INFINITY;
      for (int i = 0; i < N; ++i)
        _max = std::max(_max, pS[i]);

      // 2. compute sum
      float _sum = .0f;
      for (int j = 0; j < N; ++j) {
        _sum += expf(pS[j] - _max);
      }
      // 3. compute norm
      for (int o = 0; o < N; ++o) {
        pS_norm[o] = expf(pS[o] - _max) / _sum;
      }
    }

    // 3. compute weighted sum with V
    gemm(S_norm, h_V, h_golden, M, N, dv);
    // end golden compute

    // Verify
    double epsilon          = 1.0E-4;  // Relaxed tolerance for Attention accumulation
    unsigned long long errs = 0;
    for (int i = 0; i < len_o; ++i) {
      if (std::abs(h_o[i] - h_golden[i]) > epsilon) ++errs;
      if (errs && errs < 10)
        printf("Test FAIL! Errors = %llu/%ld, result[%d]=%.5f, golden[%d]=%.5f\n\n",
               errs,
               ((long)len_o),
               i,
               h_o[i],
               i,
               h_golden[i]);
    }
    if (errs)
      printf("Test FAIL! Errors = %llu/%ld\n\n", errs, ((long)size_output));
    else
      printf("Test PASS!\n\n");
  }

  long long get_bytes() override
  {
    const int threadsPerBlock = 512;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    long long bytes = 0;
    // 1) S = Q * K^T
    bytes += 1LL * len_Q * sizeof(float);  // Q read
    bytes += 1LL * len_K * sizeof(float);  // K read
    bytes += 1LL * len_o * sizeof(float);  // S write (M*N)

    // 2) Softmax per row
    bytes += 1LL * len_o * sizeof(float);              // S read (bmax/bsum)
    bytes += 1LL * len_o * sizeof(float);              // S_norm write
    bytes += 2LL * M * blocksPerGrid * sizeof(float);  // bmax + bsum
    bytes += 2LL * M * sizeof(float);                  // max + sum

    // 3) O = S_norm * V
    bytes += 1LL * len_o * sizeof(float);  // S_norm read
    bytes += 1LL * len_V * sizeof(float);  // V read
    bytes += 1LL * len_o * sizeof(float);  // output write

    return bytes;
  }

  long long get_flops() override
  {
    // GEMM: 2*M*N*d (mul+add)
    long long flops = 2LL * M * N * d;  // Q * K^T
    flops += 2LL * M * N * d;           // S_norm * V

    // Softmax: per element exp + sub + div, plus reductions
    flops += 3LL * M * N;  // exp/sub/div per element
    flops += 1LL * M * N;  // sum reduction (adds)
    flops += 1LL * M * N;  // max reduction (comparisons)

    return flops;
  }
};

Problem* create_problem() { return new AttentionProblem(); }
