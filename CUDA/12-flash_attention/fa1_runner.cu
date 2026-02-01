#include <cmath>
#include <cstdlib>
#include "../common.h"

extern "C" void solve(
  const float* Q, const float* K, const float* V, float* output, int M, int N, int d);

class FA1Problem : public Problem {
 private:
  int M, N, d;
  int len_Q, len_K, len_V, len_o;
  size_t size_Q, size_K, size_V, size_output;
  float *h_Q, *h_K, *h_V, *h_o, *h_golden;
  float *d_Q, *d_K, *d_V, *d_o;

 public:
  FA1Problem() : M(1024), N(1024), d(128)
  {
    len_Q       = M * d;
    len_K       = N * d;
    len_V       = N * d;
    len_o       = M * d;
    size_Q      = len_Q * sizeof(float);
    size_K      = len_K * sizeof(float);
    size_V      = len_V * sizeof(float);
    size_output = len_o * sizeof(float);
  }

  ~FA1Problem()
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
    printf("Init. FlashAttention-V1 params: Q(%d, %d), K(%d, %d), V(%d, %d)\n", M, d, N, d, N, d);

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
    printf("Run FlashAttention V1 Kernel.\n");
    solve(d_Q, d_K, d_V, d_o, M, N, d);
  }

  void verify() override
  {
    // Copy result back to host
    CHECK(cudaMemcpy(h_o, d_o, size_output, cudaMemcpyDeviceToHost));

    printf("Verifying result on CPU (Naive Attention)...\n");

    // ----------------------------------------------------------------
    // CPU Golden Reference Calculation
    // ----------------------------------------------------------------

    // 1. S = Q * K^T (Scaled)
    std::vector<float> S(M * N);
    float scale = 1.0f / std::sqrt(static_cast<float>(d));

    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float val = 0.0f;
        for (int k = 0; k < d; ++k) {
          val += h_Q[m * d + k] * h_K[n * d + k];
        }
        S[m * N + n] = val * scale;
      }
    }

    // 2. P = Softmax(S) row-wise
    std::vector<float> P(M * N);
    for (int m = 0; m < M; ++m) {
      // findmax
      float row_max = -1e9;
      for (int n = 0; n < N; ++n) {
        row_max = std::fmax(row_max, S[m * N + n]);
      }

      // exp and sum
      float row_sum = 0.0f;
      for (int n = 0; n < N; ++n) {
        P[m * N + n] = std::exp(S[m * N + n] - row_max);
        row_sum += P[m * N + n];
      }

      // norm
      for (int n = 0; n < N; ++n) {
        P[m * N + n] /= row_sum;
      }
    }

    // 3. O = P * V
    for (int m = 0; m < M; ++m) {
      for (int x = 0; x < d; ++x) {
        float val = 0.0f;
        for (int n = 0; n < N; ++n) {
          val += P[m * N + n] * h_V[n * d + x];
        }
        h_golden[m * d + x] = val;
      }
    }

    // ----------------------------------------------------------------
    // Compare
    // ----------------------------------------------------------------
    double epsilon          = 1.0E-3;
    unsigned long long errs = 0;
    for (int i = 0; i < len_o; ++i) {
      if (std::abs(h_o[i] - h_golden[i]) > epsilon) {
        errs++;
        if (errs < 10) {
          printf("FAIL at idx [%d]: GPU=%.5f, CPU=%.5f, Diff=%.5f\n",
                 i,
                 h_o[i],
                 h_golden[i],
                 std::abs(h_o[i] - h_golden[i]));
        }
      }
    }

    if (errs)
      printf("Test FAIL! Total Errors = %llu / %d\n\n", errs, len_o);
    else
      printf("Test PASS!\n\n");
  }

  long long get_bytes() override
  {
    // FlashAttention Effective Bandwidth Calculation
    // We only count compulsory reads (Q, K, V) and writes (O).
    // Intermediate S and P matrices are never materialized in global memory.
    return sizeof(float) * (len_Q + len_K + len_V + len_o);
  }

  long long get_flops() override
  {
    // Standard Attention FLOPs
    // 1. MatMul Q*K^T: 2 * M * N * d
    // 2. MatMul P*V:   2 * M * N * d
    // (Softmax flops are negligible O(MN) vs O(MNd))
    return 4LL * M * N * d;
  }
};

Problem* create_problem() { return new FA1Problem(); }