#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include "../common.h"

extern "C" void solve(const float* input,
                      const float* gamma,
                      const float* beta,
                      float* output,
                      int N,
                      int C,
                      float eps);

class BatchNormProblem : public Problem {
 private:
  int input_size, output_size, N, C;
  size_t size_i, size_o, size_params;
  float eps;
  float *h_i, *h_gamma, *h_beta, *h_o, *h_golden;
  float *d_i, *d_gamma, *d_beta, *d_o;

 public:
  BatchNormProblem() : N(96), C(64), eps(1.0e-5f)
  {
    input_size  = N * C;
    output_size = input_size;
    size_i      = input_size * sizeof(float);
    size_o      = size_i;
    size_params = C * sizeof(float);
  }

  ~BatchNormProblem()
  {
    if (d_i) cudaFree(d_i);
    if (d_o) cudaFree(d_o);
    if (h_i) free(h_i);
    if (h_gamma) free(h_gamma);
    if (h_beta) free(h_beta);
    if (h_o) free(h_o);
    if (h_golden) free(h_golden);
  }

  void init() override
  {
    printf("(B)Batches: %d, (C)channels: %d\n", N, C);

    // Allocate host memory
    h_i      = (float*)malloc(size_i);
    h_o      = (float*)malloc(size_o);
    h_gamma  = (float*)malloc(size_params);
    h_beta   = (float*)malloc(size_params);
    h_golden = (float*)malloc(size_o);

    // Initialize data with random values
    std::random_device rd;
    std::mt19937 gen(rd());

    // 2. Define the distribution range [a, b)
    std::uniform_real_distribution<float> dis_input(-100, 100.0);
    std::uniform_real_distribution<float> dis_gamma(-0.1, 10.0);
    std::uniform_real_distribution<float> dis_beta(-10.0, 10.0);
    for (int i = 0; i < input_size; i++)
      h_i[i] = dis_input(gen);
    for (int i = 0; i < C; i++) {
      h_gamma[i] = dis_gamma(gen);
      h_beta[i]  = dis_beta(gen);
    }

    // Allocate device memory
    CHECK(cudaMalloc((void**)&d_i, size_i));
    CHECK(cudaMalloc((void**)&d_o, size_o));
    CHECK(cudaMalloc((void**)&d_gamma, size_params));
    CHECK(cudaMalloc((void**)&d_beta, size_params));

    // Copy data from host to device
    CHECK(cudaMemcpy(d_i, h_i, size_i, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_gamma, h_gamma, size_params, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_beta, h_beta, size_params, cudaMemcpyHostToDevice));
  }

  void run() override { solve(d_i, d_gamma, d_beta, d_o, N, C, eps); }

  void verify() override
  {
    // Copy result back to host
    CHECK(cudaMemcpy(h_o, d_o, size_o, cudaMemcpyDeviceToHost));

    printf("Verifying result on CPU...\n");
    // Compute reference on CPU
    // 1. Mini-batch Mean
    float h_mean[C];
    for (int c = 0; c < C; ++c) {
      float sum = 0.0f;
      for (int n = 0; n < N; ++n) {
        sum += *(h_i + n * C + c);
      }
      h_mean[c] = sum / N;
    }

    // 2. Mini-batch Mean, standard
    float h_variance[C];
    for (int c = 0; c < C; ++c) {
      float sum = 0.0f;
      for (int n = 0; n < N; ++n) {
        float minus = *(h_i + n * C + c) - *(h_mean + c);
        sum += minus * minus;
      }
      sum /= N;
      h_variance[c] = std::sqrt(sum + eps);
    }

    // 3. Normalization, sacle, and shift
    for (int c = 0; c < C; ++c) {
      for (int n = 0; n < N; ++n) {
        float x                 = *(h_i + n * C + c);
        float m                 = h_mean[c];
        float v                 = h_variance[c];
        float g                 = h_gamma[c];
        float b                 = h_beta[c];
        float x_new             = (x - m) / v;
        *(h_golden + n * C + c) = g * x_new + b;
      }
    }
    // Verify
    double epsilon          = 1.0E-4;  // Relaxed tolerance for BatchNorm accumulation
    unsigned long long errs = 0;
    for (int i = 0; i < output_size; ++i) {
      if (std::abs(h_o[i] - h_golden[i]) > epsilon) ++errs;
      if (errs && errs < 10)
        printf("Test FAIL! Errors = %llu/%ld, result[%d]=%.5f, golden[%d]=%.5f\n\n",
               errs,
               ((long)output_size),
               i,
               h_o[i],
               i,
               h_golden[i]);
    }
    if (errs)
      printf("Test FAIL! Errors = %llu/%ld\n\n", errs, ((long)output_size));
    else
      printf("Test PASS!\n\n");
  }

  long long get_bytes() override
  {
    // Naive softmax (K0) performs multiple full passes:
    // 1) read input for max, 2) read input for sum, 3) read input for output.
    // Additionally, with one output write and intermediate block reductions (bmax/bsum).
    long long bytes = 0;
    bytes += 3LL * input_size * sizeof(float);   // input read 3 times
    bytes += 1LL * output_size * sizeof(float);  // output write once
    bytes += 3LL * size_params;                  // mean
    bytes += 2LL * size_params;                  // gamma
    bytes += 2LL * size_params;
    return bytes;
  }

  long long get_flops() override
  {
    // Rough FLOP estimate (treat exp/fmax/div as 1 op each):
    // - max reduction: (N-1) fmax
    // - sum pass: N sub + N exp + (N-1) add
    // - output pass: N sub + N exp + N div
    const long long n_mean     = input_size;
    const long long n_variance = 5LL * n_mean;  // minus, square, add, minus-magic, sqrt
    const long long n_NSS      = (2 /* minus, divide */ + 2 /* mul, add */) * input_size;
    return n_mean + n_variance + n_NSS;
  }
};

Problem* create_problem() { return new BatchNormProblem(); }
