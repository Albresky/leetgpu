#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include "../common.h"

extern "C" void solve(const float* input,
                      float* output,
                      const float gamma,
                      const float beta,
                      const float eps,
                      const float N);

class LayerNormProblem : public Problem {
 private:
  int input_size, output_size, N;
  size_t size_i, size_o;
  float gamma, beta, eps;
  float *h_i, *h_o, *h_golden;
  float *d_i, *d_o;

  int threadBlocks  = 512;
  int blocksPerGrid = (N + threadBlocks - 1) / threadBlocks;

 public:
  LayerNormProblem() : N(2048), eps(1.0e-5f)
  {
    input_size  = N;
    output_size = input_size;
    size_i      = input_size * sizeof(float);
    size_o      = size_i;
    gamma = beta = 0.0f;
  }

  ~LayerNormProblem()
  {
    if (d_i) cudaFree(d_i);
    if (d_o) cudaFree(d_o);
    if (h_i) free(h_i);
    if (h_o) free(h_o);
    if (h_golden) free(h_golden);
  }

  void init() override
  {
    printf("N: %d, gamma: %.6f, beta: %.6f, eps: %.6f\n", N, gamma, beta, eps);

    // Allocate host memory
    h_i      = (float*)malloc(size_i);
    h_o      = (float*)malloc(size_o);
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
    gamma = dis_gamma(gen);
    beta  = dis_beta(gen);

    // Allocate device memory
    CHECK(cudaMalloc((void**)&d_i, size_i));
    CHECK(cudaMalloc((void**)&d_o, size_o));

    // Copy data from host to device
    CHECK(cudaMemcpy(d_i, h_i, size_i, cudaMemcpyHostToDevice));
  }

  void run() override { solve(d_i, d_o, gamma, beta, eps, N); }

  void verify() override
  {
    // Copy result back to host
    CHECK(cudaMemcpy(h_o, d_o, size_o, cudaMemcpyDeviceToHost));

    printf("Verifying result on CPU...\n");
    // Compute reference on CPU
    // 1. compute mean
    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
      sum += h_i[n];
    }
    float mean = sum / N;

    // 2. compute standard variance
    float sqsum = .0f;
    for (int n = 0; n < N; ++n) {
      sqsum += (h_i[n] - mean) * (h_i[n] - mean);
    }
    float std_var = std::sqrt(sqsum / N + eps);

    // 3. compute norm, scale, and shift
    for (int n = 0; n < N; ++n) {
      float x_norm = (h_i[n] - mean) / std_var;
      float y      = gamma * x_norm + beta;
      h_golden[n]  = y;
    }

    // Verify
    double epsilon          = 1.0E-4;  // Relaxed tolerance for LayerNorm accumulation
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
    long long bytes = 0;
    bytes += 3LL * input_size * sizeof(float);             // input read 3 times
    bytes += 1LL * output_size * sizeof(float);            // output write once
    bytes += 2LL * blocksPerGrid * sizeof(float) + 3 + 2;  // bsum + mean + #sum
    bytes += 2LL * blocksPerGrid * sizeof(float) + 2;      // bvar + std_var
    return bytes;
  }

  long long get_flops() override
  {
    const long long n_bsum = 1LL * (input_size + blocksPerGrid * 2);  // bsum
    const long long n_sum =
      1LL * (blocksPerGrid + /* log(threadBlock) */ threadBlocks);      // reduce, mean, add, sqrt
    const long long n_bvar = 2LL * input_size + 1LL * threadBlocks;     // bvar
    const long long n_var  = 1LL * blocksPerGrid + 1LL * threadBlocks;  // var
    const long long n_norm = 4LL * input_size;                          // norm
    return n_bsum + n_sum + n_bvar + n_var + n_norm;
  }
};

Problem* create_problem() { return new LayerNormProblem(); }
