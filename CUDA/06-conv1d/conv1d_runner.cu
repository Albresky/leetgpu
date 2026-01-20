#include <cmath>
#include <iostream>
#include <vector>
#include "../common.h"

extern "C" void solve(
  const float* input, const float* kernel, float* output, int input_size, int kernel_size);

class Conv1DProblem : public Problem {
 private:
  int input_size, kernel_size, output_size;
  size_t size_i, size_k, size_o;
  float *h_i, *h_o, *h_k, *h_golden;
  float *d_i, *d_o, *d_k;

 public:
  Conv1DProblem() : input_size(6), kernel_size(3)
  {
    output_size = input_size - kernel_size + 1;
    size_i      = input_size * sizeof(float);
    size_o      = output_size * sizeof(float);
    size_k      = kernel_size * sizeof(float);
  }

  ~Conv1DProblem()
  {
    if (d_i) cudaFree(d_i);
    if (d_o) cudaFree(d_o);
    if (d_k) cudaFree(d_k);
    if (h_i) free(h_i);
    if (h_k) free(h_k);
    if (h_o) free(h_o);
    if (h_golden) free(h_golden);
  }

  void init() override
  {
    printf(
      "Input size: %d, kernel size: %d, output size: %d\n", input_size, kernel_size, output_size);

    // Allocate host memory
    h_i      = (float*)malloc(size_i);
    h_k      = (float*)malloc(size_k);
    h_o      = (float*)malloc(size_o);
    h_golden = (float*)malloc(size_o);

    // Initialize data with random values
    for (int i = 0; i < input_size; i++)
      h_i[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    for (int i = 0; i < kernel_size; i++)
      h_k[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    // Allocate device memory
    CHECK(cudaMalloc((void**)&d_i, size_i));
    CHECK(cudaMalloc((void**)&d_o, size_o));
    CHECK(cudaMalloc((void**)&d_k, size_k));

    // Copy data from host to device
    CHECK(cudaMemcpy(d_i, h_i, size_i, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_k, h_k, size_k, cudaMemcpyHostToDevice));
  }

  void run() override { solve(d_i, d_k, d_o, input_size, kernel_size); }

  void verify() override
  {
    // Copy result back to host
    CHECK(cudaMemcpy(h_o, d_o, size_o, cudaMemcpyDeviceToHost));

    printf("Verifying result on CPU...\n");
    // Compute reference on CPU
    for (int o = 0; o < output_size; ++o) {
      float sum = 0;
      for (int k = 0; k < kernel_size; k++) {
        sum += h_i[o + k] * h_k[k];
      }
      h_golden[o] = sum;
    }

    // Verify
    double epsilon          = 1.0E-4;  // Relaxed tolerance for Conv1D accumulation
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
    return (long long)(input_size + kernel_size + output_size) * sizeof(float);
  }

  long long get_flops() override
  {
    return 2LL * (kernel_size * output_size);
  }
};

Problem* create_problem() { return new Conv1DProblem(); }
