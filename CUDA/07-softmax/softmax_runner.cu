#include <cmath>
#include <iostream>
#include <vector>
#include "../common.h"

extern "C" void solve(const float* input, float* output, int N);

#define __ONLINE_SOFTMAX__

class SoftmaxProblem : public Problem {
 private:
  int input_size, output_size, N;
  size_t size_i, size_o;
  float maxv = -INFINITY;
  float sum  = .0f;
  float *h_i, *h_o, *h_golden;
  float *d_i, *d_o;

 public:
  SoftmaxProblem() : input_size(16384)
  {
    output_size = input_size;
    size_i      = input_size * sizeof(float);
    size_o      = size_i;
    N           = input_size;
  }

  ~SoftmaxProblem()
  {
    if (d_i) cudaFree(d_i);
    if (d_o) cudaFree(d_o);
    if (h_i) free(h_i);
    if (h_o) free(h_o);
    if (h_golden) free(h_golden);
  }

  void init() override
  {
    printf("Input size: %d, output size: %d\n", input_size, output_size);

    // Allocate host memory
    h_i      = (float*)malloc(size_i);
    h_o      = (float*)malloc(size_o);
    h_golden = (float*)malloc(size_o);

    // Initialize data with random values
    for (int i = 0; i < input_size; i++)
      h_i[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    // Allocate device memory
    CHECK(cudaMalloc((void**)&d_i, size_i));
    CHECK(cudaMalloc((void**)&d_o, size_o));

    // Copy data from host to device
    CHECK(cudaMemcpy(d_i, h_i, size_i, cudaMemcpyHostToDevice));
  }

  void run() override { solve(d_i, d_o, N); }

  void verify() override
  {
    // Copy result back to host
    CHECK(cudaMemcpy(h_o, d_o, size_o, cudaMemcpyDeviceToHost));

    printf("Verifying result on CPU...\n");
    // Compute reference on CPU
    // 1. find max
    for (int i = 0; i < input_size; ++i) {
      maxv = std::max(maxv, h_i[i]);
    }
    // 2. compute sum
    for (int j = 0; j < input_size; ++j) {
      sum += expf(h_i[j] - maxv);
    }
    // 3. compute golden
    for (int o = 0; o < output_size; ++o) {
      h_golden[o] = expf(h_i[o] - maxv) / sum;
    }

    // Verify
    double epsilon          = 1.0E-4;  // Relaxed tolerance for Softmax accumulation
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
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    long long bytes = 0;
#ifndef __ONLINE_SOFTMAX__
    // Naive softmax (K0) performs 3 passes:
    // 1) read input for max
    // 2) read input for sum
    // 3) read input for output.
    // Additionally, with one output write and intermediate block reductions (bmax/bsum).
    bytes += 3LL * input_size * sizeof(float);     // input read 3 times
    bytes += 1LL * output_size * sizeof(float);    // output write once
    bytes += 2LL * blocksPerGrid * sizeof(float);  // bmax write + read
    bytes += 2LL * blocksPerGrid * sizeof(float);  // bsum write + read
    bytes += 2LL * sizeof(float);                  // max + sum scalars
#else
    // Online softmax performs 2 passes over input:
    // 1. Online reduce: read input, write bmax/bsum
    // 2. Apply: read input, read global max/sum, write output
    // Plus intermediate global reduce of bmax/bsum
    int effective_blocks = blocksPerGrid;
    if (effective_blocks > 2048) effective_blocks = 2048;

    bytes += 2LL * input_size * sizeof(float);        // input read 2 times
    bytes += 1LL * output_size * sizeof(float);       // output write once
    bytes += 2LL * effective_blocks * sizeof(float);  // bmax/bsum write
    bytes += 2LL * effective_blocks * sizeof(float);  // bmax/bsum read
    bytes += 2LL * sizeof(float);                     // global max/sum write
#endif
    return bytes;
  }

  long long get_flops() override
  {
#ifndef __ONLINE_SOFTMAX__
    // Rough FLOP estimate (treat exp/fmax/div as 1 op each):
    // - max reduction: (N-1) fmax
    // - sum pass: N sub + N exp + (N-1) add
    // - output pass: N sub + N exp + N div
    const long long n          = input_size;
    const long long reduce_max = (n > 0) ? (n - 1) : 0;
    const long long reduce_sum = (n > 0) ? (n - 1) : 0;
    const long long per_elem   = 2 /*sub*/ + 2 /*exp*/ + 1 /*div*/;
    return n * per_elem + reduce_max + reduce_sum;
#else
    // Online Softmax FLOPs:
    // Pass 1 (Reduce): Combine per element (assuming b.d=1 optimization)
    // Combine: max(1) + 2*sub + 2*exp + 1*mul + 1*add = 7 ops
    const long long reduce = 1 + 2 + 2 + 1 + 1;
    // Pass 3 (Apply): sub(1) + exp(1) + div(1) = 3 ops
    const long long norm = 1 + 1 + 1;
    
    // Total: ~10 ops per element
    return (reduce + norm) * input_size;
#endif
  }
};

Problem* create_problem() { return new SoftmaxProblem(); }
