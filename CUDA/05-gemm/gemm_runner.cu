#include "../common.h"
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <iostream>

extern "C" void solve(
  const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta);

class GEMMProblem : public Problem {
private:
    int M, N, K;
    float alpha, beta;
    size_t size_A, size_B, size_C;
    __half *h_A, *h_B, *h_C, *hostRef;
    half *d_A, *d_B, *d_C;

public:
    GEMMProblem() : M(1024), K(1024), N(1024), alpha(1.0), beta(1.0) {
        size_A = M * K * sizeof(__half);
        size_B = K * N * sizeof(__half);
        size_C = M * N * sizeof(__half);
    }

    ~GEMMProblem() {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        if (h_A) free(h_A);
        if (h_B) free(h_B);
        if (h_C) free(h_C);
        if (hostRef) free(hostRef);
    }

    void init() override {
        printf("Matrix info: M=%d, N=%d, K=%d, alpha=%.6f, beta=%.6f\n", M, N, K, alpha, beta);

        // Allocate host memory
        h_A = (__half *)malloc(size_A);
        h_B = (__half *)malloc(size_B);
        h_C = (__half *)malloc(size_C);
        hostRef = (__half *)malloc(size_C);

        // Initialize data with random values
        for (int i = 0; i < M * K; i++) h_A[i] = static_cast<__half>(rand()) / (__half)RAND_MAX;
        for (int i = 0; i < K * N; i++) h_B[i] = static_cast<__half>(rand()) / (__half)RAND_MAX;

        // Allocate device memory
        CHECK(cudaMalloc((void **)&d_A, size_A));
        CHECK(cudaMalloc((void **)&d_B, size_B));
        CHECK(cudaMalloc((void **)&d_C, size_C));

        // Copy data from host to device
        CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    }

    void run() override {
        solve(d_A, d_B, d_C, M, N, K, alpha, beta);
    }

    void verify() override {
        // Copy result back to host
        CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

        printf("Verifying result on CPU...\n");
        // Compute reference on CPU
        // C = alpha * A * B + beta * C
        for (int m = 0; m < M; ++m) {
          for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
              sum += (float)h_A[m * K + k] * (float)h_B[k * N + n];  // row-major for both A and B
            }
            float c            = hostRef[m * N + n];
            c                  = alpha * sum + beta * c;
            hostRef[m * N + n] = (__half)c;
          }
        }

        // Verify
        double epsilon = 1.0E-2; // Relaxed tolerance for matmul accumulation
        unsigned long long errs = 0;
        for (int i = 0; i < M * N; i++) {
          if (std::abs(__half2float(hostRef[i] - h_C[i])) > epsilon)
            ++errs;
        }
        if (errs)
          printf("Test FAIL! Errors = %llu/%ld, result[0][0]=%.4f\n\n", errs, ((long)M) * K, __half2float(h_C[0]));
        else
          printf("Test PASS!\n\n");
    }

    long long get_bytes() override {
        return (long long)(M * K + K * N + M * N) * sizeof(__half);
    }

    long long get_flops() override {
        // 2 * M * N * K (multiply and add)
        return 2LL * M * N * K;
    }
};

Problem* create_problem() {
    return new GEMMProblem();
}
