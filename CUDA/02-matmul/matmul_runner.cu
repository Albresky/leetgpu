#include "../common.h"
#include <vector>
#include <cmath>
#include <iostream>

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K);

class MatMulProblem : public Problem {
private:
    int M, N, K;
    size_t size_A, size_B, size_C;
    float *h_A, *h_B, *h_C, *hostRef;
    float *d_A, *d_B, *d_C;

public:
    MatMulProblem() : M(1024), N(1024), K(1024) {
        size_A = M * N * sizeof(float);
        size_B = N * K * sizeof(float);
        size_C = M * K * sizeof(float);
    }

    ~MatMulProblem() {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        if (h_A) free(h_A);
        if (h_B) free(h_B);
        if (h_C) free(h_C);
        if (hostRef) free(hostRef);
    }

    void init() override {
        printf("Matrix sizes: M=%d, N=%d, K=%d\n", M, N, K);

        // Allocate host memory
        h_A = (float *)malloc(size_A);
        h_B = (float *)malloc(size_B);
        h_C = (float *)malloc(size_C);
        hostRef = (float *)malloc(size_C);

        // Initialize data with random values
        for (int i = 0; i < M * N; i++) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        for (int i = 0; i < N * K; i++) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

        // Allocate device memory
        CHECK(cudaMalloc((void **)&d_A, size_A));
        CHECK(cudaMalloc((void **)&d_B, size_B));
        CHECK(cudaMalloc((void **)&d_C, size_C));

        // Copy data from host to device
        CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    }

    void run() override {
        solve(d_A, d_B, d_C, M, N, K);
    }

    void verify() override {
        // Copy result back to host
        CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

        printf("Verifying result on CPU...\n");
        // Compute reference on CPU
        // C = A * B^T
        // C[m][k] = sum(A[m][n] * B[n][k])
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < K; ++k) {
                float sum = 0.0f;
                for (int n = 0; n < N; ++n) {
                    sum += h_A[m * N + n] * h_B[n * K + k]; // row-major for both A and B
                }
                hostRef[m * K + k] = sum;
            }
        }

        // Verify
        double epsilon = 1.0E-2; // Relaxed tolerance for matmul accumulation
        unsigned long long errs = 0;
        for (int i = 0; i < M * K; i++) {
          if (std::abs(hostRef[i] - h_C[i]) > epsilon)
            ++errs;
        }
        if (errs)
          printf("Test FAIL! Errors = %llu/%ld, result[0][0]=%.5f\n\n", errs, ((long)M) * K, h_C[0]);
        else
          printf("Test PASS!\n\n");
    }

    long long get_bytes() override {
        return (long long)(M * N + N * K + M * K) * sizeof(float);
    }

    long long get_flops() override {
        // 2 * M * N * K (multiply and add)
        return 2LL * M * N * K;
    }
};

Problem* create_problem() {
    return new MatMulProblem();
}
