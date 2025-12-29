#include "../common.h"
#include <vector>
#include <cmath>
#include <iostream>

extern "C" void solve(const float* A, const float* B, float* C, int N);

class VecAddProblem : public Problem {
private:
    int N;
    size_t nBytes;
    float *h_A, *h_B, *h_C, *hostRef;
    float *d_A, *d_B, *d_C;

public:
    VecAddProblem() : N(1 << 24) {
        nBytes = N * sizeof(float);
    }

    ~VecAddProblem() {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        if (h_A) free(h_A);
        if (h_B) free(h_B);
        if (h_C) free(h_C);
        if (hostRef) free(hostRef);
    }

    void init() override {
        printf("Vector size: %d\n", N);

        // alloca host
        h_A = (float *)malloc(nBytes);
        h_B = (float *)malloc(nBytes);
        h_C = (float *)malloc(nBytes);
        hostRef = (float *)malloc(nBytes);

        // init
        for (int i = 0; i < N; i++) {
            h_A[i] = static_cast<float>(i);
            h_B[i] = static_cast<float>(i * 2);
        }

        // alloca device
        CHECK(cudaMalloc((void **)&d_A, nBytes));
        CHECK(cudaMalloc((void **)&d_B, nBytes));
        CHECK(cudaMalloc((void **)&d_C, nBytes));

        // h2d
        CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    }

    void run() override {
        solve(d_A, d_B, d_C, N);
    }

    void verify() override {
        // d2h
        CHECK(cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost));

        // compute golden
        for (int i = 0; i < N; i++) hostRef[i] = h_A[i] + h_B[i];

        // check results
        double epsilon = 1.0E-8;
        bool match = true;
        for (int i = 0; i < N; i++) {
            if (std::abs(hostRef[i] - h_C[i]) > epsilon) {
                match = false;
                printf("Test FAIL!\n");
                printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], h_C[i], i);
                break;
            }
        }
        if (match) printf("Test PASS!\n\n");
    }

    long long get_bytes() override {
        return 3LL * N * sizeof(float);
    }

    long long get_flops() override {
        return N;
    }
};

Problem* create_problem() {
    return new VecAddProblem();
}
