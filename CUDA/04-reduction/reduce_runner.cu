#include "../common.h"
#include <vector>
#include <cmath>
#include <iostream>

extern "C" void solve(const float* input, float* output, int N);

class ReduceProblem : public Problem {
private:
    int length;
    size_t size_vec;
    size_t size_val;
    float *h_vec,*h_val, *hostRef;
    float *d_vec, *d_val;

public:
    ReduceProblem() : length(2048) {
        size_vec = length * sizeof(float);
        size_val = 1 * sizeof(float);
    }

    ~ReduceProblem() {
        if (d_vec) cudaFree(d_vec);
        if (d_val) cudaFree(d_val);
        if (h_vec) free(h_vec);
        if (hostRef) free(hostRef);
    }

    void init() override {
        printf("Vector sizes: length=%d\n", length);

        // Allocate host memory
        h_vec = (float *)malloc(size_vec);
        h_val = (float *)malloc(size_val);
        hostRef = (float *)malloc(size_val);

        // Initialize data with random values
        for (int i = 0; i < length; ++i) h_vec[i] = static_cast<float>(rand()) / RAND_MAX;

        // Allocate device memory
        CHECK(cudaMalloc((void **)&d_vec, size_vec));
        CHECK(cudaMalloc((void **)&d_val, size_val));

        // Copy data from host to device
        CHECK(cudaMemcpy(d_vec, h_vec, size_vec, cudaMemcpyHostToDevice));
    }

    void run() override {
        solve(d_vec, d_val, length);
    }

    void verify() override {
        // Copy result back to host
        CHECK(cudaMemcpy(h_val, d_val, size_val, cudaMemcpyDeviceToHost));

        printf("Verifying result on CPU...\n");
        // Compute reference on CPU
        float sum = 0.0f;
        for (int l = 0; l < length; ++l) {
          sum += h_vec[l];
        }
        *hostRef = sum;

        // Verify
        double epsilon = 1.0E-2;
        unsigned long long errs = 0;
        if (std::abs(*hostRef - *h_val) > epsilon)
            ++errs;
        if (errs)
          printf("Test FAIL! result=%.5f, golden=%0.5f\n\n", *h_val, *hostRef);
        else
          printf("Test PASS!\n\n");
    }

    long long get_bytes() override {
        return 2LL * length * sizeof(float);
    }

    long long get_flops() override {
        return 1L * length;
    }
};

Problem* create_problem() {
    return new ReduceProblem();
}
