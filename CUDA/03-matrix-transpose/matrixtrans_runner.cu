#include "../common.h"
#include <vector>
#include <cmath>
#include <iostream>

extern "C" void solve(const float* input, float* output, int rows, int cols);

class MatTransProblem : public Problem {
private:
    int rows, cols;
    size_t size_;
    float *h_mat,*h_matT, *hostRef;
    float *d_mat, *d_matT;

public:
    MatTransProblem() : rows(16384), cols(16384) {
        size_ = rows * cols * sizeof(float);
    }

    ~MatTransProblem() {
        if (d_mat) cudaFree(d_mat);
        if (d_matT) cudaFree(d_matT);
        if (h_mat) free(h_mat);
        if (hostRef) free(hostRef);
    }

    void init() override {
        printf("Matrix sizes: rows=%d, cols=%d\n", rows, cols);

        // Allocate host memory
        h_mat = (float *)malloc(size_);
        h_matT = (float *)malloc(size_);
        hostRef = (float *)malloc(size_);

        // Initialize data with random values
        for (int i = 0; i < rows * cols; i++) h_mat[i] = static_cast<float>(rand()) / RAND_MAX;

        // Allocate device memory
        CHECK(cudaMalloc((void **)&d_mat, size_));
        CHECK(cudaMalloc((void **)&d_matT, size_));

        // Copy data from host to device
        CHECK(cudaMemcpy(d_mat, h_mat, size_, cudaMemcpyHostToDevice));
    }

    void run() override {
        solve(d_mat, d_matT, rows, cols);
    }

    void verify() override {
        // Copy result back to host
        CHECK(cudaMemcpy(h_matT, d_matT, size_, cudaMemcpyDeviceToHost));

        printf("Verifying result on CPU...\n");
        // Compute reference on CPU
        for (int m = 0; m < rows; ++m) {
            for (int n = 0; n < cols; ++n) {
                hostRef[n * rows + m] = h_mat[m*cols+n];
            }
        }

        // Verify
        double epsilon = 0.0;
        unsigned long long errs = 0;
        for (int i = 0; i < rows * cols; i++) {
          if (std::abs(hostRef[i] - h_matT[i]) > epsilon)
            ++errs;
        }
        if (errs)
          printf("Test FAIL! Errors = %llu/%ld, result[0][1]=%.5f, golden[0][1]=%0.5f\n\n", errs, ((long)rows) * cols, h_matT[1], hostRef[1]);
        else
          printf("Test PASS!\n\n");
    }

    long long get_bytes() override {
        return 2LL * rows * cols * sizeof(float);
    }

    long long get_flops() override {
        return 0;
    }
};

Problem* create_problem() {
    return new MatTransProblem();
}
