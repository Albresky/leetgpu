#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// Declaration of the solve function.
// Note: This signature must match the one in your problem files (e.g., 01-vecadd/vecadd.cu).
// If you change the signature in the problem, you must update it here.
extern "C" void solve(const float* A, const float* B, float* C, int N);

void checkResult(const float* hostRef, const float* gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = true;
    for (int i = 0; i < N; i++) {
        if (std::abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = false;
            printf("Test FAIL!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match) printf("Test PASS!\n\n");
}

int main(int argc, char **argv) {
    printf("Running global main for %s...\n", argv[0]);

    int N = 1 << 24; // 16M elements
    size_t nBytes = N * sizeof(float);
    printf("Vector size: %d\n", N);

    // Allocate host memory
    float *h_A = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes);
    float *h_C = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, nBytes));
    CHECK(cudaMalloc((void **)&d_B, nBytes));
    CHECK(cudaMalloc((void **)&d_C, nBytes));

    // Copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    constexpr int WARMUP_ITER = 5;
    constexpr int TEST_ITER = 20;
    // Warmup
    printf("Warming up...\n");
    for (int i = 0; i < WARMUP_ITER; ++i) {
        solve(d_A, d_B, d_C, N);
    }
    CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    float total_time = 0.0f;
    float min_time = 1e9f;
    float max_time = 0.0f;

    printf("Running benchmark (%d iterations)...\n", TEST_ITER);

    for (int i = 0; i < TEST_ITER; ++i) {
        float milliseconds = 0;
        CHECK(cudaEventRecord(start));
        solve(d_A, d_B, d_C, N);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        total_time += milliseconds;
        if (milliseconds < min_time) min_time = milliseconds;
        if (milliseconds > max_time) max_time = milliseconds;
    }

    float avg_time = total_time / TEST_ITER;
    
    // Metrics
    // Vector Add: Read A, Read B, Write C -> 3 * N * 4 bytes
    double mem_bandwidth = (3.0 * N * sizeof(float)) / (avg_time * 1e-3) / 1e9; // GB/s
    // Flops: 1 add per element -> N flops
    double gflops = (1.0 * N) / (avg_time * 1e-3) / 1e9; // GFLOPS

    printf("\nPerformance Metrics:\n");
    printf("  Avg Latency: %.4f ms\n", avg_time);
    printf("  Min Latency: %.4f ms\n", min_time);
    printf("  Max Latency: %.4f ms\n", max_time);
    printf("  Throughput:  %.4f GB/s\n", mem_bandwidth);
    printf("  Compute:     %.4f GFLOPS\n", gflops);
    printf("\n");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaGetLastError()); 

    // Copy result back to host
    CHECK(cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost));

    // Verify result
    // Compute reference on host
    for (int i = 0; i < N; i++) hostRef[i] = h_A[i] + h_B[i];
    
    checkResult(hostRef, h_C, N);

    // Free memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(hostRef);

    return 0;
}
